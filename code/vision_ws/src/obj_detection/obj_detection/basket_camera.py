"""
basket_camera.py
=================
바스켓 4개(B1~B4)를 내려다보도록 고정 설치된 일반 USB 웹캠(예: 로지텍 C270,
/dev/videoN)으로 각 바스켓에 사과가 있는지 없는지를 판단해서 `/basket_status`
토픽으로 계속 알려주는 노드 (basket_detection_node).

픽 카메라(realsense.py/detection.py)는 depth까지 쓰는 RealSense라 ROS 토픽으로
구독하지만, 이 카메라는 "바스켓 안에 뭐라도 있는지"만 컬러 프레임으로 빠르게
판단하면 되므로 별도의 카메라 드라이버 노드 없이 cv2.VideoCapture로 이 노드
안에서 직접 연다 (RealSense가 아닌 일반 V4L2 웹캠이라 realsense2_camera로
publish할 수도 없음).

파이프라인 (_publish_status 기준):
    1) YOLO(yolo.AppleStatusModel, obj_detection과 동일 체크포인트 재사용 -
       apple_normal/apple_rotten/apple_damaged/basket 클래스가 이미 다 있음)로
       현재 프레임의 모든 후보 박스를 가져온다.
    2) label이 "basket" 또는 "full_basket"인 박스만 추려서 좌상단 좌표(y1, x1) 오름차순으로
       정렬한다 - 실제 카메라 구도상 바스켓 4개가 위에서 아래로 배치되므로
       화면 맨 위가 b1, 맨 아래가 b4가 되도록 y(세로) 우선 정렬, 같은 높이면
       x(가로)로 tie-break. 4개보다 적게 잡히면 잡힌 만큼만 채우고
       valid=False로 표시한다.
    3) apple_normal/apple_rotten/apple_damaged 박스의 중심점이 어떤 바스켓
       bbox 안에 들어오면 그 바스켓을 occupied=True로 표시한다.
    4) {"b1":bool, "b2":bool, "b3":bool, "b4":bool, "valid":bool,
       "timestamp":float} JSON을 std_msgs/String으로 `/basket_status`에
       publish한다 (기존 /decision/result, /robot/command와 동일하게
       String+JSON 패턴을 그대로 따름 - 커스텀 topic msg를 새로 만들지 않음).

카메라는 로봇 팔에 달린 게 아니라 바스켓 전체가 잘 보이는 곳에 고정
설치되는 별도의 물리 카메라이므로, prepare_camera.py처럼 로봇을 움직여
포지셔닝할 필요가 없다. 어떤 /dev/videoN을 열지는 ROS 파라미터
`device_index`로 지정한다 (`v4l2-ctl --list-devices`로 실제 장치/인덱스
확인 - basket_camera_live_test.py의 --device_index와 동일한 값을 쓰면 됨).
"""

import json
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from obj_detection.ros_image_utils import cv2_to_imgmsg
from obj_detection.yolo import AppleStatusModel

# 바스켓 점유 판정을 몇 초마다 다시 계산해서 publish할지.
# 메인 픽 카메라(debug_overlay.DEBUG_IMAGE_PERIOD_SEC)와 동일한 주기로 맞춤.
STATUS_PUBLISH_PERIOD_SEC = 0.05

# 관리하는 바스켓 개수(B1~B4 고정).
BASKET_NAMES = ["b1", "b2", "b3", "b4"]

# 디버그 오버레이에 b1~b4와 함께 보여줄 표시 이름 (사용자 지정값).
# /basket_status로 나가는 JSON 키(b1..b4)는 로봇/백엔드가 그대로 쓰고 있으므로
# 여기서는 건드리지 않고, 디버그 오버레이 표시에만 쓴다.
#
# 실측으로 확인된 문제: 이 표시 이름이 실제 물리 배치(box_sequence_test.py의
# DESTINATION_TO_BOX, "b1=가공용, b2=못난이, b3=정상, b4=폐기")와 어긋나 있었음
# (예전 설계 당시 이름이 그대로 남아있던 것으로 보임) - b2는 시스템 전체에서
# 항상 "못난이(ugly_box)"를 의미하므로 여기서도 그에 맞춰 통일한다.
BASKET_DISPLAY_NAMES = {
    "b1": "processing",
    "b2": "ugly",
    "b3": "normal",
    "b4": "discard",
}

APPLE_LABELS = {"apple_normal", "apple_rotten", "apple_damaged"}

# 로지텍 C270 등 UVC 웹캠은 장치가 실제로 지원하는 해상도가 정해져 있는데
# (`v4l2-ctl --list-devices`로 확인 가능, C270은 최대 1280x720/1280x960),
# cv2.VideoCapture를 해상도 지정 없이 열면 OpenCV/V4L2가 카메라가 실제로
# 못 내보내는 해상도(예: 1920x1080)로 협상해버려서 ok=True인데 프레임 전체가
# 0으로 채워진 완전 검은 화면만 나오는 문제가 실측으로 확인됨. 그래서 오픈
# 직후 반드시 지원 해상도로 명시적으로 맞춘다.
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720

# 웹캠 프레임 읽기가 이 횟수만큼 연속으로 실패하면(카메라 분리 등), 마지막
# 성공 프레임을 계속 붙들지 않고 비워서 오래된 프레임이 실시간처럼 재발행되는
# 것을 막는다. (0.1초 간격 재시도이므로 10회 = 약 1초)
STALE_FRAME_FAILURE_THRESHOLD = 10


class WebcamCapture:
    """일반 V4L2 웹캠을 백그라운드 스레드에서 계속 읽어 최신 프레임만 들고 있는 헬퍼.

    realsense.py의 ImgNode(콜백 기반 구독)와 달리 cv2.VideoCapture는 자체적으로
    스트림을 안 밀어주고 read()를 호출해야 하므로, 이 스레드가 계속 read()를
    돌며 최신 프레임을 갱신해둔다 - 그래야 _publish_status 타이머가 항상 최신
    프레임을 즉시 가져다 쓸 수 있다(느린 read()가 타이머 콜백을 막지 않음)."""

    def __init__(
        self, device_index: int, logger,
        frame_width: int = DEFAULT_FRAME_WIDTH, frame_height: int = DEFAULT_FRAME_HEIGHT,
    ):
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"/dev/video{device_index}를 열 수 없습니다. "
                "'v4l2-ctl --list-devices'로 장치/인덱스를 확인하세요."
            )

        # 카메라가 지원하지 않는 해상도로 열리면 검은 화면만 나오므로(위 설명),
        # 반드시 지원되는 해상도로 명시 설정. 요청한 값을 카메라가 그대로 못
        # 받아들이면 드라이버가 가장 가까운 지원 해상도로 대체하므로, 실제로
        # 뭐가 잡혔는지 cap.get()으로 다시 읽어 로그로 남긴다.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._logger = logger
        self._frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._logger.info(
            f"웹캠 오픈 완료: /dev/video{device_index} "
            f"(요청 해상도 {frame_width}x{frame_height}, 실제 {actual_w}x{actual_h})"
        )

    def _capture_loop(self):
        consecutive_failures = 0
        while not self._stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                consecutive_failures += 1
                self._logger.warn("웹캠 프레임 읽기 실패 - 재시도")
                # 읽기 실패가 계속되면(카메라 분리/재연결 등), 마지막으로 성공했던
                # 오래된 프레임을 붙들고 있지 않도록 비워서 "새 프레임 없음"을
                # 정직하게 알린다 (get_color_frame()이 None을 반환 -> _publish_status
                # 가 옛날 프레임을 계속 재사용/재발행하는 것을 막음).
                if consecutive_failures >= STALE_FRAME_FAILURE_THRESHOLD:
                    with self._lock:
                        self._frame = None
                time.sleep(0.1)
                continue
            consecutive_failures = 0
            with self._lock:
                self._frame = frame

    def get_color_frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


def _box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


BASKET_LABELS = {"basket", "full_basket"}


def assign_baskets(detections):
    """YOLO 전체 후보 목록에서 basket 박스만 골라 좌상단(y1,x1) 오름차순으로
    정렬한 뒤 b1..b4에 순서대로 배정한다 (화면 맨 위=b1 ... 맨 아래=b4).

    바스켓이 가득 차면 모델이 "basket" 대신 "full_basket"으로 분류하므로,
    바스켓 위치 자체를 놓치지 않도록 두 라벨을 모두 바스켓 박스로 취급한다.

    반환값: {"b1": box_or_None, "b2": ..., "b3": ..., "b4": ...},
    {"b1": label_or_None, ...} (바스켓 자신이 "basket"/"full_basket" 중 어느 쪽으로
    잡혔는지), 그리고 실제로 4개가 다 잡혔는지(valid) 여부.
    """
    basket_detections = sorted(
        ((label, box) for label, _score, box in detections if label in BASKET_LABELS),
        key=lambda item: (item[1][1], item[1][0]),
    )
    assignment = {name: None for name in BASKET_NAMES}
    basket_labels = {name: None for name in BASKET_NAMES}
    for name, (label, box) in zip(BASKET_NAMES, basket_detections):
        assignment[name] = box
        basket_labels[name] = label
    valid = len(basket_detections) >= len(BASKET_NAMES)
    return assignment, valid, basket_labels


def compute_occupancy(detections, basket_assignment, basket_labels):
    """바스켓 점유 여부 판단.

    모델이 바스켓 자체를 "full_basket"으로 분류했다면 그 신호를 그대로 신뢰한다
    (바스켓이 사과로 가득 차서 개별 사과 박스가 서로 가려 검출이 안 되는
    경우에도 놓치지 않기 위함). 그 외에는 기존처럼 바스켓 bbox 안에 사과
    클래스 검출의 중심점이 들어오는지로 판단한다.
    """
    apple_centers = [
        _box_center(box) for label, _score, box in detections if label in APPLE_LABELS
    ]
    occupancy = {}
    for name, basket_box in basket_assignment.items():
        if basket_box is None:
            occupancy[name] = False
            continue
        if basket_labels.get(name) == "full_basket":
            occupancy[name] = True
            continue
        occupancy[name] = any(
            _point_in_box(cx, cy, basket_box) for cx, cy in apple_centers
        )
    return occupancy


def draw_basket_overlay(frame, basket_assignment, occupancy):
    """바스켓 bbox + b1~b4 라벨 + 점유 여부를 그린 프레임을 반환 (원본은 건드리지 않음).

    BasketDetectionNode(_publish_status)와 ROS 없이 실시간으로 로직만 확인하는
    basket_camera_live_test.py가 동일한 시각화를 공유하기 위해 분리했다.
    """
    annotated = frame.copy()
    for name, box in basket_assignment.items():
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = (0, 0, 255) if occupancy[name] else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        display_name = BASKET_DISPLAY_NAMES.get(name, name)
        label = f"{name}({display_name}) {'APPLE' if occupancy[name] else 'EMPTY'}"
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    return annotated


class BasketDetectionNode(Node):
    """바스켓 점유 상태를 주기적으로 판정해서 /basket_status로 publish하는 노드."""

    def __init__(self):
        super().__init__('basket_detection_node')

        # 기본값은 실제 바스켓용 웹캠(C270)의 /dev/videoN 번호로 맞춰둔다.
        # (v4l2-ctl --list-devices로 확인 - 0번은 v4l2loopback 가상 장치라 실제
        # 카메라가 아니므로 여기 쓰면 안 됨). USB 포트/부팅 순서가 바뀌면
        # 번호가 달라질 수 있으니, 다르면 실행 시 --ros-args -p device_index:=N
        # 으로 덮어쓸 것.
        self.declare_parameter('device_index', 39)
        self.declare_parameter('frame_width', DEFAULT_FRAME_WIDTH)
        self.declare_parameter('frame_height', DEFAULT_FRAME_HEIGHT)
        device_index = self.get_parameter('device_index').get_parameter_value().integer_value
        frame_width = self.get_parameter('frame_width').get_parameter_value().integer_value
        frame_height = self.get_parameter('frame_height').get_parameter_value().integer_value

        self.webcam = WebcamCapture(device_index, self.get_logger(), frame_width, frame_height)
        self.model = AppleStatusModel()

        self.status_pub = self.create_publisher(String, 'basket_status', 10)
        self.debug_image_pub = self.create_publisher(Image, 'basket_camera/debug_image', 10)

        self.create_timer(STATUS_PUBLISH_PERIOD_SEC, self._publish_status)

        self.get_logger().info(
            f"BasketDetectionNode initialized (device_index={device_index})"
        )

    def _publish_status(self):
        frame = self.webcam.get_color_frame()
        if frame is None:
            return

        detections = self.model.get_all_detections(frame)
        basket_assignment, valid, basket_labels = assign_baskets(detections)
        occupancy = compute_occupancy(detections, basket_assignment, basket_labels)

        if not valid:
            self.get_logger().warn(
                f"바스켓 4개를 전부 찾지 못함 - 감지된 개수: "
                f"{sum(1 for b in basket_assignment.values() if b is not None)}/4"
            )

        payload = {name: occupancy[name] for name in BASKET_NAMES}
        payload["valid"] = valid
        payload["timestamp"] = time.time()

        self.status_pub.publish(String(data=json.dumps(payload)))
        annotated = draw_basket_overlay(frame, basket_assignment, occupancy)
        self.debug_image_pub.publish(cv2_to_imgmsg(annotated, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = BasketDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.webcam.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
