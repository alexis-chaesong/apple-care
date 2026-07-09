"""
basket_camera.py
=================
바스켓 4개(B1~B4)를 내려다보도록 고정 설치된 두 번째 RealSense 카메라로
각 바스켓에 사과가 있는지 없는지를 판단해서 `/basket_status` 토픽으로
계속 알려주는 노드 (basket_camera_node).

detection.py의 사과 1개 상태 판정(get_apple_status 서비스, ~1초 소요, depth
검증 포함)과는 목적이 다르다 - 여기는 "바스켓 안에 뭐라도 있는지"만 빠르게,
주기적으로 판단하면 되므로 depth 검증 없이 YOLO 컬러 프레임만 쓴다.

파이프라인 (_publish_status 기준):
    1) YOLO(yolo.AppleStatusModel, obj_detection과 동일 체크포인트 재사용 -
       apple_normal/apple_rotten/apple_damaged/basket 클래스가 이미 다 있음)로
       현재 프레임의 모든 후보 박스를 가져온다.
    2) label == "basket"인 박스만 추려서 좌상단 좌표(x1, y1) 오름차순으로
       정렬한다 - 사용자 요구사항: "좌상단에 가까우면 b1, 그 다음 b2, b3,
       가장 오른쪽이 b4" -> 화면 왼쪽부터 순서대로 b1..b4로 배정.
       4개보다 적게 잡히면 잡힌 만큼만 채우고 valid=False로 표시한다.
    3) apple_normal/apple_rotten/apple_damaged 박스의 중심점이 어떤 바스켓
       bbox 안에 들어오면 그 바스켓을 occupied=True로 표시한다.
    4) {"b1":bool, "b2":bool, "b3":bool, "b4":bool, "valid":bool,
       "timestamp":float} JSON을 std_msgs/String으로 `/basket_status`에
       publish한다 (기존 /decision/result, /robot/command와 동일하게
       String+JSON 패턴을 그대로 따름 - 커스텀 topic msg를 새로 만들지 않음).

카메라는 로봇 팔에 달린 게 아니라 바스켓 전체가 잘 보이는 곳에 고정
설치되는 별도의 물리 RealSense이므로, prepare_camera.py처럼 로봇을 움직여
포지셔닝할 필요가 없다. 토픽 네임스페이스만 obj_detection이 쓰는 기본
카메라(/camera/camera/...)와 겹치지 않게 파라미터로 분리한다
(기본값 /camera2/camera - basket_camera.launch.py가 실제 RealSense 노드를
camera_name:=camera2로 띄우는 것과 짝을 맞춤).
"""

import json
import time

import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from obj_detection.ros_image_utils import imgmsg_to_cv2, cv2_to_imgmsg
from obj_detection.yolo import AppleStatusModel

# 바스켓 점유 판정을 몇 초마다 다시 계산해서 publish할지.
STATUS_PUBLISH_PERIOD_SEC = 0.5

# 관리하는 바스켓 개수(B1~B4 고정).
BASKET_NAMES = ["b1", "b2", "b3", "b4"]

APPLE_LABELS = {"apple_normal", "apple_rotten", "apple_damaged"}


class BasketImgNode(Node):
    """세컨 카메라(바스켓 조망용) 컬러 프레임만 구독하는 노드.

    바스켓 점유 판단은 YOLO bbox 포함관계만으로 충분해서 depth는 구독하지
    않는다 (realsense.py의 ImgNode와 달리 depth/camera_info 불필요)."""

    def __init__(self, topic_prefix: str):
        super().__init__('basket_img_node')
        self.color_frame = None
        self.color_subscription = self.create_subscription(
            Image, f'{topic_prefix}/color/image_raw', self.color_callback, 10)
        self.get_logger().info(
            f"Waiting for basket camera topic: {topic_prefix}/color/image_raw"
        )

    def color_callback(self, msg):
        self.color_frame = imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def get_color_frame(self):
        return self.color_frame


def _box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def assign_baskets(detections):
    """YOLO 전체 후보 목록에서 basket 박스만 골라 좌상단(x1,y1) 오름차순으로
    정렬한 뒤 b1..b4에 순서대로 배정한다.

    반환값: {"b1": box_or_None, "b2": ..., "b3": ..., "b4": ...}, 그리고
    실제로 4개가 다 잡혔는지(valid) 여부.
    """
    basket_boxes = sorted(
        (box for label, _score, box in detections if label == "basket"),
        key=lambda box: (box[0], box[1]),
    )
    assignment = {name: None for name in BASKET_NAMES}
    for name, box in zip(BASKET_NAMES, basket_boxes):
        assignment[name] = box
    valid = len(basket_boxes) >= len(BASKET_NAMES)
    return assignment, valid


def compute_occupancy(detections, basket_assignment):
    """각 바스켓 bbox 안에 사과 클래스 검출의 중심점이 들어오는지로 점유 여부 판단."""
    apple_centers = [
        _box_center(box) for label, _score, box in detections if label in APPLE_LABELS
    ]
    occupancy = {}
    for name, basket_box in basket_assignment.items():
        if basket_box is None:
            occupancy[name] = False
            continue
        occupancy[name] = any(
            _point_in_box(cx, cy, basket_box) for cx, cy in apple_centers
        )
    return occupancy


def draw_basket_overlay(frame, basket_assignment, occupancy):
    """바스켓 bbox + b1~b4 라벨 + 점유 여부를 그린 프레임을 반환 (원본은 건드리지 않음).

    ROS 노드(_publish_debug_image)와 ROS 없이 정지 이미지로 로직만 확인하는
    basket_camera_static_test.py가 동일한 시각화를 공유하기 위해 분리했다.
    """
    annotated = frame.copy()
    for name, box in basket_assignment.items():
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = (0, 0, 255) if occupancy[name] else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {'APPLE' if occupancy[name] else 'EMPTY'}"
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    return annotated


class BasketDetectionNode(Node):
    """바스켓 점유 상태를 주기적으로 판정해서 /basket_status로 publish하는 노드."""

    def __init__(self):
        super().__init__('basket_detection_node')

        self.declare_parameter('camera_topic_prefix', '/camera2/camera')
        topic_prefix = self.get_parameter('camera_topic_prefix').get_parameter_value().string_value

        self.img_node = BasketImgNode(topic_prefix)
        self.model = AppleStatusModel()

        self.status_pub = self.create_publisher(String, 'basket_status', 10)
        self.debug_image_pub = self.create_publisher(Image, 'basket_camera/debug_image', 10)

        self.create_timer(STATUS_PUBLISH_PERIOD_SEC, self._publish_status)

        self.get_logger().info(
            f"BasketDetectionNode initialized (camera_topic_prefix={topic_prefix})"
        )

    def _publish_status(self):
        frame = self.img_node.get_color_frame()
        if frame is None:
            return

        detections = self.model.get_all_detections(frame)
        basket_assignment, valid = assign_baskets(detections)
        occupancy = compute_occupancy(detections, basket_assignment)

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
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(node.img_node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.img_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
