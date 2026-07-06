import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from apple_care_msgs.srv import SrvAppleStatus
from obj_detection.realsense import ImgNode
from obj_detection.yolo import AppleStatusModel, MIN_KNOWN_CONFIDENCE

# 트레이/배경까지의 대략적인 거리(mm). 실제 설치 환경에 맞춰 캘리브레이션 필요.
BACKGROUND_DISTANCE_MM = 700
PRESENCE_MARGIN_MM = 50
MIN_PRESENCE_PIXELS = 500

# 디버그용 인식 결과 시각화 이미지 퍼블리시 주기(초)
DEBUG_IMAGE_PERIOD_SEC = 0.05


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.img_node = ImgNode()
        # img_node는 별도 스레드에서 계속 spin -> 서비스 콜백이 오래 걸려도(프레임 수집 1초)
        # 디버그 이미지 타이머나 다른 콜백이 카메라 데이터 갱신에 막히지 않게 함
        self.img_executor = SingleThreadedExecutor()
        self.img_executor.add_node(self.img_node)
        self._img_spin_thread = threading.Thread(target=self.img_executor.spin, daemon=True)
        self._img_spin_thread.start()

        self.model = AppleStatusModel()
        self.intrinsics = self._wait_for_valid_data(
            self.img_node.get_camera_intrinsic, "camera intrinsics"
        )

        # 서비스(느림, ~1초)와 디버그 이미지 타이머(0.1초 주기)가 서로를 막지 않도록
        # 콜백 그룹을 분리하고 MultiThreadedExecutor로 동시에 돌린다 (main() 참고)
        self.service_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        self.create_service(
            SrvAppleStatus,
            'get_apple_status',
            self.handle_get_status,
            callback_group=self.service_cb_group,
        )

        self.bridge = CvBridge()
        self.debug_image_pub = self.create_publisher(Image, 'obj_detection/debug_image', 10)
        self.create_timer(
            DEBUG_IMAGE_PERIOD_SEC, self._publish_debug_image, callback_group=self.timer_cb_group
        )

        self.get_logger().info("ObjectDetectionNode initialized.")

    def _publish_debug_image(self):
        """최신 컬러 프레임에 YOLO 인식 결과를 그려서 디버그 토픽으로 퍼블리시."""
        frame = self.img_node.get_color_frame()
        if frame is None:
            return
        annotated = self.model.annotate_frame(frame)
        msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        self.debug_image_pub.publish(msg)

    def handle_get_status(self, request, response):
        """카메라에 보이는 사과 상태를 판정해서 반환한다."""
        label, score, box = self.model.get_best_detection(self.img_node)

        if box is None:
            # YOLO가 아무 박스도 못 찾음 -> depth로 "물체 자체는 있는지" 확인
            depth_frame = self.img_node.get_depth_frame()
            if self._object_present_by_depth(depth_frame):
                self.get_logger().info("No YOLO box, but depth shows an object -> unknown")
                response.status = "unknown"
                response.confidence = 0.0
                response.position = []
            else:
                response.status = "empty"
                response.confidence = 0.0
                response.position = []
            return response

        cx, cy = map(int, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        position = self._compute_position(cx, cy)

        if score < MIN_KNOWN_CONFIDENCE:
            self.get_logger().info(f"Low confidence detection ({score:.2f}) -> unknown")
            response.status = "unknown"
        else:
            response.status = label

        response.confidence = float(score)
        response.position = [float(x) for x in position]
        return response

    def _object_present_by_depth(self, depth_frame):
        """배경보다 확실히 가까운 픽셀이 충분히 있으면 물체가 있다고 판단."""
        if depth_frame is None:
            return False
        closer_than_background = (depth_frame > 0) & (
            depth_frame < (BACKGROUND_DISTANCE_MM - PRESENCE_MARGIN_MM)
        )
        return int(closer_than_background.sum()) >= MIN_PRESENCE_PIXELS

    def _compute_position(self, cx, cy):
        """픽셀 좌표 -> 카메라 좌표계 3D 위치. depth가 없으면 (0,0,0)."""
        depth_frame = self._wait_for_valid_data(self.img_node.get_depth_frame, "depth frame")
        try:
            cz = depth_frame[cy, cx]
        except IndexError:
            self.get_logger().warn(f"Coordinates ({cx},{cy}) out of range.")
            return 0.0, 0.0, 0.0
        if cz == 0:
            return 0.0, 0.0, 0.0
        return self._pixel_to_camera_coords(cx, cy, cz)

    def _wait_for_valid_data(self, getter, description):
        # img_executor가 별도 스레드에서 계속 spin 중이므로 여기서는 그냥 대기만 한다.
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            time.sleep(0.1)
            self.get_logger().info(f"Retry getting {description}.")
            data = getter()
        return data

    def _pixel_to_camera_coords(self, x, y, z):
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']
        return (
            (x - ppx) * z / fx,
            (y - ppy) * z / fy,
            z,
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    # 서비스(get_apple_status, ~1초 소요)와 디버그 이미지 타이머(0.1초 주기)를
    # 서로 다른 콜백 그룹으로 등록해뒀으므로, MultiThreadedExecutor로 돌려야
    # 실제로 동시에(다른 스레드에서) 실행되어 서로 막지 않는다.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.img_executor.shutdown()
        node.img_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
