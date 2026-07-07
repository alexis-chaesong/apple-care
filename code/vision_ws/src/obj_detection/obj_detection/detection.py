import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
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
DEBUG_IMAGE_PERIOD_SEC = 0.1


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.img_node = ImgNode()
        # 서비스 콜백 안에서 카메라 노드를 spin하기 위한 전용 executor
        self.img_executor = SingleThreadedExecutor()
        self.img_executor.add_node(self.img_node)

        self.model = AppleStatusModel()
        self.intrinsics = self._wait_for_valid_data(
            self.img_node.get_camera_intrinsic, "camera intrinsics"
        )

        self.create_service(
            SrvAppleStatus,
            'get_apple_status',
            self.handle_get_status,
        )

        self.bridge = CvBridge()
        self.debug_image_pub = self.create_publisher(Image, 'obj_detection/debug_image', 10)
        self.create_timer(DEBUG_IMAGE_PERIOD_SEC, self._publish_debug_image)

        self.get_logger().info("ObjectDetectionNode initialized.")

    def _publish_debug_image(self):
        """최신 컬러 프레임에 YOLO 인식 결과를 그려서 디버그 토픽으로 퍼블리시."""
        self.img_executor.spin_once(timeout_sec=0.0)
        frame = self.img_node.get_color_frame()
        if frame is None:
            return
        annotated = self.model.annotate_frame(frame)
        msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        self.debug_image_pub.publish(msg)

    def handle_get_status(self, request, response):
        """카메라에 보이는 사과 상태를 판정해서 반환한다."""
        self.img_executor.spin_once(timeout_sec=0.1)

        label, score, box = self.model.get_best_detection(self.img_node, self.img_executor)

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
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            self.img_executor.spin_once(timeout_sec=0.1)
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
    try:
        rclpy.spin(node)
    finally:
        node.img_executor.shutdown()
        node.img_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()