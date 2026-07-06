import os

import requests
import rclpy
from rclpy.node import Node

from apple_care_msgs.srv import SrvAppleStatus

# backend/config.py의 app_host/app_port 기본값(0.0.0.0:8000)에 맞춘 기본 URL.
# 백엔드가 다른 호스트/포트에서 뜨면 환경변수로 덮어쓰면 됨.
BACKEND_URL = os.getenv("BACKEND_VISION_URL", "http://localhost:8000/api/vision/result")

POLL_INTERVAL_SEC = 1.0

# obj_detection의 status -> backend의 (fruit_type, defect_type, unknown_flag) 매핑
STATUS_MAP = {
    "apple_normal": ("apple", None, False),
    "apple_rotten": ("apple", "rotten", False),
    "apple_damaged": ("apple", "damaged", False),
    "unknown": ("unknown", None, True),
}


class VisionBridge(Node):
    """get_apple_status 서비스를 주기적으로 호출해서, 새로 감지된 물체가 있으면
    backend(POST /api/vision/result)로 판정 결과를 전달한다."""

    def __init__(self):
        super().__init__('vision_bridge')
        self.client = self.create_client(SrvAppleStatus, 'get_apple_status')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_apple_status 서비스 대기 중...')

        self.last_status = None
        self.timer = self.create_timer(POLL_INTERVAL_SEC, self.poll_status)
        self.get_logger().info(f"VisionBridge initialized. backend -> {BACKEND_URL}")

    def poll_status(self):
        future = self.client.call_async(SrvAppleStatus.Request())
        future.add_done_callback(self._on_response)

    def _on_response(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"서비스 호출 실패: {e}")
            return

        status = response.status

        if status == "empty":
            # 트레이가 비었으니 다음에 새로 올라오는 물체를 다시 보고할 수 있게 리셋
            self.last_status = None
            return

        # 같은 물체가 프레임에 계속 잡히는 동안 매번 다시 보고하지 않도록,
        # 상태가 "바뀐 순간"에만 backend로 보낸다.
        if status == self.last_status:
            return
        self.last_status = status

        if status not in STATUS_MAP:
            self.get_logger().warn(f"알 수 없는 status 값: {status}")
            return

        fruit_type, defect_type, unknown_flag = STATUS_MAP[status]
        payload = {
            "fruit_type": fruit_type,
            "defect_type": defect_type,
            "confidence": response.confidence,
            "unknown_flag": unknown_flag,
            "center": list(response.position) if response.position else None,
        }
        self._post_to_backend(payload)

    def _post_to_backend(self, payload):
        try:
            res = requests.post(BACKEND_URL, json=payload, timeout=2.0)
            res.raise_for_status()
            self.get_logger().info(f"backend 전송 완료: {payload}")
        except requests.RequestException as e:
            self.get_logger().error(f"backend 전송 실패: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VisionBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
