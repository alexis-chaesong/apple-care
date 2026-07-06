import os

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from std_msgs.msg import String

from apple_care_msgs.srv import SrvAppleStatus
from voice_processing.tts import TTS

PACKAGE_NAME = "voice_processing"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
ENV_PATH = os.path.join(PACKAGE_PATH, "resource", ".env")
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

POLL_INTERVAL_SEC = 2.0
UNKNOWN_QUESTION = "이게 뭐예요?"


class UnknownObjectWatcher(Node):
    """obj_detection의 get_apple_status 서비스를 주기적으로 호출해서,
    상태가 unknown으로 바뀌는 순간 TTS로 질문하고 HMI용 토픽을 publish한다."""

    def __init__(self):
        super().__init__('unknown_object_watcher')
        self.client = self.create_client(SrvAppleStatus, 'get_apple_status')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get_apple_status 서비스 대기 중...')

        self.tts = TTS(openai_api_key=OPENAI_API_KEY)
        self.question_pub = self.create_publisher(String, '/hmi/question', 10)

        self.last_status = None
        self.timer = self.create_timer(POLL_INTERVAL_SEC, self.poll_status)
        self.get_logger().info("UnknownObjectWatcher initialized.")

    def poll_status(self):
        request = SrvAppleStatus.Request()
        future = self.client.call_async(request)
        future.add_done_callback(self._on_response)

    def _on_response(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"서비스 호출 실패: {e}")
            return

        status = response.status
        # 계속 unknown이어도 매번 질문하지 않도록, 상태가 "바뀐 순간"에만 반응한다.
        if status == "unknown" and self.last_status != "unknown":
            self.get_logger().info("미확인 물체 감지 -> 질문 트리거")
            self.question_pub.publish(String(data=UNKNOWN_QUESTION))
            self.tts.speak(UNKNOWN_QUESTION)

        self.last_status = status


def main(args=None):
    rclpy.init(args=args)
    node = UnknownObjectWatcher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
