import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MockRobotNode(Node):
    def __init__(self):
        super().__init__("mock_robot_node")
        # 백엔드가 publish하는 /robot/command 토픽을 구독합니다.
        self.subscription = self.create_subscription(
            String,
            "/robot/command",
            self.command_callback,
            10
        )
        self.get_logger().info("🤖 [가짜 로봇 노드]가 가동되었습니다. 백엔드의 명령을 기다리는 중...")

    def command_callback(self, msg: String):
        self.get_logger().info("🔔 [백엔드 신호 수신] 토픽 데이터가 들어왔습니다!")
        try:
            # 수신한 JSON 문자열 파싱
            data = json.loads(msg.data)
            print("\n" + "="*50)
            print(f"▶️ 수신된 명령: {data.get('command')}")
            print(f"🆔 발급된 Task ID: {data.get('task_id')}")
            print(f"⚙️ 선택된 Mode ID: {data.get('mode_id')}")
            print(f"⏰ 타임스탬프: {data.get('timestamp')}")
            print("="*50 + "\n")
            
            print("💡 로봇 제어 팀원의 노드는 이 데이터를 파싱해서 모션을 구동하게 됩니다.")
        except json.JSONDecodeError:
            self.get_logger().error(f"❌ JSON 파싱 실패! raw 데이터: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = MockRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()