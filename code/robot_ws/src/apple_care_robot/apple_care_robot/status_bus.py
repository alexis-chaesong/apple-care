"""
StatusBus
==========

백엔드(apple-care/code/backend/robot_bridge.py)가 구독하는 로봇 상태 토픽들을
발행하는 헬퍼. robot_bridge.py 주석에서 "현민님의 StatusBus가 발행하는 3개
토픽"으로 언급되는 바로 그 컴포넌트를 로봇 쪽에 구현한 것.

발행 토픽 (robot_bridge.py가 구독하는 이름/타입/포맷에 맞춤):
    /robot/process_state  (std_msgs/String) - "STATE" 또는 "STATE:message"
    /robot/motion_status   (std_msgs/String) - "STATE" 또는 "STATE:message"
    /robot/safety_event    (std_msgs/String) - "CODE:message" (항상 콜론 포함)
    /robot/recovery_stage  (std_msgs/String)
    /gripper/status        (std_msgs/Bool)   - True면 그리퍼가 뭔가를 잡고 있음
"""

from std_msgs.msg import String, Bool

TOPIC_PROCESS_STATE = "/robot/process_state"
TOPIC_MOTION_STATUS = "/robot/motion_status"
TOPIC_SAFETY_EVENT = "/robot/safety_event"
TOPIC_RECOVERY_STAGE = "/robot/recovery_stage"
TOPIC_GRIPPER_STATUS = "/gripper/status"


class StatusBus:
    """rclpy 노드 하나를 받아 백엔드용 상태 토픽 발행자(Publisher)들을 묶어서 제공."""

    def __init__(self, node):
        self.node = node
        self._process_state_pub = node.create_publisher(String, TOPIC_PROCESS_STATE, 10)
        self._motion_status_pub = node.create_publisher(String, TOPIC_MOTION_STATUS, 10)
        self._safety_event_pub = node.create_publisher(String, TOPIC_SAFETY_EVENT, 10)
        self._recovery_stage_pub = node.create_publisher(String, TOPIC_RECOVERY_STAGE, 10)
        self._gripper_status_pub = node.create_publisher(Bool, TOPIC_GRIPPER_STATUS, 10)

    def set_state(self, state: str, msg: str = None) -> None:
        """/robot/process_state 발행. state 예: IDLE/INIT/READY/MOVING/COMPLETE/ERROR 등."""
        text = state if not msg else f"{state}:{msg}"
        self._process_state_pub.publish(String(data=text))

    def set_motion(self, status: str, msg: str = None) -> None:
        """/robot/motion_status 발행. 세부 동작 단위(PICKING/PLACING 등) 표시용."""
        text = status if not msg else f"{status}:{msg}"
        self._motion_status_pub.publish(String(data=text))

    def publish_safety(self, error_code: str, error_msg: str = "") -> None:
        """/robot/safety_event 발행. 항상 'CODE:message' 형태로 보냄."""
        self._safety_event_pub.publish(String(data=f"{error_code}:{error_msg}"))

    def publish_recovery_stage(self, stage: str) -> None:
        """/robot/recovery_stage 발행."""
        self._recovery_stage_pub.publish(String(data=stage))

    def publish_gripper_status(self, grasped: bool) -> None:
        """/gripper/status 발행. True면 뭔가를 잡고 있는 상태."""
        self._gripper_status_pub.publish(Bool(data=grasped))
