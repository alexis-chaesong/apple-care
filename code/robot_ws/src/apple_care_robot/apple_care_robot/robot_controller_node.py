"""
Robot Controller Node
=======================

역할 (기획안 9번 항목):
    Motion Planner가 정해준 순서(motion queue)를 하나씩 받아서
    실제로 두산 M0609 로봇을 Pick & Place 시키는 부서.

비유:
    Motion Planner가 "오늘 할 일 목록"을 순서대로 적어주는 매니저라면,
    Robot Controller는 그 목록을 들고 실제로 몸을 움직이는 작업자.
    "사과 집기 -> 상자까지 걷기 -> 놓기 -> 제자리로" 를 하나씩 순서대로 해내는 역할.

상태머신(State Machine) 개념:
    로봇이 지금 "뭘 하고 있는 중"인지를 나타내는 상태를 하나씩 정의해두고,
    각 상태에서 할 일이 끝나면 다음 상태로 넘어가는 구조입니다.
    신호등이 빨강->초록->노랑->빨강 순서로만 바뀌는 것과 같은 원리예요.

구독(Subscribe): /motion/queue   (정렬된 작업 목록, JSON 리스트)

TODO 표시된 부분에 실제 두산 로봇 제어 API(DSR 라이브러리 등)를 연결하면 됩니다.
지금은 print/log로 "이 동작을 할 차례"라는 것만 흉내내는 뼈대입니다.
"""

import json
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RobotState(Enum):
    IDLE = auto()
    MOVE_TO_PICK = auto()
    GRASP = auto()
    MOVE_TO_PLACE = auto()
    RELEASE = auto()
    RETURN_HOME = auto()
    SAFETY_HOLD = auto()   # 예외 상황(그립 실패 등) 발생 시 대기 상태


class RobotControllerNode(Node):

    def __init__(self):
        super().__init__('robot_controller_node')

        self.state = RobotState.IDLE
        self.task_queue = []      # motion planner가 준 작업 목록
        self.current_task = None  # 지금 처리 중인 항목 하나
        self.retry_count = 0
        self.max_retries = 2      # 기획안 예외4: Pick 실패 시 재시도 횟수

        self.subscription = self.create_subscription(
            String,
            '/motion/queue',
            self.queue_callback,
            10
        )

        # 일정 주기로 상태머신을 진행시키는 타이머
        # (실제 로봇 동작은 시간이 걸리므로, 매번 callback 안에서 다 처리하지 않고
        #  주기적으로 "지금 상태에서 할 일 하나"씩만 진행하는 방식)
        self.timer = self.create_timer(0.5, self.tick)

        self.get_logger().info('Robot Controller Node started. waiting for /motion/queue ...')

    def queue_callback(self, msg: String):
        try:
            items = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f'Invalid JSON received: {msg.data}')
            return

        self.task_queue.extend(items)
        self.get_logger().info(f'Queue received. total pending tasks: {len(self.task_queue)}')

    def tick(self):
        """0.5초마다 호출됨. 현재 state에 따라 다음 행동을 결정."""

        if self.state == RobotState.IDLE:
            if self.task_queue:
                self.current_task = self.task_queue.pop(0)
                self.retry_count = 0
                self.get_logger().info(f'New task started: {self.current_task}')
                self.state = RobotState.MOVE_TO_PICK

        elif self.state == RobotState.MOVE_TO_PICK:
            # TODO: 여기서 실제 두산 API로 pick pose까지 이동
            #   예: movel(self.current_task['pose'], ...)  <- DSR API 예시, 실제 함수명은 확인 필요
            self.get_logger().info(f"[TODO] Move to pick pose: {self.current_task.get('pose')}")
            self.state = RobotState.GRASP

        elif self.state == RobotState.GRASP:
            success = self.grasp_object()  # TODO 안에 실제 그리퍼 close + 성공여부 확인 로직
            if success:
                self.state = RobotState.MOVE_TO_PLACE
            else:
                self.handle_grasp_failure()

        elif self.state == RobotState.MOVE_TO_PLACE:
            destination = self.current_task.get('destination')
            # TODO: destination(normal_box / processing_box / discard_box 등)에 맞는
            #       실제 좌표로 이동하는 코드 작성
            self.get_logger().info(f"[TODO] Move to destination box: {destination}")
            self.state = RobotState.RELEASE

        elif self.state == RobotState.RELEASE:
            # TODO: 그리퍼 open
            self.get_logger().info('[TODO] Gripper open (drop object)')
            self.state = RobotState.RETURN_HOME

        elif self.state == RobotState.RETURN_HOME:
            # TODO: home / 대기 위치로 복귀
            self.get_logger().info('[TODO] Return to home position')
            self.current_task = None
            self.state = RobotState.IDLE

        elif self.state == RobotState.SAFETY_HOLD:
            # 기획안 예외4: 재시도 다 실패하면 여기서 멈춰서 사용자 알림 대기
            # TODO: TTS로 "그립에 계속 실패했습니다" 알림 보내는 로직 연결
            #       (지금은 사람이 직접 확인 후 재개 명령을 줘야 하는 상태)
            pass

    def grasp_object(self) -> bool:
        """
        TODO: 실제 그리퍼 close 명령 + 성공 여부 판단.
        (RG2 그리퍼는 보통 힘/거리 피드백으로 물체를 잡았는지 확인 가능)

        지금은 항상 성공했다고 가정하는 더미 구현.
        형이 실습해볼 것:
          - 일부러 False를 리턴하게 만들어서 아래 handle_grasp_failure()의
            재시도 로직이 잘 동작하는지 먼저 확인해보세요.
        """
        return True

    def handle_grasp_failure(self):
        """기획안 예외4: Pick 실패 -> Retry -> 재실패 시 Safety Position."""
        self.retry_count += 1
        self.get_logger().warn(f'Grasp failed. retry {self.retry_count}/{self.max_retries}')

        if self.retry_count <= self.max_retries:
            self.state = RobotState.MOVE_TO_PICK  # 다시 pick pose로 이동해서 재시도
        else:
            self.get_logger().error('Max retries exceeded. Entering SAFETY_HOLD.')
            self.state = RobotState.SAFETY_HOLD


def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
