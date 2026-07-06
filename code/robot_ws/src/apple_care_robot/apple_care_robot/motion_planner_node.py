"""
Motion Planner Node
====================

역할 (기획안 9번 항목):
    Decision Planner가 "이 사과는 정상박스, 저 사과는 폐기박스" 라고 판정을 내리면,
    그 판정들을 받아서 "로봇이 어떤 순서로 움직여야 이동거리가 최소가 될지" 계산하는 부서.

비유:
    택배 분류 알바생이 컨베이어벨트에서 상자를 하나씩 집어 옮긴다고 생각하면,
    상자가 들어온 순서대로(A→B→C→D) 그냥 옮기는 게 아니라
    "어차피 왼쪽 트럭에 실을 상자들끼리 먼저 몰아서 옮기고, 오른쪽은 나중에" 처럼
    동선을 짜는 역할.

구독(Subscribe): /decision/result   (하나의 사과에 대한 판정 결과, JSON)
발행(Publish):   /motion/queue      (정렬된 작업 순서 전체, JSON 리스트)

TODO 표시된 부분이 형이 직접 채워야 할 핵심 로직입니다.
지금은 "들어오는 대로 큐에 쌓았다가, 일정 개수 모이면 정렬해서 내보내는" 최소 골격만 있어요.
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MotionPlannerNode(Node):

    def __init__(self):
        super().__init__('motion_planner_node')

        # 아직 로봇에게 보내지 않은, 대기 중인 판정 결과들을 쌓아두는 리스트
        # 예: [{"fruit": "apple", "destination": "normal_box", "pose": [x,y,z], ...}, ...]
        self.pending_items = []

        # 몇 개가 모이면 한번에 순서를 계산해서 내보낼지 (실습용 임시값)
        # TODO: 실제로는 "테이블 위에 사과가 더 없다"는 신호를 받아서 처리하는 게 맞을 수도 있음.
        #       우선은 개수 기반으로 단순하게 시작해보고, 나중에 이벤트 기반으로 바꿔도 됨.
        self.batch_size = 4

        self.subscription = self.create_subscription(
            String,
            '/decision/result',
            self.decision_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            '/motion/queue',
            10
        )

        self.get_logger().info('Motion Planner Node started. waiting for /decision/result ...')

    def decision_callback(self, msg: String):
        """Decision Planner로부터 판정 결과 하나를 받았을 때 호출됨."""
        try:
            item = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f'Invalid JSON received: {msg.data}')
            return

        self.get_logger().info(f'Received decision: {item}')
        self.pending_items.append(item)

        if len(self.pending_items) >= self.batch_size:
            ordered = self.plan_motion_order(self.pending_items)
            self.publish_queue(ordered)
            self.pending_items = []

    def plan_motion_order(self, items: list) -> list:
        """
        핵심 로직 (TODO): 목적지 + 이동거리를 고려해서 순서를 최적화하는 부분.

        기획안 9번 예시:
            입력 순서: A(정상) B(폐기) C(정상) D(정상)
            출력 순서: A(정상) C(정상) D(정상) B(폐기)
            -> 같은 목적지끼리 몰아서 처리 = 왕복 이동 감소

        지금은 "그냥 들어온 순서 그대로" 반환하는 더미 로직만 있습니다.
        형이 채워야 할 것:
          1) destination 별로 그룹화 (예: dict[destination] = [item, item, ...])
          2) 그룹 안에서는 로봇의 현재 위치(pick pose)를 기준으로 가까운 것부터 정렬
             -> 이때 item['pose'] = [x, y, z] 를 이용해서 거리 계산 (유클리드 거리면 충분)
          3) 그룹들을 어떤 순서로 이어붙일지 결정 (예: 가장 가까운 그룹부터)

        힌트: 처음엔 그냥 destination 별로 그룹화만 해서 순서를 만들어도
              기획안의 예시 결과(A,C,D,B)는 재현할 수 있어요.
              "거리 기반 정렬"은 그 다음 단계로 추가해보세요.
        """
        # ------------------- TODO: 여기부터 구현 -------------------
        ordered_items = items  # 지금은 그대로 반환 (더미)
        # ------------------------------------------------------------

        return ordered_items

    def publish_queue(self, ordered_items: list):
        msg = String()
        msg.data = json.dumps(ordered_items)
        self.publisher.publish(msg)
        self.get_logger().info(f'Published motion queue with {len(ordered_items)} items.')


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
