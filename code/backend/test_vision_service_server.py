"""
test_vision_service_server.py
===============================
진짜 obj_detection 노드 없이 vision_bridge.py를 검증하기 위한 가짜
get_apple_status 서비스 서버.

아래 5단계 시퀀스를 순서대로(끝나면 처음부터 반복) 응답해서, vision_bridge.py의
다수결 확정(CONFIRM_SAMPLE_COUNT=10) + 디바운스 로직과 전체 파이프라인을 단계별로
검증할 수 있게 함. vision_bridge.py가 같은 물체로 보이는 폴링을 10개 모아 다수결로
확정한 뒤에만 큐에 넣도록 바뀌었으므로, 각 스테이지 반복 횟수를 10개를 확실히
채우고도 여유가 남게 잡음(그래야 다수결 확정 + 그 다음 디바운스까지 한 스테이지
안에서 같이 확인됨):

  1) apple_normal, 같은 position 12번 연속 -> 10개째에 다수결 확정돼 vision_queue에
     1번만 들어가야 정상 (11~12번째는 확정 직후 새 창을 채우기 시작할 뿐 재확정 안 됨)
  2) apple_normal, 다른 position 12번 연속 -> "새 사과"로 인식해서 다시 1번 들어가야 정상
  3) unknown, 같은 position 여러 번 연속 -> 10개째에 다수결로 unknown 확정돼 최초
     1번만 HITL 세션 시작, 이후는 디바운스
  4) empty 여러 번 -> 디바운스 상태 + 샘플 창 리셋 (물체 없음)
  5) apple_damaged, 새 position -> empty 이후라 "새 사과"로 다시 큐에 들어가야 정상

사용법:
    source vision_ws/install/setup.bash
    python3 backend/test_vision_service_server.py
"""

import rclpy
from rclpy.node import Node

from apple_care_msgs.srv import SrvAppleStatus

SERVICE_NAME = "get_apple_status"

# (status, confidence, position, 이 상태를 몇 번 연속 응답할지)
STAGES = [
    ("apple_normal", 0.92, [100.0, 100.0, 300.0], 12),  # 10개째 다수결 확정 -> dedup 확인
    ("apple_normal", 0.90, [300.0, 300.0, 300.0], 12),  # 다른 position -> 새 사과 확인
    ("unknown", 0.35, [500.0, 500.0, 300.0], 20),       # 충분히 오래 유지 -> HITL 세션 확인
    ("empty", 0.0, [], 5),                              # 물체 없음 -> 디바운스 리셋 확인
    ("apple_damaged", 0.85, [700.0, 700.0, 300.0], 12), # empty 이후 새 사과 확인
]

_TOTAL_TICKS = sum(count for _, _, _, count in STAGES)


def _stage_for_tick(tick: int):
    """0부터 시작하는 누적 호출 횟수(tick)에 해당하는 STAGES 항목을 반환."""
    idx = tick % _TOTAL_TICKS
    for status, confidence, position, count in STAGES:
        if idx < count:
            return status, confidence, position
        idx -= count
    raise AssertionError("unreachable")  # _TOTAL_TICKS 계산과 어긋나면 즉시 드러나도록


class FakeVisionServiceServer(Node):
    def __init__(self):
        super().__init__("fake_vision_service_server")
        self._tick = 0
        self.create_service(SrvAppleStatus, SERVICE_NAME, self._handle_request)
        self.get_logger().info(
            f"FakeVisionServiceServer 시작. '{SERVICE_NAME}' 서비스 제공 중 "
            f"(총 {_TOTAL_TICKS}스텝 시퀀스 반복)."
        )

    def _handle_request(self, request, response):
        status, confidence, position = _stage_for_tick(self._tick)
        response.status = status
        response.confidence = confidence
        response.position = position

        self.get_logger().info(
            f"[tick={self._tick}] 응답: status={status} confidence={confidence} position={position}"
        )
        self._tick += 1
        return response


def main(args=None):
    rclpy.init(args=args)
    node = FakeVisionServiceServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
