"""
vision_bridge.py
=================
Vision(obj_detection 패키지)의 get_apple_status ROS2 서비스를 주기적으로 폴링해서
결과를 models.VisionFeatureIn으로 변환한 뒤 asyncio 큐(vision_queue)에 적재하는
유일한 창구.

robot_bridge.py와 동일한 원칙:
  - ROS2 콜백/타이머 안에서는 절대 무거운 작업(DB 조회, LLM 호출 등)을 하지 않음.
    여기서는 "서비스 호출 -> 응답을 VisionFeatureIn으로 변환 -> 큐 적재 예약"만 함.
  - ROS2 스레드(SingleThreadedExecutor)와 FastAPI의 asyncio 이벤트 루프 사이를
    call_soon_threadsafe()로 안전하게 넘나듦.
  - 전역 싱글톤 패턴(vision_bridge_manager)으로 관리, main.py의 lifespan에서 start/shutdown.

⚠️ Vision은 토픽을 publish하지 않고 요청-응답 서비스(get_apple_status)만 제공함
(1단계 조사 결과). 그래서 robot_bridge.py처럼 "구독 콜백"이 아니라, ROS2 타이머로
"주기적으로 서비스를 호출하는 폴링 클라이언트"로 구현되어 있음.

필드 매핑 (사용자 확정):
    status=apple_normal  -> fruit_type=apple, defect_type=None
    status=apple_rotten  -> fruit_type=apple, defect_type=mold
    status=apple_damaged -> fruit_type=apple, defect_type=bruise
    status=apple_small   -> fruit_type=apple, defect_type=None, size=small
        (예전에는 depth 실측 지름으로 apple_normal/apple_damaged를 재분류해
        apple_small_normal/apple_small_damaged 두 값으로 냈지만, 이제 vision이
        apple_small을 직접 학습한 모델을 쓰므로 YOLO가 낸 라벨을 그대로 받음)
    status=unknown        -> fruit_type=unknown, unknown_flag=True
    status=empty         -> 무시(큐에 안 넣음), 디바운스 상태 리셋
    bbox                 -> 항상 None (Vision이 안 줌)
    confidence           -> 서비스 응답 그대로 (이미 0~1 스케일)
    position             -> center

디바운스: 폴링 방식 특성상 같은 사과가 안 치워졌으면 매 폴링마다 같은 결과가 반복됨.
직전에 큐에 넣은 (status, position)과 이번 응답이 실질적으로 같으면(각 축
vision_position_dedup_threshold_mm 이내) 큐에 넣지 않음. status가 바뀌었거나
position이 유의미하게 다르면 "새 사과"로 취급해 큐에 넣음.
"""

import logging
import threading
import time
from asyncio import AbstractEventLoop, Queue, QueueFull
from collections import Counter
from typing import Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from config import settings
from models import VisionFeatureIn

logger = logging.getLogger(__name__)

try:
    from apple_care_msgs.srv import SrvAppleStatus
except ModuleNotFoundError:
    # vision_ws가 이 프로세스 환경에 build/source 되어 있지 않은 경우.
    # Vision ROS2 폴링(get_apple_status)만 비활성화하고, 나머지 백엔드
    # (로봇 제어, HITL, HTTP vla_router.py 경로 등)는 정상 동작해야 하므로
    # 여기서 서버 전체를 죽이지 않고 경고만 남김 - 실제 비활성화는
    # start_bridge()에서 SrvAppleStatus is None을 보고 처리함
    SrvAppleStatus = None
    logger.warning(
        "apple_care_msgs를 찾을 수 없어 Vision ROS2 폴링을 비활성화합니다. "
        "vision_ws에서 'colcon build --packages-select apple_care_msgs' 후 "
        "'source install/setup.bash'를 실행한 뒤 서버를 재시작하세요. "
        "그 전까지는 HTTP 경로(POST /api/vision/result)로만 Vision 데이터를 받습니다."
    )

ROS_NODE_NAME = "fastapi_vision_bridge"
SERVICE_NAME = "get_apple_status"
SPIN_TIMEOUT_SEC = 0.1

# "empty" 응답이 몇 번 연속으로 와야 진짜로 물체가 사라졌다고 보고 디바운스를 리셋할지.
# 1이면 예전 동작(즉시 리셋)과 같음.
EMPTY_RESET_STREAK = 3

# 작업자가 HITL 질문에 "무시해"로 답한 사과를 vision이 다시 최우선으로 고르지
# 않도록 얼마나 오래 제외해둘지(초). 이 시간이 지나면 자동으로 제외를 풀어서
# 다른 사과가 없으면 결국 이 사과도 다시 처리 대상이 됨 (영원히 방치되지 않게).
SKIP_EXCLUDE_COOLDOWN_SEC = 60.0

# 실측으로 확인된 문제: YOLO 판정이 프레임마다 흔들려서, 흠집(apple_damaged)이
# 종종 곰팡이(apple_rotten)로 잘못 튀는 등 확실한 분류값도 단 한 프레임만으로
# 바로 실행/질문까지 이어지는 경우가 있었음. 그래서 confident한 분류든 unknown이든
# 첫 폴링에 바로 큐에 넣지 않고, 같은 물체로 보이는 폴링을 CONFIRM_SAMPLE_COUNT번
# 모아서 그 중 다수결(최빈값)로 확정한 뒤에만 큐에 넣는다 - "확실한 것부터 처리하고
# unknown은 가장 후순위로"라는 기존 원칙은 다수결 자체가 자연히 만족함(정상 분류가
# 다수면 정상으로, unknown이 다수면 그때만 진짜 unknown으로 확정됨).
CONFIRM_SAMPLE_COUNT = 5

# obj_detection/yolo.py의 CLASS_NAMES + detection.py가 붙이는 unknown/empty를 포함한
# 전체 status 값 중, 실제 fruit_type/defect_type으로 명확히 매핑되는 값들만 여기 정의.
# unknown/empty는 별도 분기로 처리함 (아래 _status_to_vision_feature 참고)
STATUS_TO_FRUIT_DEFECT = {
    "apple_normal": ("apple", None),
    "apple_rotten": ("apple", "mold"),
    "apple_damaged": ("apple", "bruise"),
    "apple_small": ("apple", None),
}

# size 필드로 넘겨야 하는 status 값 (defect_type과는 별개로 취급됨).
STATUS_TO_SIZE = {
    "apple_small": "small",
}


def _positions_close(a: list[float], b: list[float], threshold_mm: float) -> bool:
    """두 position(x,y,z)의 각 축 차이가 전부 threshold_mm 이내면 '같은 위치'로 봄."""
    if not a or not b or len(a) != len(b):
        return False
    return all(abs(x - y) <= threshold_mm for x, y in zip(a, b))


def _is_zero_position(position: list[float]) -> bool:
    """
    obj_detection/detection.py의 _compute_position은 depth 측정 실패("cz == 0")
    시에도 빈 리스트가 아니라 [0.0, 0.0, 0.0]을 반환한다 - "물체 없음"이 아니라
    "박스는 찾았는데 그 지점의 depth를 못 읽었음"이라는 뜻. 이 값을 진짜 카메라
    좌표 (0,0,0)으로 착각해서 그대로 흘려보내면, 로봇 쪽에서 camera_to_base()로
    변환했을 때 "카메라 렌즈 자체의 위치"로 계산되어 엉뚱한 곳으로 이동하게 됨
    (box_sequence_test.py에서 실제로 재현된 버그). 그래서 position을 쓸 때는
    항상 "비어있는지"뿐 아니라 "전부 0인지"도 같이 걸러야 한다.
    """
    return bool(position) and all(v == 0.0 for v in position)


def _status_to_vision_feature(status: str, confidence: float, position: list[float]) -> VisionFeatureIn:
    """
    다수결로 확정된 (status, confidence, position) -> VisionFeatureIn 변환.
    confidence/position 검증은 이미 _on_response가 샘플을 모으는 시점에 각각
    끝냈으므로(범위 밖 confidence는 샘플에 안 들어감, 0 position은 빈 리스트로
    정규화됨) 여기서는 status -> fruit_type/defect_type/size 매핑만 담당한다.
    """
    if status == "unknown":
        fruit_type = "unknown"
        defect_type = None
        unknown_flag = True
    else:
        fruit_type, defect_type = STATUS_TO_FRUIT_DEFECT.get(status, (status, None))
        unknown_flag = False
        if status not in STATUS_TO_FRUIT_DEFECT:
            logger.warning("알 수 없는 Vision status 값: %s (fruit_type으로 그대로 사용)", status)
    size = STATUS_TO_SIZE.get(status)

    return VisionFeatureIn(
        fruit_type=fruit_type,
        size=size,
        defect_type=defect_type,
        confidence=confidence,
        unknown_flag=unknown_flag,
        bbox=None,
        center=position or None,
        frame_id=None,
    )


class VisionBridgeManager:
    """ROS2 서비스 클라이언트(get_apple_status) + 폴링 타이머 + Executor/Thread 생명주기 관리."""

    def __init__(self) -> None:
        self.node: Optional[Node] = None
        self.client = None
        self.executor: Optional[SingleThreadedExecutor] = None
        self.ros_thread: Optional[threading.Thread] = None
        self.loop: Optional[AbstractEventLoop] = None
        self.queue: Optional[Queue] = None
        self._stop_event = threading.Event()
        self._poll_timer = None
        self._pending_call = False  # 이전 폴링 응답이 아직 안 왔으면 중복 호출 방지

        # 디바운스 상태: 직전에 큐에 넣었던 (status, position)
        self._last_status: Optional[str] = None
        self._last_position: Optional[list[float]] = None
        # "empty" 응답이 이 횟수만큼 연속으로 와야 디바운스 상태를 초기화한다.
        # depth 판정이 경계값 근처라 "empty"/"unknown"을 한두 번씩 flicker할 수 있는데,
        # 매번 바로 리셋하면 사실 같은 물체를 계속 "새로 감지됨"으로 취급해 ask_human이
        # 반복 발생(HITL 팝업 스팸)하므로 약간의 유예(hysteresis)를 둔다.
        self._empty_streak = 0

        # 다수결 확정용 샘플 창(window). (status, confidence, position) 튜플을
        # CONFIRM_SAMPLE_COUNT개 모을 때까지는 큐에 넣지 않고 계속 쌓기만 함.
        # _sample_position_ref는 평균 계산의 시작값 참고용일 뿐, 창을 리셋하는
        # 기준으로는 쓰지 않음 (position 흔들림으로 리셋하면 안 되는 이유는
        # _on_response의 해당 주석 참고) - "다른 물체로 바뀜"은 오직 empty
        # 스트릭(응답 자체가 empty로 몇 번 연속 옴)으로만 판단한다.
        self._samples: list[tuple[str, float, list[float]]] = []
        self._sample_position_ref: Optional[list[float]] = None

        # "무시" 응답으로 제외된 사과의 카메라 좌표 + 언제부터 제외했는지.
        # _poll_once가 매 폴링마다 SrvAppleStatus.Request.avoid_position에 실어
        # 보내서, obj_detection이 이 좌표 근처 후보를 이번 선택에서 빼도록 함
        # (yolo.py의 get_best_detection 참고). SKIP_EXCLUDE_COOLDOWN_SEC이 지나면
        # 자동으로 풀림.
        self._skip_position: Optional[list[float]] = None
        self._skip_started_at: float = 0.0

    # ------------------------------------------------------------------
    def start_bridge(self, fastapi_loop: AbstractEventLoop, output_queue: Queue) -> None:
        """main.py의 lifespan에서 호출되어 백그라운드 스레드로 ROS2 폴링을 구동함."""
        if SrvAppleStatus is None:
            # apple_care_msgs가 없는 환경 - 이미 모듈 로드 시점에 경고를 남겼으므로
            # 여기서는 조용히 비활성화 상태로 남고 서버 나머지는 정상 기동되게 함
            logger.warning("Vision Bridge 비활성화 상태로 시작 건너뜀 (apple_care_msgs 없음).")
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = Node(ROS_NODE_NAME)
        self.loop = fastapi_loop
        self.queue = output_queue

        self.client = self.node.create_client(SrvAppleStatus, SERVICE_NAME)

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self._poll_timer = self.node.create_timer(
            settings.vision_poll_interval_sec, self._poll_once
        )

        self._stop_event.clear()
        self.ros_thread = threading.Thread(
            target=self._spin_loop, name="ros2-vision-spin-thread", daemon=True
        )
        self.ros_thread.start()
        logger.info(
            "Vision Bridge 시작: service=%s poll_interval=%.2fs",
            SERVICE_NAME, settings.vision_poll_interval_sec,
        )

    def shutdown(self) -> None:
        logger.info("VisionBridgeManager shutdown 시작...")
        self._stop_event.set()

        if self.ros_thread is not None:
            self.ros_thread.join(timeout=5.0)
            if self.ros_thread.is_alive():
                logger.warning("vision spin 스레드가 timeout 내에 종료되지 않았습니다.")

        if self.executor is not None and self.node is not None:
            self.executor.remove_node(self.node)
        if self.node is not None:
            self.node.destroy_node()

        logger.info("VisionBridgeManager shutdown 완료.")

    # ------------------------------------------------------------------
    def _spin_loop(self) -> None:
        assert self.executor is not None
        logger.info("Vision ROS2 spin 루프 시작.")

        while not self._stop_event.is_set():
            try:
                self.executor.spin_once(timeout_sec=SPIN_TIMEOUT_SEC)
            except rclpy.executors.ExternalShutdownException:
                logger.info("ExternalShutdownException 감지, vision spin 루프 종료.")
                self._stop_event.set()
                break
            except Exception:  # noqa: BLE001
                logger.error("vision spin_once 루프에서 예외 발생", exc_info=True)
                time.sleep(0.5)

        logger.info("Vision ROS2 spin 루프 종료.")

    # ------------------------------------------------------------------
    # 폴링 타이머 콜백 (ROS2 스레드에서 실행) - 절대 블로킹하지 않음
    # ------------------------------------------------------------------
    def _poll_once(self) -> None:
        if self._pending_call:
            # 이전 호출 응답이 아직 안 왔으면 이번 틱은 건너뜀 (중복 호출 누적 방지)
            return
        if self.client is None or not self.client.service_is_ready():
            logger.debug("get_apple_status 서비스가 아직 준비되지 않음, 이번 폴링 건너뜀")
            return

        request = SrvAppleStatus.Request()
        if self._skip_position is not None:
            if time.time() - self._skip_started_at >= SKIP_EXCLUDE_COOLDOWN_SEC:
                logger.info("스킵 제외 시간(%.0f초) 만료 - 해당 사과를 다시 후보에 포함시킴", SKIP_EXCLUDE_COOLDOWN_SEC)
                self._skip_position = None
            else:
                request.avoid_position = self._skip_position

        self._pending_call = True
        future = self.client.call_async(request)
        future.add_done_callback(self._on_response)

    def set_skip_position(self, position: Optional[list[float]]) -> None:
        """
        HITL "무시" 응답을 받았을 때 hitl_state_machine.py가 호출함. 이후 폴링부터
        SKIP_EXCLUDE_COOLDOWN_SEC 동안 obj_detection이 이 좌표 근처 사과를 최우선
        후보에서 제외하도록 한다 (다른 사과 먼저 처리되게 함).
        """
        if not position:
            self._skip_position = None
            return
        self._skip_position = position
        self._skip_started_at = time.time()

    def _on_response(self, future) -> None:
        """서비스 응답 도착 콜백 (ROS2 스레드에서 실행)."""
        self._pending_call = False
        try:
            response = future.result()
        except Exception:  # noqa: BLE001
            logger.error("get_apple_status 서비스 호출 실패", exc_info=True)
            return

        try:
            if response.status == "empty":
                # depth 판정이 경계값 근처면 "empty"/"unknown"을 한두 번씩 flicker할 수
                # 있으므로, 몇 번 연속으로 "empty"가 와야 진짜로 물체가 사라졌다고 보고
                # 디바운스 상태를 리셋한다 (한 번의 flicker로 바로 리셋하면 같은 물체가
                # 계속 "새로 감지됨"으로 취급되어 ask_human이 반복 발생함).
                self._empty_streak += 1
                if self._empty_streak >= EMPTY_RESET_STREAK:
                    self._last_status = None
                    self._last_position = None
                    self._reset_sample_window()
                return
            self._empty_streak = 0

            confidence = float(response.confidence)
            if not (0.0 <= confidence <= 1.0):
                logger.warning(
                    "Vision 서비스 confidence가 범위를 벗어남(%.3f), 이번 샘플 버림: status=%s",
                    confidence, response.status,
                )
                return

            position = list(response.position or [])
            if _is_zero_position(position):
                position = []

            # 주의: 여기서 position 흔들림을 이유로 창을 리셋하면 안 됨 - 실측으로
            # 확인된 문제(사과 하나를 옮기고 CAMERA로 복귀해도 로봇이 계속 가만히
            # 있음): RealSense depth 노이즈는 같은 사과를 보고 있어도 프레임마다
            # vision_position_dedup_threshold_mm(5mm)을 넘게 흔들리는 경우가 흔한데,
            # 창의 "첫 샘플 위치"만을 기준으로 흔들릴 때마다 리셋하면 10개를 다
            # 채우기 전에 계속 리셋되어 영원히 다수결 확정에 도달하지 못함. 이
            # 파이프라인은 main.py의 READY 게이팅상 CAMERA 위치에서 한 번에 한
            # 물체만 보이는 게 전제이므로, "다른 물체로 바뀜"은 이미 empty
            # 스트릭(위 응답 분기)으로만 판단하고 여기서는 position 유사성으로
            # 따로 리셋하지 않는다 - _sample_position_ref는 평균 계산 참고용으로만 씀.
            if self._sample_position_ref is None and position:
                self._sample_position_ref = position
            self._samples.append((response.status, confidence, position))

            if len(self._samples) < CONFIRM_SAMPLE_COUNT:
                return  # 아직 다수결 확정 전 - 큐에 안 넣고 다음 폴링을 더 모음

            majority_status, avg_confidence, avg_position = self._resolve_majority()
            self._reset_sample_window()

            if self._is_duplicate(majority_status, avg_position):
                return

            feature = _status_to_vision_feature(majority_status, avg_confidence, avg_position)

            self._last_status = majority_status
            self._last_position = avg_position

            self._dispatch(feature)

        except Exception:  # noqa: BLE001
            # ROS2 콜백에서 처리 안 된 예외가 올라가면 노드 전체가 죽을 수 있으므로
            # 반드시 여기서 잡아서 로그만 남기고 이번 메시지는 버림
            logger.error("Vision 서비스 응답 처리 중 예외 발생, 메시지를 버림", exc_info=True)

    def force_redetect(self) -> None:
        """
        직전에 큐에 넣은 (status, position) 디바운스 상태를 초기화.

        services/voice_policy.py가 음성으로 새 정책을 등록했을 때 호출함 - 카메라에
        아직 같은 사과가 그대로 있으면, 이 초기화가 없을 경우 다음 폴링에서도
        "이미 처리한 것"으로 취급되어 재큐잉되지 않고, 방금 등록한 새 정책이 그
        사과에는 절대 반영되지 않는 문제가 있었음.
        """
        self._last_status = None
        self._last_position = None
        self._reset_sample_window()

    def _is_duplicate(self, status: str, position: list[float]) -> bool:
        """직전에 큐에 넣은 (status, position)과 실질적으로 같으면 True (같은 사과)."""
        if self._last_status is None or status != self._last_status:
            return False
        return _positions_close(
            position, self._last_position or [], settings.vision_position_dedup_threshold_mm
        )

    def _reset_sample_window(self) -> None:
        """다수결 확정용 샘플 창을 비움 (확정 직후, 물체가 사라졌을 때, 도중에
        다른 물체로 바뀐 것으로 판단됐을 때 호출됨)."""
        self._samples = []
        self._sample_position_ref = None

    def _resolve_majority(self) -> tuple[str, float, list[float]]:
        """
        샘플 창(self._samples)에서 최빈값 status를 다수결로 뽑고, 그 status와
        일치하는 샘플들만으로 confidence/position을 평균낸다 (다수결에서 밀린
        소수 status의 값은 평균에 섞지 않음 - 예: 10개 중 7개가 apple_damaged,
        3개가 apple_rotten이면 apple_damaged로 확정하고 그 7개의 값만 평균).
        """
        counts = Counter(status for status, _, _ in self._samples)
        majority_status, _ = counts.most_common(1)[0]

        matching = [(conf, pos) for status, conf, pos in self._samples if status == majority_status]
        avg_confidence = sum(conf for conf, _ in matching) / len(matching)

        positions_with_value = [pos for _, pos in matching if pos]
        if positions_with_value:
            axis_count = len(positions_with_value[0])
            avg_position = [
                sum(pos[i] for pos in positions_with_value) / len(positions_with_value)
                for i in range(axis_count)
            ]
        else:
            avg_position = []

        return majority_status, avg_confidence, avg_position

    # ------------------------------------------------------------------
    def _dispatch(self, feature: VisionFeatureIn) -> None:
        """ROS2 스레드 -> 메인 이벤트 루프로 vision_queue 적재 예약."""
        if self.loop is not None and self.queue is not None:
            self.loop.call_soon_threadsafe(self._enqueue, feature)

    def _enqueue(self, feature: VisionFeatureIn) -> None:
        """메인 이벤트 루프 안에서 실행됨 (call_soon_threadsafe로 스케줄됨)."""
        try:
            self.queue.put_nowait(feature)
            logger.info(
                "Vision 감지 결과 큐 적재: fruit_type=%s defect_type=%s confidence=%.2f unknown=%s",
                feature.fruit_type, feature.defect_type, feature.confidence, feature.unknown_flag,
            )
        except QueueFull:
            logger.warning("vision_queue가 가득 참, 가장 오래된 항목을 버리고 최신 데이터를 우선함")
            try:
                self.queue.get_nowait()
            except Exception:  # noqa: BLE001
                pass
            try:
                self.queue.put_nowait(feature)
            except QueueFull:
                logger.warning("vision_queue가 여전히 가득 차 있어 메시지를 버립니다.")


# 어디서나 싱글톤처럼 접근 가능하도록 전역 인스턴스 생성 (robot_bridge.py의 bridge_manager와 동일 패턴)
vision_bridge_manager = VisionBridgeManager()
