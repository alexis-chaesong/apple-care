# IDLE→HOLD→ASKING→LISTENING→INTERPRETING→RESUME
# 로봇 Hold~Resume까지의 대화 상태 전이 관리 (가장 꼬이기 쉬운 부분이라 별도 모듈로 격리)

"""
hitl_state_machine.py
======================
로봇이 Hold된 순간부터 사람의 답을 받아 Resume하기까지의 전체 대화 흐름을 관리.

상태 전이:
    IDLE -> HOLD -> ASKING -> LISTENING -> INTERPRETING -> RESUME
                       ^                        |
                       |___(재질문, 최대 N회)_____|
                                                 |
                                      (N회 초과) -> STUCK (사람 개입 필요, HOLD 유지)

핵심 설계: STT 음성 답변과 HMI 수동 텍스트 입력을 asyncio.Future 하나로 통합.
    질문을 던지는 순간 "답변 대기용 Future"를 하나 열어두고,
    stt_service가 마이크로 들은 답이든, hitl_router.py가 HMI에서 받은 수동 입력이든
    먼저 도착하는 쪽이 submit_answer()를 통해 이 Future를 채운다.
    두 입력 경로가 서로 다른 상태를 갖지 않고 완전히 동일하게 처리되어
    "STT 듣는 중에 관리자가 텍스트를 입력하면?" 같은 경합 상황이 자연스럽게 해소됨.

동시 Unknown Object 처리 (기획서 12번 예외3 관련):
    이미 세션이 진행 중일 때 새로운 ask_human 요청이 들어오면
    거부하거나 무시하지 않고 _pending 큐에 쌓아둠. 현재 세션이 끝나면
    (성공적으로 RESUME 하든, 재질문 횟수를 다 써서 STUCK에 빠지든)
    다음 대기 항목을 이어서 처리함. MVP 범위: 동시에 여러 개를 병렬로
    질문하지는 않음 (사람이 한 번에 하나씩만 답할 수 있다고 가정).

STT/TTS 모듈에 기대하는 인터페이스 (다음 단계에서 구현 예정):
    stt_tts.stt_service.listen_once(timeout_sec: float) -> Optional[str]
        - 지정 시간 내에 음성을 텍스트로 변환해 반환, 못 들으면 None
    stt_tts.tts_service.speak(text: str) -> None (async)
        - 문장을 음성으로 출력, 출력이 끝날 때까지 대기
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Optional

from config import settings
from robot_bridge import bridge_manager
from connection_manager import connection_manager
from services import llm_service
from services.llm_service import LLMTimeoutError, LLMParseError, LLMAuthError
from services.bayesian_policy import record_human_feedback

logger = logging.getLogger(__name__)


class HITLState(str, Enum):
    IDLE = "IDLE"
    HOLD = "HOLD"
    ASKING = "ASKING"
    LISTENING = "LISTENING"
    INTERPRETING = "INTERPRETING"
    STUCK = "STUCK"
    # STUCK: 재질문 한도를 다 썼는데도 해석 실패. 로봇은 계속 Hold 상태.
    # 여기서 벗어나려면 관리자가 /api/robot/reset을 명시적으로 호출해야 함
    # (자동으로 아무거나 추측해서 진행하면 절대 안 되므로 의도적으로 만든 막다른 상태)


@dataclass
class HITLSession:
    fruit_type: str
    condition: str
    vision_confidence: float
    position: Optional[list[float]] = None
    # Vision이 이 사과를 감지했을 때의 카메라 좌표 (VisionFeatureIn.center 그대로).
    # 답변이 해석되면 이 좌표를 그대로 실어서 /decision/result로 발행해야
    # 로봇 쪽(apple_sorting_cycle.py)이 실제로 이 사과를 집어서 옮길 수 있음.
    state: HITLState = HITLState.IDLE
    attempt: int = 0
    session_id: str = dc_field(default_factory=lambda: __import__("uuid").uuid4().hex[:8])


class HITLStateMachine:
    """
    프로젝트 전체에서 단 하나만 존재하는 싱글톤.
    bridge_manager, connection_manager와 동일한 전역 인스턴스 패턴.
    """

    def __init__(self) -> None:
        self._current: Optional[HITLSession] = None
        self._answer_future: Optional[asyncio.Future] = None
        self._pending: deque[tuple[str, str, float, Optional[list[float]]]] = deque()
        self._lock = asyncio.Lock()
        # _lock: _current와 _pending을 동시에 여러 코루틴이 건드리는 것을 방지.
        # (예: consumer loop가 새 ask_human을 넣는 동시에 세션 종료 처리가 겹치는 경우)

    # ------------------------------------------------------------------
    # 외부(consumer loop, decision_planner 호출부)에서 부르는 진입점
    # ------------------------------------------------------------------
    async def handle_ask_human(
        self,
        fruit_type: str,
        condition: str,
        vision_confidence: float,
        position: Optional[list[float]] = None,
    ) -> None:
        """
        decision_planner.decide()가 action="ask_human"을 반환했을 때 호출.
        이미 세션이 진행 중이면 대기열에 쌓고, 아니면 즉시 세션을 시작.
        블로킹하지 않도록 백그라운드 task로 실행됨 (main.py의 consumer loop가
        HITL 대화가 끝날 때까지 다른 Vision 프레임 처리를 멈추지 않게 하기 위함).

        position: DecisionResult.position(=Vision이 감지한 카메라 좌표)을 그대로
        전달받아 세션에 보관함 - 답변이 해석된 뒤 이 좌표 없이 RESUME만 보내면
        로봇이 실제로 어느 사과를 어디로 옮겨야 하는지 알 수 없어 아무 동작도
        하지 않는 문제가 있었음 (아래 _ask_and_wait 참고).
        """
        async with self._lock:
            if self._current is not None:
                logger.info(
                    "HITL 세션 진행 중, 대기열에 추가: fruit=%s condition=%s",
                    fruit_type, condition,
                )
                self._pending.append((fruit_type, condition, vision_confidence, position))
                return
            self._current = HITLSession(
                fruit_type=fruit_type,
                condition=condition,
                vision_confidence=vision_confidence,
                position=position,
            )

        asyncio.create_task(self._run_session())

    # ------------------------------------------------------------------
    # 답변 제출 (STT 백그라운드 리스너 / hitl_router.py 양쪽에서 공통 호출)
    # ------------------------------------------------------------------
    def submit_answer(self, raw_answer: str) -> bool:
        """
        음성이든 HMI 수동 입력이든, 답변 텍스트가 도착하면 이 함수 하나로 제출.
        현재 열려있는 답변 대기 Future가 없으면(질문 중이 아니면) False를 반환하고 무시.
        """
        if self._answer_future is not None and not self._answer_future.done():
            self._answer_future.set_result(raw_answer)
            return True
        logger.warning("현재 대기 중인 HITL 질문이 없는데 답변이 도착함: %s", raw_answer)
        return False

    @property
    def current_session(self) -> Optional[HITLSession]:
        """hitl_router.py가 '지금 이 답변이 어떤 fruit_type/condition에 대한 것인지'
        알아야 할 때 조회용으로 사용"""
        return self._current

    # ------------------------------------------------------------------
    # 관리자 강제 개입 (POST /api/robot/reset)
    # ------------------------------------------------------------------
    async def force_reset(self, destination: Optional[str] = None) -> dict:
        """
        관리자가 STUCK(또는 진행 중인 어떤 세션이든) 상태에서 강제로 벗어나게 함.

        destination이 주어지면: 관리자가 최종 판단을 내린 것으로 간주해서
        record_human_feedback()으로 정책에 반영 (raw_answer="[관리자 강제 리셋]"으로 기록).
        destination이 없으면: 정책 변경 없이 그냥 현재 세션만 정리하고 로봇 재개.

        어느 경우든 반드시 RESUME을 발행하고, 대기 중인 다음 HITL 요청이 있으면
        이어서 시작함.
        """
        async with self._lock:
            if self._current is None:
                return {"result": "NO_ACTIVE_SESSION"}
            session = self._current

        result_destination = None
        if destination is not None:
            updated = record_human_feedback(
                fruit_type=session.fruit_type,
                condition=session.condition,
                destination=destination,
                raw_answer="[관리자 강제 리셋]",
                session_id=session.session_id,
            )
            result_destination = updated["destination"]

        if result_destination is not None:
            # 관리자가 destination을 지정한 경우에만 실제로 옮기게 함 (지정 안 하면
            # 이 사과는 그냥 세션만 정리하고 로봇 판단에 맡기지 않음 - 아래 comment 참고)
            bridge_manager.publish_decision_result(
                fruit=session.fruit_type,
                destination=result_destination,
                pose=session.position,
                reason="admin_force_reset",
                condition=session.condition,
            )
        logger.warning(
            "관리자 강제 리셋 (session_id=%s): destination=%s",
            session.session_id, result_destination,
        )

        # 세션이 STUCK이 아니라 아직 ASKING/LISTENING 등으로 실제 실행 중이었을 경우
        # (관리자가 STUCK을 기다리지 않고 진행 중인 세션을 강제로 끊는 경우),
        # _run_session() task가 답변 Future를 기다리며 멈춰있을 수 있음.
        # 그냥 self._current만 비우면 그 task는 최대 hitl_response_timeout_sec초 동안
        # 좀비처럼 계속 돌면서 새로 시작된 다음 세션과 마이크/스피커를 두고 경합할 수 있으므로,
        # 대기 중인 Future를 명시적으로 취소해서 즉시 깨어나 스스로 멈추게 함
        # (_run_session()의 finally에 있는 should_cleanup 체크가 이 task의 중복 정리를 막아줌)
        if self._answer_future is not None and not self._answer_future.done():
            self._answer_future.cancel()
        self._answer_future = None

        async with self._lock:
            self._current = None
        await self._start_next_pending()

        return {"result": "RESET_DONE", "session_id": session.session_id, "destination": result_destination}

    # ------------------------------------------------------------------
    # 내부 세션 실행 루프
    # ------------------------------------------------------------------
    async def _run_session(self) -> None:
        session = self._current
        assert session is not None

        bridge_manager.publish_command(
            command_type="HOLD",
            payload={"fruit_type": session.fruit_type, "condition": session.condition},
        )
        session.state = HITLState.HOLD
        logger.info(
            "HITL 세션 시작 (session_id=%s): fruit=%s condition=%s",
            session.session_id, session.fruit_type, session.condition,
        )
        # VLA_ASK_HUMAN은 여기서 "실제로 세션이 시작될 때" 딱 한 번만 브로드캐스트한다.
        # (예전엔 consumer loop가 ask_human 판정이 나올 때마다 매번 브로드캐스트했는데,
        # 이미 세션이 진행 중이면 handle_ask_human()이 _pending에 쌓기만 하고 여기까지
        # 안 오므로, vision이 flicker로 같은 물체를 반복 보고해도 팝업이 스팸처럼
        # 계속 새로 뜨지 않음. 대기열에서 다음 항목이 시작될 때도 자연히 한 번 더 뜬다.)
        await connection_manager.broadcast({
            "type": "VLA_ASK_HUMAN",
            "payload": {
                "session_id": session.session_id,
                "fruit_type": session.fruit_type,
                "condition": session.condition,
                "confidence": session.vision_confidence,
            },
            "timestamp": time.time(),
        })

        try:
            while session.attempt < settings.hitl_max_reask_attempts:
                resolved = await self._ask_and_wait(session)
                if resolved:
                    return  # 성공적으로 해석/저장/RESUME까지 완료
                session.attempt += 1
                logger.info(
                    "재질문 진행 (session_id=%s, attempt=%d/%d)",
                    session.session_id, session.attempt, settings.hitl_max_reask_attempts,
                )

            # 재질문 한도 초과 -> STUCK. 절대 자동으로 아무 destination이나 추측하지 않음
            session.state = HITLState.STUCK
            logger.error(
                "HITL 세션 STUCK (session_id=%s): fruit=%s condition=%s - 관리자 개입 필요",
                session.session_id, session.fruit_type, session.condition,
            )
            # 로봇은 계속 HOLD 상태 유지. 관리자가 /api/robot/reset으로 직접 개입해야 함
            # (아래 finally에서 self._current를 정리하지 않고 그대로 남겨둠 - 그래야
            # force_reset()이 current_session으로 이 세션을 찾아서 복구할 수 있음)
            await connection_manager.broadcast({
                "type": "HITL_STUCK",
                "payload": {
                    "session_id": session.session_id,
                    "fruit_type": session.fruit_type,
                    "condition": session.condition,
                },
                "timestamp": time.time(),
            }) 
            return

        except Exception:
            # 예상 못한 예외(버그, 네트워크 문제 등)로 asyncio Task가 조용히 죽어버리면
            # session.state가 HOLD/ASKING/LISTENING 등 어중간한 상태로 멈춰서
            # 로봇이 영원히 Hold된 채 아무 로그도 안 남는 최악의 상황이 됨.
            # 그런 경우에도 반드시 STUCK으로 안전하게 귀결시킴 (never fail silently)
            session.state = HITLState.STUCK
            logger.critical(
                "HITL 세션 처리 중 예기치 못한 에러로 STUCK 처리 (session_id=%s)",
                session.session_id, exc_info=True,
            )
            await connection_manager.broadcast({
                "type": "HITL_STUCK",
                "payload": {
                    "session_id": session.session_id,
                    "fruit_type": session.fruit_type,
                    "condition": session.condition,
                },
                "timestamp": time.time(),
            })
            return

        finally:
            # should_cleanup이 필요한 이유(force_reset()과의 경합 방지):
            #   - session.state == STUCK: 위에서 의도적으로 정리를 건너뛴 경우이므로
            #     여기서도 건너뛰어야 함 (STUCK 세션은 force_reset()이 정리할 때까지 보존)
            #   - self._current is not session: force_reset()이 이미 이 세션을 강제로
            #     정리(self._current=None, _start_next_pending() 호출)한 뒤이므로,
            #     뒤늦게 깨어난 이 task가 또 정리하거나 다음 세션을 중복 시작하면 안 됨
            #     (예: LISTENING 중 force_reset()이 답변 Future를 cancel()해서 이
            #     task가 CancelledError로 깨어나는 경우가 여기 해당)
            should_cleanup = (self._current is session) and (session.state != HITLState.STUCK)
            if should_cleanup:
                self._answer_future = None
                async with self._lock:
                    self._current = None
                await self._start_next_pending()

    async def _ask_and_wait(self, session: HITLSession) -> bool:
        """
        질문 1회 던지고 답을 기다려 해석/저장까지 시도.
        성공(RESUME까지 완료)하면 True, 실패(재질문 필요)하면 False 반환.
        """
        # 답변 Future를 질문 생성/TTS보다 먼저 만들어둠. VLA_ASK_HUMAN 브로드캐스트는
        # _run_session()에서 이미 TTS보다 훨씬 먼저 나가서 HMI 팝업이 즉시 뜨는데,
        # 이 Future가 TTS가 다 끝난 뒤에야 생기면 사람이 팝업을 보자마자(TTS가 아직
        # 재생 중일 때) 버튼을 눌렀을 때 submit_answer()가 "아직 답변 대기 중이 아님"으로
        # 판단해 무조건 409를 반환하는 문제가 있었음 (실제로 겪은 버그 - HMI에서
        # 팝업 누르면 오류가 뜸). 미리 만들어두면 ASKING 단계에서 답이 와도 그대로 받힘.
        self._answer_future = asyncio.get_event_loop().create_future()

        # 1) 질문 생성 및 TTS 출력
        session.state = HITLState.ASKING
        try:
            fruit_guess = session.fruit_type if session.fruit_type != "unknown" else None
            question = await llm_service.generate_unknown_question(fruit_guess)
        except (LLMTimeoutError, LLMParseError, LLMAuthError):
            # 질문 생성 자체가 실패하면 정형화된 fallback 문장 사용
            # (여기서마저 실패하면 로봇이 영원히 조용히 멈춰있게 되므로 반드시 fallback 필요)
            logger.warning("질문 생성 실패, fallback 문장 사용 (session_id=%s)", session.session_id)
            question = f"{session.fruit_type} 처리 방법을 확인해 주세요. 정상, 가공, 폐기 중 선택해 주세요."

        from stt_tts import tts_service  # 지연 import: 순환 참조 방지 및 아직 미구현 상태 대비

        # TTS가 재생되는 동안에도 이미 위에서 만들어둔 Future가 답을 받을 수 있으므로,
        # HMI에서 TTS보다 먼저 버튼을 눌러도 더 이상 유실되지 않음. 단, 이 시점에
        # 이미 답이 와서 Future가 끝나 있으면 굳이 TTS를 끝까지 들려줄 필요 없음.
        if not self._answer_future.done():
            await tts_service.speak(question)

        # 2) 답변 대기 (STT 백그라운드 + 수동 입력 동시 대기)
        session.state = HITLState.LISTENING
        stt_task = asyncio.create_task(self._listen_via_stt())

        try:
            raw_answer = await asyncio.wait_for(
                self._answer_future, timeout=settings.hitl_response_timeout_sec
            )
        except asyncio.TimeoutError:
            logger.info("답변 대기 타임아웃 (session_id=%s)", session.session_id)
            return False
        finally:
            stt_task.cancel()

        # 3) LLM 해석
        session.state = HITLState.INTERPRETING
        try:
            interpreted = await llm_service.interpret_human_answer(raw_answer)
        except (LLMTimeoutError, LLMParseError, LLMAuthError):
            logger.warning("답변 해석 실패 (session_id=%s): %s", session.session_id, raw_answer)
            return False

        destination = interpreted.get("destination")
        if destination is None:
            logger.info("애매한 답변으로 재질문 필요 (session_id=%s): %s", session.session_id, raw_answer)
            return False

        # 4) 정책 학습 반영 + 로봇 재개
        updated_policy = record_human_feedback(
            fruit_type=session.fruit_type,
            condition=session.condition,
            destination=destination,
            raw_answer=raw_answer,
            session_id=session.session_id,
        )
        # 로봇 쪽(apple_sorting_cycle.py)은 /robot/command의 RESUME을 실제로 처리하지
        # 않고 경고 로그만 남기고 무시하도록 되어 있음 (아직 그 부분은 미구현). 로봇의
        # decision_queue는 오직 /decision/result만 구독해서 pick&place를 실행하므로,
        # 이 사과를 실제로 옮기게 하려면 "execute" 경로와 동일하게 반드시
        # publish_decision_result()로 보내야 함 - publish_command(RESUME)만 보내면
        # 정책은 저장되지만 로봇은 아무 동작도 하지 않는 버그가 있었음.
        bridge_manager.publish_decision_result(
            fruit=session.fruit_type,
            destination=destination,
            pose=session.position,
            reason="human_feedback",
            condition=session.condition,
        )
        logger.info(
            "HITL 세션 완료 (session_id=%s): destination=%s confidence=%.2f",
            session.session_id, destination, updated_policy["confidence"],
        )
        await connection_manager.broadcast({
            "type": "HITL_RESOLVED",
            "payload": {
                "session_id": session.session_id,
                "fruit_type": session.fruit_type,
                "condition": session.condition,
                "destination": destination,
                "confidence": updated_policy["confidence"],
            },
            "timestamp": time.time(),
        })
        return True

    async def _listen_via_stt(self) -> None:
        """STT로 음성을 듣고, 인식되면 submit_answer()로 Future를 채움.
        HMI 수동 입력이 먼저 도착해 Future가 이미 닫혔다면 조용히 무시됨.

        mic_lock으로 감싸는 이유: 웨이크워드 대기 루프(wakeup_listener.py)나 음성
        정책 명령(voice_policy.py)이 동시에 같은 마이크 장치를 열지 못하게 막기
        위함. 이미 다른 주체가 마이크를 쓰고 있으면 그게 끝날 때까지 대기했다가
        듣기 시작함 (hitl_response_timeout_sec 안에 안 끝나면 시간 초과로 재질문)."""
        from stt_tts import stt_service  # 지연 import
        from stt_tts.mic_lock import mic_lock

        try:
            async with mic_lock:
                raw = await stt_service.listen_once(timeout_sec=settings.hitl_response_timeout_sec)
            if raw:
                self.submit_answer(raw)
        except asyncio.CancelledError:
            # 수동 입력이 먼저 도착해 정상적으로 취소된 경우 - 에러 아님
            raise
        except Exception:  # noqa: BLE001
            logger.error("STT 리스닝 중 예외 발생", exc_info=True)

    async def _start_next_pending(self) -> None:
        """현재 세션이 끝난 뒤, 대기열에 쌓인 다음 ask_human 요청을 이어서 시작.

        곧바로 시작하지 않고 hitl_batch_grace_sec만큼 기다렸다가 시작함 - 방금
        세션에 답하거나 지켜보던 관리자가 화면에서 눈을 뗄 여유도 없이 바로 다음
        질문의 재질문 카운트다운이 시작되면 타이밍을 놓쳐 STUCK에 빠지기 쉬움
        (실제로 겪은 문제). _current를 먼저 채워서 그 사이 들어오는 새 ask_human
        요청은 여전히 정상적으로 대기열 뒤에 쌓이게 함.
        """
        async with self._lock:
            if not self._pending or self._current is not None:
                return
            fruit_type, condition, confidence, position = self._pending.popleft()
            self._current = HITLSession(
                fruit_type=fruit_type, condition=condition, vision_confidence=confidence,
                position=position,
            )

        if settings.hitl_batch_grace_sec > 0:
            await asyncio.sleep(settings.hitl_batch_grace_sec)

        asyncio.create_task(self._run_session())


# 전역 싱글톤 인스턴스
hitl_state_machine = HITLStateMachine()