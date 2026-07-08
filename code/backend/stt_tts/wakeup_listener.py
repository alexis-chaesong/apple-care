# 상시 웨이크워드 대기 백그라운드 태스크

"""
stt_tts/wakeup_listener.py
============================
상시 웨이크워드("헬로 로키") 대기 백그라운드 태스크.

포팅 출처: cobot2_ws/src/voice_processing (실제 로봇 환경인 ROS2 Jazzy + Python 3.12에
맞게 이미 조정된 버전)을 기준으로 이식함. origin/voice_processing(Humble 기준: tflite_runtime
고정, scipy.signal.resample, ai_edge_litert 폴백 없음)이 아니라 cobot2_ws 버전을 기준으로
삼은 이유는, 이 프로젝트가 실제로 구동되는 환경(Python 3.12라 tflite_runtime이 없어
ai_edge_litert로 폴백해야 함)이 cobot2_ws 쪽과 동일하기 때문.

ROS 패키지가 아니므로 cobot2_ws 원본이 쓰던 ament_index_python.packages.
get_package_share_directory()로 모델 경로를 찾는 방식은 쓸 수 없음 - 이 파일과 같은
디렉토리의 resource/ 아래에 tflite 모델 파일을 직접 복사해서 상대 경로로 찾음.

mic_lock을 hitl_state_machine._listen_via_stt(), voice_policy.run_voice_policy_command()와
공유해서, 마이크를 쓰는 세 주체 중 하나만 동작하도록 보장함 (mic_lock.py 참고).

주의: device_index=4, chunk/buffer_size=3840은 cobot2_ws 쪽에서 실제 마이크로 확인된
값을 그대로 가져온 것. 로봇의 마이크 장치가 바뀌면 .env의 WAKEWORD_MIC_DEVICE_INDEX만
수정하면 됨 (pyaudio.PyAudio().get_device_info_by_index(i)로 재확인 필요).
"""

import asyncio
import logging
import os
import sys
import types
from typing import Awaitable, Callable, Optional

import numpy as np

try:
    import tflite_runtime.interpreter  # Python 3.11 이하 등 tflite_runtime이 있는 환경
except ModuleNotFoundError:
    import ai_edge_litert.interpreter as litert_interpreter

    tflite_runtime = types.ModuleType("tflite_runtime")
    tflite_runtime.interpreter = litert_interpreter
    sys.modules["tflite_runtime"] = tflite_runtime
    sys.modules["tflite_runtime.interpreter"] = litert_interpreter

import pyaudio
from openwakeword.model import Model
from scipy.signal import resample_poly

from config import settings
from stt_tts.mic_lock import mic_lock

logger = logging.getLogger(__name__)

MODEL_NAME = "hello_rokey_8332_32.tflite"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "resource", MODEL_NAME)

WAKEWORD_RATE = 48000
WAKEWORD_CHUNK = 3840
WAKEWORD_CONFIDENCE_THRESHOLD = 0.3

# mic_lock이 잡혀있을 때(HITL 질문응답 중이거나 정책명령 처리 중) 재확인하는 주기.
LOCK_POLL_INTERVAL_SEC = 0.2


class WakeupWord:
    def __init__(self) -> None:
        self.model = Model(wakeword_models=[MODEL_PATH])
        self.model_name = MODEL_NAME.split(".", maxsplit=1)[0]

    def is_wakeup(self, stream: "pyaudio.Stream") -> bool:
        """블로킹 read 1회 + 예측. asyncio.to_thread에서만 호출되어야 함."""
        audio_chunk = np.frombuffer(
            stream.read(WAKEWORD_CHUNK, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample_poly(audio_chunk.astype(np.float32), 1, 3)
        audio_chunk = np.clip(audio_chunk, -32768, 32767).astype(np.int16)
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        return confidence > WAKEWORD_CONFIDENCE_THRESHOLD


def _open_stream(audio: "pyaudio.PyAudio") -> "pyaudio.Stream":
    return audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=WAKEWORD_RATE,
        input=True,
        input_device_index=settings.wakeword_mic_device_index,
        frames_per_buffer=WAKEWORD_CHUNK,
    )


class WakeupListener:
    """lifespan에서 시작/종료를 관리하는 백그라운드 태스크 매니저.
    bridge_manager, hitl_state_machine과 동일한 전역 싱글톤 패턴."""

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    def start(self, on_wakeup: Callable[[], Awaitable[None]]) -> None:
        if not settings.voice_wakeword_enabled:
            logger.info("VOICE_WAKEWORD_ENABLED=false - 웨이크워드 대기 비활성화")
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._loop(on_wakeup))

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self, on_wakeup: Callable[[], Awaitable[None]]) -> None:
        logger.info("웨이크워드 대기 루프 시작")
        audio = pyaudio.PyAudio()
        wakeup = WakeupWord()
        assert self._stop_event is not None

        try:
            while not self._stop_event.is_set():
                # mic_lock이 다른 주체(HITL 질문응답, 정책명령 처리)에 잡혀있는 동안은
                # 스트림 자체를 열지 않고 대기만 함 (동시 마이크 접근 방지)
                if mic_lock.locked():
                    await asyncio.sleep(LOCK_POLL_INTERVAL_SEC)
                    continue

                stream = await asyncio.to_thread(_open_stream, audio)
                detected = False
                try:
                    while not self._stop_event.is_set() and not mic_lock.locked():
                        detected = await asyncio.to_thread(wakeup.is_wakeup, stream)
                        if detected:
                            break
                finally:
                    stream.stop_stream()
                    stream.close()

                if detected:
                    logger.info("웨이크워드 감지됨 - 정책 명령 리스닝 시작")
                    try:
                        await on_wakeup()
                    except Exception:  # noqa: BLE001
                        logger.error("웨이크워드 트리거 처리 중 에러 발생", exc_info=True)
        except asyncio.CancelledError:
            logger.info("웨이크워드 대기 루프 취소됨")
            raise
        finally:
            audio.terminate()


wakeup_listener = WakeupListener()
