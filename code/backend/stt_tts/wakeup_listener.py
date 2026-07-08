# 상시 웨이크워드 대기 백그라운드 태스크

"""
stt_tts/wakeup_listener.py
============================
상시 웨이크워드("헬로 로키") 대기 백그라운드 태스크.

포팅 출처: cobot2_ws/src/voice_processing (ROS2 Jazzy + Python 3.12 환경에 맞게 조정된
버전)을 기준으로 이식하되, 팀원마다 ROS 배포판(Humble/Jazzy)과 마이크 하드웨어가 다를 수
있으므로 특정 환경에만 맞는 하드코딩은 전부 설정값/자동계산으로 바꿔서 양쪽 다 돌아가게 함:

  - tflite 인터프리터: Humble(Python 3.10/3.11)은 보통 tflite_runtime 설치 가능,
    Jazzy(Python 3.12)는 tflite_runtime wheel이 없어 ai_edge_litert로 폴백해야 함
    -> try/except로 자동 분기 (아래)
  - 마이크 native 샘플레이트: 48kHz 고정 가정 대신 settings.wakeword_mic_native_rate로
    설정하고, openwakeword가 요구하는 16kHz 프레임(1280 샘플=80ms)에 맞춰 chunk 크기와
    리샘플 비율을 매 실행 시 자동 계산함 (다른 샘플레이트 마이크를 쓰는 동료도 .env
    설정값만 바꾸면 코드 수정 없이 동작)

ROS 패키지가 아니므로 cobot2_ws 원본이 쓰던 ament_index_python.packages.
get_package_share_directory()로 모델 경로를 찾는 방식은 쓸 수 없음 - 이 파일과 같은
디렉토리의 resource/ 아래에 tflite 모델 파일을 직접 복사해서 상대 경로로 찾음.

mic_lock을 hitl_state_machine._listen_via_stt(), voice_policy.run_voice_policy_command()와
공유해서, 마이크를 쓰는 세 주체 중 하나만 동작하도록 보장함 (mic_lock.py 참고).

주의: device_index=4는 지금까지 확인된 로봇 환경 기준 기본값일 뿐, 마이크 장치가 바뀌면
.env의 WAKEWORD_MIC_DEVICE_INDEX만 수정하면 됨 (pyaudio.PyAudio().get_device_info_by_index(i)로
재확인 필요, scripts/list_audio_devices.py 참고).
"""

import asyncio
import logging
import os
import sys
import types
from math import gcd
from typing import Awaitable, Callable, Optional

import numpy as np

try:
    import tflite_runtime.interpreter  # Humble 등 Python 3.11 이하, tflite_runtime이 있는 환경
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

# openwakeword의 처리 단위: 16kHz에서 1280샘플(=80ms). 마이크 native rate가 몇이든
# 이 프레임 길이(80ms)에 맞는 chunk 크기를 아래에서 자동 계산함.
OPENWAKEWORD_FRAME_SAMPLES_16K = 1280
OPENWAKEWORD_TARGET_RATE = 16000
WAKEWORD_CONFIDENCE_THRESHOLD = 0.3

# mic_lock이 잡혀있을 때(HITL 질문응답 중이거나 정책명령 처리 중) 재확인하는 주기.
LOCK_POLL_INTERVAL_SEC = 0.2


def _wakeword_chunk_size(native_rate: int) -> int:
    """native_rate 마이크에서 80ms(=16kHz 기준 1280샘플)에 해당하는 샘플 수.
    예: 48000Hz -> 3840, 16000Hz -> 1280, 44100Hz -> 3528"""
    return native_rate * OPENWAKEWORD_FRAME_SAMPLES_16K // OPENWAKEWORD_TARGET_RATE


class WakeupWord:
    def __init__(self, native_rate: int) -> None:
        self.model = Model(wakeword_models=[MODEL_PATH])
        self.model_name = MODEL_NAME.split(".", maxsplit=1)[0]
        self.native_rate = native_rate
        self.chunk = _wakeword_chunk_size(native_rate)

        if native_rate == OPENWAKEWORD_TARGET_RATE:
            self._up, self._down = 1, 1
        else:
            divisor = gcd(native_rate, OPENWAKEWORD_TARGET_RATE)
            self._up, self._down = OPENWAKEWORD_TARGET_RATE // divisor, native_rate // divisor

    def is_wakeup(self, stream: "pyaudio.Stream") -> bool:
        """블로킹 read 1회 + 예측. asyncio.to_thread에서만 호출되어야 함."""
        audio_chunk = np.frombuffer(
            stream.read(self.chunk, exception_on_overflow=False),
            dtype=np.int16,
        )
        if self._up != 1 or self._down != 1:
            # int16 배열을 그대로 resample_poly에 넘기면 SciPy 조합에 따라 출력이
            # 0(무음)으로 소실되는 경우가 있어 반드시 float32로 먼저 변환함
            # (python310_to_312_voice_changes.md 8번 항목에서 실측 확인된 문제)
            audio_chunk = resample_poly(audio_chunk.astype(np.float32), self._up, self._down)
            audio_chunk = np.clip(audio_chunk, -32768, 32767).astype(np.int16)
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        return confidence > WAKEWORD_CONFIDENCE_THRESHOLD


def _open_stream(audio: "pyaudio.PyAudio", native_rate: int, chunk: int) -> "pyaudio.Stream":
    return audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=native_rate,
        input=True,
        input_device_index=settings.wakeword_mic_device_index,
        frames_per_buffer=chunk,
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
        native_rate = settings.wakeword_mic_native_rate
        logger.info(
            "웨이크워드 대기 루프 시작 (device=%d, native_rate=%dHz, chunk=%d)",
            settings.wakeword_mic_device_index, native_rate, _wakeword_chunk_size(native_rate),
        )
        audio = pyaudio.PyAudio()
        wakeup = WakeupWord(native_rate)
        assert self._stop_event is not None

        try:
            while not self._stop_event.is_set():
                # mic_lock이 다른 주체(HITL 질문응답, 정책명령 처리)에 잡혀있는 동안은
                # 스트림 자체를 열지 않고 대기만 함 (동시 마이크 접근 방지)
                if mic_lock.locked():
                    await asyncio.sleep(LOCK_POLL_INTERVAL_SEC)
                    continue

                stream = await asyncio.to_thread(_open_stream, audio, native_rate, wakeup.chunk)
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
