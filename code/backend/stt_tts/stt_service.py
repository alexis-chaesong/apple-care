# 마이크 입력 → 텍스트

"""
stt_tts/stt_service.py
========================
마이크로 음성을 녹음하고 OpenAI Whisper API로 텍스트 변환하는 유일한 창구.

hitl_state_machine.py가 기대하는 인터페이스:
    async def listen_once(timeout_sec: float) -> Optional[str]

설계 원칙:
  - 마이크 녹음(sounddevice)은 블로킹 I/O이므로 asyncio.to_thread()로 감싸서
    메인 이벤트 루프를 절대 막지 않게 함 (llm_service.py의 비동기 원칙과 동일)
  - VAD(음성 구간 자동 감지) 없이 MVP는 "timeout_sec 동안 고정 길이 녹음" 방식으로 시작.
    이후 무음 감지 기반 조기 종료가 필요하면 이 파일만 교체하면 됨
    (hitl_state_machine.py는 인터페이스만 알고 내부 구현은 모르므로 영향 없음)

필요 패키지: sounddevice, numpy, scipy (requirements.txt에 추가 필요)
    pip install sounddevice numpy scipy --break-system-packages

주의: 이전 프로젝트에서 Mac + Continuity Camera + OpenCV 조합이 구조적으로
문제였던 것처럼, 마이크 장치도 Mac에서는 기본 입력 장치가 예상과 다르게 잡힐 수 있음.
실제 로봇이 구동되는 Ubuntu 환경에서 반드시 별도로 장치 인덱스를 확인할 것
(sounddevice.query_devices()로 확인 가능)
"""

import asyncio
import io
import logging
import tempfile
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import AsyncOpenAI, APITimeoutError
from scipy.signal import resample_poly

from config import settings

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=settings.openai_api_key)
# llm_service.py와 별개의 클라이언트 인스턴스.
# (같은 API 키를 쓰지만 모듈 간 결합을 피하기 위해 공유하지 않고 각자 생성)


def _record_blocking(duration_sec: float, native_rate: int) -> np.ndarray:
    """
    실제 마이크 녹음을 수행하는 블로킹 함수 (하드웨어가 지원하는 native_rate로 녹음).
    asyncio.to_thread()를 통해서만 호출되어야 함 (직접 await하면 이벤트 루프가 멈춤).

    device를 명시하지 않으면 sounddevice(PortAudio)의 시스템 기본 입력 장치를 쓰는데,
    이게 실제 마이크가 아닌 엉뚱한 장치(HDMI 등)를 가리키면 말을 해도 녹음 레벨이
    거의 0이라 무음으로 판정됨 - settings.stt_mic_device_index로 명시적으로 지정 가능
    (scripts/check_mic_level.py로 올바른 장치 번호부터 확인할 것)

    반드시 하드웨어가 실제로 지원하는 샘플레이트(native_rate, 기본 48000)로 녹음해야 함.
    python310_to_312_voice_changes.md에서 확인된 문제: 마이크 장치가 16kHz 직접 캡처를
    지원하지 않는데 samplerate=16000으로 바로 요청하면 ALSA가 조용히 무음/깨진 신호를
    반환해서 Whisper가 계속 "you"로만 인식했음. listen_once()가 이후 resample_poly로
    Whisper용 sample_rate(기본 16000)로 다운샘플링함.
    """
    logger.info("마이크 녹음 시작 (%.1f초, %dHz)", duration_sec, native_rate)
    recording = sd.rec(
        int(duration_sec * native_rate),
        samplerate=native_rate,
        channels=1,
        dtype="int16",
        device=settings.stt_mic_device_index,
    )
    sd.wait()  # 녹음이 끝날 때까지 이 스레드 안에서만 대기 (메인 루프는 영향 없음)
    logger.info("마이크 녹음 종료")
    return recording


def _resample_to_target(recording: np.ndarray, native_rate: int, target_rate: int) -> np.ndarray:
    """
    native_rate로 녹음된 int16 오디오를 target_rate(Whisper용, 기본 16000)로 다운샘플링.

    반드시 float32로 변환한 뒤 resample_poly를 호출해야 함 - int16 배열을 그대로 넘기면
    SciPy 조합에 따라 출력이 전부 0(무음)으로 소실되는 경우가 있었음
    (python310_to_312_voice_changes.md 8번/11번 항목에서 실측 확인된 문제:
    수정 전 리샘플 결과 RMS 0.0, 수정 후 원본과 거의 동일한 RMS 유지).
    """
    if native_rate == target_rate:
        return recording

    from math import gcd
    divisor = gcd(native_rate, target_rate)
    up, down = target_rate // divisor, native_rate // divisor

    mono = recording[:, 0] if recording.ndim > 1 else recording
    resampled = resample_poly(mono.astype(np.float32), up, down)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


async def listen_once(timeout_sec: float) -> Optional[str]:
    """
    timeout_sec 동안 마이크로 녹음한 뒤 OpenAI Whisper API로 텍스트 변환.
    녹음/변환 중 어떤 이유로든 실패하면 None을 반환 (예외를 던지지 않음).
    hitl_state_machine.py는 None을 "이번엔 답을 못 들었다"로 해석해 재질문 로직을 탐.
    """
    native_rate = settings.stt_mic_native_rate
    target_rate = settings.stt_sample_rate

    # 녹음 길이를 timeout_sec 그대로 쓰면, 녹음 종료 시점과 hitl_state_machine.py의
    # asyncio.wait_for 타임아웃이 거의 동시에 발생해 스케줄링 오차로 결과 전송 전에
    # Task가 cancel되는 경쟁 상태가 생김. 답변 대기 시간보다 최소 3초 여유를 두어
    # Whisper API 호출/응답 시간을 확보함 (최소 1초는 보장)
    recording_duration = min(settings.stt_recording_duration_sec, max(timeout_sec - 3.0, 1.0))

    try:
        recording = await asyncio.to_thread(_record_blocking, recording_duration, native_rate)
    except Exception:  # noqa: BLE001
        logger.error("마이크 녹음 실패 (장치 확인 필요)", exc_info=True)
        return None

    recording = _resample_to_target(recording, native_rate, target_rate)

    # 녹음된 음성이 사실상 무음(볼륨 매우 낮음)인 경우 -> API 호출 자체를 생략
    # (사람이 아예 말을 안 한 경우까지 매번 API를 호출하면 비용/지연 낭비)
    if np.abs(recording).mean() < 50:  # int16 기준 경험적 임계값, 실제 환경 소음 보고 조정 필요
        logger.info("무음으로 판단되어 STT 호출 생략")
        return None

    try:
        buffer = io.BytesIO()
        sf.write(buffer, recording, target_rate, format="WAV")
        buffer.seek(0)
        buffer.name = "audio.wav"  # OpenAI SDK가 확장자로 포맷을 추론하므로 필요

        transcript = await _client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer,
            language="ko",
            timeout=settings.openai_timeout_sec,
        )
        text = transcript.text.strip()
        logger.info("STT 결과: %s", text)
        return text if text else None

    except APITimeoutError:
        logger.error("Whisper API 타임아웃")
        return None
    except Exception:  # noqa: BLE001
        logger.error("STT 변환 중 에러 발생", exc_info=True)
        return None