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

from config import settings

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=settings.openai_api_key)
# llm_service.py와 별개의 클라이언트 인스턴스.
# (같은 API 키를 쓰지만 모듈 간 결합을 피하기 위해 공유하지 않고 각자 생성)


def _record_blocking(duration_sec: float, sample_rate: int) -> np.ndarray:
    """
    실제 마이크 녹음을 수행하는 블로킹 함수.
    asyncio.to_thread()를 통해서만 호출되어야 함 (직접 await하면 이벤트 루프가 멈춤).
    """
    logger.info("마이크 녹음 시작 (%.1f초)", duration_sec)
    recording = sd.rec(
        int(duration_sec * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()  # 녹음이 끝날 때까지 이 스레드 안에서만 대기 (메인 루프는 영향 없음)
    logger.info("마이크 녹음 종료")
    return recording


async def listen_once(timeout_sec: float) -> Optional[str]:
    """
    timeout_sec 동안 마이크로 녹음한 뒤 OpenAI Whisper API로 텍스트 변환.
    녹음/변환 중 어떤 이유로든 실패하면 None을 반환 (예외를 던지지 않음).
    hitl_state_machine.py는 None을 "이번엔 답을 못 들었다"로 해석해 재질문 로직을 탐.
    """
    sample_rate = settings.stt_sample_rate

    try:
        recording = await asyncio.to_thread(_record_blocking, timeout_sec, sample_rate)
    except Exception:  # noqa: BLE001
        logger.error("마이크 녹음 실패 (장치 확인 필요)", exc_info=True)
        return None

    # 녹음된 음성이 사실상 무음(볼륨 매우 낮음)인 경우 -> API 호출 자체를 생략
    # (사람이 아예 말을 안 한 경우까지 매번 API를 호출하면 비용/지연 낭비)
    if np.abs(recording).mean() < 50:  # int16 기준 경험적 임계값, 실제 환경 소음 보고 조정 필요
        logger.info("무음으로 판단되어 STT 호출 생략")
        return None

    try:
        buffer = io.BytesIO()
        sf.write(buffer, recording, sample_rate, format="WAV")
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