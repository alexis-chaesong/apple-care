# TTS 출력 
# 텍스트 → 음성 출력

"""
stt_tts/tts_service.py
========================
텍스트를 OpenAI TTS API로 음성 변환하고 로컬 스피커로 재생하는 유일한 창구.

hitl_state_machine.py가 기대하는 인터페이스:
    async def speak(text: str) -> None   (재생이 끝날 때까지 대기 후 반환)

설계 원칙:
  - 파일 재생(subprocess)도 블로킹이므로 asyncio.create_subprocess_exec으로
    비동기 프로세스를 띄우고 await로 종료를 기다림 (이벤트 루프 블로킹 없음)
  - 재생 플레이어 명령어를 config로 뺌 (OS마다 다름 - Mac 개발환경 vs
    실제 로봇이 구동되는 Ubuntu 환경이 다르므로 .env로만 전환 가능하게 함)

필요 패키지: 없음 (OpenAI SDK만 있으면 됨). 단, 시스템에 오디오 플레이어 필요:
    Ubuntu: ffmpeg 설치 시 함께 오는 ffplay 사용 권장 (sudo apt install ffmpeg)
    Mac 개발 환경: afplay는 OS 기본 내장이라 별도 설치 불필요
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from openai import AsyncOpenAI, APITimeoutError

from config import settings

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=settings.openai_api_key)


async def speak(text: str) -> None:
    """
    텍스트를 음성으로 합성해 재생. 재생이 끝날 때까지 대기한 뒤 반환하므로,
    hitl_state_machine.py는 이 함수가 끝난 시점부터 바로 STT 리스닝을 시작해도
    로봇의 질문 음성과 겹치지 않음이 보장됨.

    합성/재생 중 에러가 나도 예외를 위로 던지지 않고 로그만 남김 -
    TTS 실패로 전체 HITL 흐름이 죽어서는 안 되므로 (질문이 안 들려도
    사람이 화면(HMI)의 텍스트를 보고 답할 수 있어야 함 - graceful degradation)
    """
    logger.info("TTS 합성 시작: %s", text)

    try:
        response = await _client.audio.speech.create(
            model="tts-1",
            voice=settings.tts_voice,
            input=text,
            timeout=settings.openai_timeout_sec,
        )
    except APITimeoutError:
        logger.error("TTS 합성 타임아웃 - 음성 출력 없이 넘어감")
        return
    except Exception:  # noqa: BLE001
        logger.error("TTS 합성 중 에러 발생 - 음성 출력 없이 넘어감", exc_info=True)
        return

    # 임시 mp3 파일로 저장 후 재생.
    # (스트리밍 재생도 가능하지만 MVP 단계에서는 파일 기반이 훨씬 단순하고 안정적)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(response.content)
        temp_path = Path(f.name)

    try:
        await _play_audio_file(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


async def _play_audio_file(path: Path) -> None:
    """
    settings.tts_audio_player 명령으로 오디오 파일 재생.
    비동기 subprocess로 실행해 재생 중에도 이벤트 루프가 다른 작업을 처리할 수 있게 함
    (다만 speak()가 이 함수를 await로 기다리므로, 호출한 쪽(hitl_state_machine)
    입장에서는 "재생이 끝나야 다음 단계로 진행"하는 순서가 그대로 보장됨)
    """
    player_cmd = settings.tts_audio_player  # 예: "afplay" (Mac) 또는 "ffplay" (Ubuntu)

    if player_cmd == "ffplay":
        args = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
    else:
        args = [player_cmd, str(path)]

    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()
    except FileNotFoundError:
        logger.error(
            "오디오 플레이어(%s)를 찾을 수 없습니다. "
            "Ubuntu는 'sudo apt install ffmpeg', Mac은 별도 설치 불필요.",
            player_cmd,
        )
    except Exception:  # noqa: BLE001
        logger.error("오디오 재생 중 에러 발생", exc_info=True)