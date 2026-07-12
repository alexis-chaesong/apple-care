# 마이크 장치를 숫자 인덱스 또는 이름으로 찾는 유틸

"""
stt_tts/mic_device.py
========================
PipeWire가 오디오를 관리하는 환경(이번에 확인된 Humble 노트북 등)에서는 ALSA raw
서브디바이스(hw:X,Y)의 PortAudio/pyaudio 장치 인덱스가 마이크 잭 감지·재연결
시점마다 바뀐다 (실측: 같은 프로세스 재실행 사이에 index 4 -> 7 -> 4로 계속 이동).
그래서 숫자 인덱스를 .env(WAKEWORD_MIC_DEVICE_INDEX 등)에 고정해도 다시 어긋날 수 있음.

이런 환경에서는 PulseAudio가 노출하는 안정적인 가상 장치("pulse")를 이름으로 찾아
쓰면 된다 - PulseAudio 쪽에서 실제 물리 마이크 라우팅을 알아서 처리하므로 밑단
ALSA 서브디바이스 번호가 흔들려도 영향을 받지 않는다.

기존 환경(숫자 인덱스가 안정적인 하드웨어, 예: Jazzy 쪽)의 동작은 그대로 두고,
*_MIC_DEVICE_NAME 환경변수가 설정된 경우에만 이름 검색을 우선 적용하는 방식으로
분기한다 (설정 안 하면 기존 *_MIC_DEVICE_INDEX 그대로 사용).

장치 인덱스가 런타임에 흔들리는 문제이므로, 캐싱하지 않고 스트림을 열 때마다
매번 새로 검색해야 함.
"""

import sounddevice as sd


def resolve_sd_device(name: str | None, index_fallback):
    """sounddevice(PortAudio) 기준으로 입력 장치를 찾음. name이 없으면 index_fallback 그대로 반환."""
    if not name:
        return index_fallback
    for idx, info in enumerate(sd.query_devices()):
        if name.lower() in info["name"].lower() and info["max_input_channels"] > 0:
            return idx
    raise RuntimeError(f"sounddevice: 이름에 '{name}'을(를) 포함하는 입력 장치를 찾지 못함")


def resolve_pyaudio_device(audio, name: str | None, index_fallback):
    """pyaudio(ALSA) 기준으로 입력 장치를 찾음. name이 없으면 index_fallback 그대로 반환."""
    if not name:
        return index_fallback
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if name.lower() in info["name"].lower() and info["maxInputChannels"] > 0:
            return i
    raise RuntimeError(f"pyaudio: 이름에 '{name}'을(를) 포함하는 입력 장치를 찾지 못함")
