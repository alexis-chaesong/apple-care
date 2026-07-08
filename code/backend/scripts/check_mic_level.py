# stt_service.py가 실제로 쓰는 sounddevice 백엔드 기준 마이크 진단 스크립트
# 실행: python3 scripts/check_mic_level.py
#
# pyaudio(list_audio_devices.py)와 sounddevice는 서로 다른 오디오 백엔드라서
# 장치 번호가 일치한다는 보장이 없음. 이 스크립트로 sounddevice 기준 장치 목록과
# 실제 녹음 레벨을 직접 확인해서, stt_service.py가 진짜 마이크를 잡고 있는지 검증함.

import sys

import numpy as np
import sounddevice as sd

RECORD_SECONDS = 3
SAMPLE_RATE = 16000

# stt_service.py의 무음 판정 임계값(_record_blocking 결과에 대해
# np.abs(recording).mean() < 50 이면 무음 처리)과 동일한 값을 여기서도 사용
SILENCE_THRESHOLD = 50


def list_devices() -> None:
    print("=== sounddevice 기준 장치 목록 ===")
    print(sd.query_devices())
    print()
    try:
        default_in, default_out = sd.default.device
        print(f"현재 기본 입력 장치 index: {default_in}")
    except Exception as exc:  # noqa: BLE001
        print(f"기본 장치 조회 실패: {exc}")
    print()


def record_and_measure(device: int | None) -> None:
    print(f"--- device={device if device is not None else '(시스템 기본값)'} 로 {RECORD_SECONDS}초간 녹음 ---")
    print("지금 마이크에 대고 말해보세요...")
    recording = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=device,
    )
    sd.wait()

    level = float(np.abs(recording).mean())
    peak = int(np.abs(recording).max())
    verdict = "무음으로 판정됨 (STT 호출 생략됨)" if level < SILENCE_THRESHOLD else "음성 감지됨 (정상)"

    print(f"평균 레벨: {level:.1f}  (임계값: {SILENCE_THRESHOLD})")
    print(f"최대 레벨: {peak}")
    print(f"판정: {verdict}")
    print()


def main() -> None:
    list_devices()

    # 1) 아무 device도 지정하지 않은 경우 (stt_service.py 현재 동작과 동일)
    record_and_measure(device=None)

    # 2) 명령줄 인자로 특정 device index를 넘기면 그 장치로도 테스트
    #    예: python3 scripts/check_mic_level.py 4
    if len(sys.argv) > 1:
        record_and_measure(device=int(sys.argv[1]))


if __name__ == "__main__":
    main()
