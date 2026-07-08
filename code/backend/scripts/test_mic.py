# 마이크만 단독으로 테스트하는 스크립트 (백엔드 앱 코드에 의존하지 않음)
#
# 녹음 -> 파형 그래프 저장 -> 레벨 측정 -> 재생까지 한 번에 해서,
# 숫자(레벨)와 그래프 둘 다로 실제 목소리가 녹음됐는지 확인할 수 있게 함.
#
# 중요: 반드시 마이크의 실제 지원 샘플레이트(native rate, 기본 48000)로 녹음한 뒤
# Whisper용 16000으로 소프트웨어 리샘플링함 (stt_service.py와 동일한 방식).
# 16000을 장치에 직접 요청하면 ALSA가 지원 안 되는 레이트에서 조용히 무음/깨진
# 신호를 반환하는 경우가 있었음 (python310_to_312_voice_changes.md 11번 항목 참고).
#
# 사용법:
#   python3 scripts/test_mic.py            # 기본 장치(index 4), 48kHz로 녹음
#   python3 scripts/test_mic.py 4          # 장치 index 4 명시
#   python3 scripts/test_mic.py 4 6        # 장치 index 4, 6초간 녹음
#   python3 scripts/test_mic.py 4 6 44100  # 장치의 native rate가 48000이 아니면 직접 지정

import subprocess
import sys
import tempfile
from math import gcd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 원격/헤드리스 환경에서도 창 없이 파일로 저장되게 함
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

DEFAULT_DEVICE = 4
DEFAULT_RECORD_SECONDS = 4.0
DEFAULT_NATIVE_RATE = 48000
TARGET_RATE = 16000  # Whisper에 실제로 보내는 레이트 (stt_service.py와 동일)
SILENCE_THRESHOLD = 50  # stt_service.py의 무음 판정 기준과 동일

PLOT_PATH = Path(__file__).parent / "test_mic_result.png"
WAV_PATH = Path(__file__).parent / "test_mic_result.wav"


def resample_to_target(recording: np.ndarray, native_rate: int, target_rate: int) -> np.ndarray:
    if native_rate == target_rate:
        return recording
    divisor = gcd(native_rate, target_rate)
    up, down = target_rate // divisor, native_rate // divisor
    mono = recording[:, 0] if recording.ndim > 1 else recording
    resampled = resample_poly(mono.astype(np.float32), up, down)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def pick_player() -> str:
    for candidate in ("ffplay", "afplay"):
        if subprocess.run(["which", candidate], capture_output=True).returncode == 0:
            return candidate
    return ""


def main() -> None:
    device = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DEVICE
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_RECORD_SECONDS
    native_rate = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_NATIVE_RATE

    print("=== 사용 가능한 마이크 장치 (sounddevice 기준) ===")
    print(sd.query_devices())
    print()

    print(f"device={device}, native_rate={native_rate}Hz, {duration:.1f}초 녹음")
    print(">>> 지금 마이크에 대고 말해보세요...")
    raw = sd.rec(
        int(duration * native_rate),
        samplerate=native_rate,
        channels=1,
        dtype="int16",
        device=device,
    )
    sd.wait()
    print("녹음 종료.\n")

    raw_mono = raw[:, 0] if raw.ndim > 1 else raw
    resampled = resample_to_target(raw, native_rate, TARGET_RATE)

    raw_level = float(np.abs(raw_mono).mean())
    resampled_level = float(np.abs(resampled).mean())
    print(f"[원본 {native_rate}Hz] 평균 레벨: {raw_level:.1f}  최대: {int(np.abs(raw_mono).max())}")
    print(f"[리샘플 {TARGET_RATE}Hz] 평균 레벨: {resampled_level:.1f}  최대: {int(np.abs(resampled).max())}")
    print(f"(무음 판정 기준: {SILENCE_THRESHOLD})")
    if resampled_level < SILENCE_THRESHOLD:
        print("판정: 무음으로 인식됨 - 장치 index/샘플레이트가 실제 마이크와 맞는지 확인하세요.")
    else:
        print("판정: 음성 감지됨 (정상)")
    print()

    # --- 파형 그래프 저장 (한글 폰트 미설치 환경 대비 라벨은 영문으로) ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    t_raw = np.arange(len(raw_mono)) / native_rate
    axes[0].plot(t_raw, raw_mono, linewidth=0.5)
    axes[0].set_title(f"Raw recording ({native_rate}Hz, mean_level={raw_level:.1f})")
    axes[0].set_xlabel("time (sec)")
    axes[0].set_ylabel("amplitude")
    axes[0].axhline(0, color="gray", linewidth=0.3)

    t_resampled = np.arange(len(resampled)) / TARGET_RATE
    axes[1].plot(t_resampled, resampled, linewidth=0.5, color="orange")
    axes[1].set_title(f"Resampled to {TARGET_RATE}Hz (mean_level={resampled_level:.1f}) - sent to Whisper")
    axes[1].set_xlabel("time (sec)")
    axes[1].set_ylabel("amplitude")
    axes[1].axhline(0, color="gray", linewidth=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"파형 그래프 저장됨: {PLOT_PATH}")
    print("그래프가 평평한 직선(0 근처)이면 마이크가 실제로 소리를 못 잡고 있는 것입니다.")
    print()

    sf.write(WAV_PATH, resampled, TARGET_RATE)

    player = pick_player()
    if not player:
        print(f"재생 프로그램(ffplay/afplay)을 찾지 못했습니다. 녹음 파일: {WAV_PATH}")
        return

    print(f"방금 녹음한 내용을 재생합니다 ({player})...")
    if player == "ffplay":
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(WAV_PATH)])
    else:
        subprocess.run([player, str(WAV_PATH)])


if __name__ == "__main__":
    main()
