# 마이크 장치 인덱스 확인용 스크립트
# 실행: python3 scripts/list_audio_devices.py
#
# 여기서 확인한 인덱스를 .env의 WAKEWORD_MIC_DEVICE_INDEX에 넣으면 됨
# (stt_tts/wakeup_listener.py가 pyaudio로 마이크를 열 때 이 값을 씀)

import pyaudio


def main() -> None:
    audio = pyaudio.PyAudio()
    try:
        default_index = None
        try:
            default_index = audio.get_default_input_device_info()["index"]
        except IOError:
            pass

        print(f"{'idx':>4}  {'in_ch':>5}  {'rate':>7}  name")
        print("-" * 60)

        found_input = False
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) <= 0:
                continue
            found_input = True
            marker = " (default)" if i == default_index else ""
            print(
                f"{i:>4}  {int(info['maxInputChannels']):>5}  "
                f"{int(info['defaultSampleRate']):>7}  {info['name']}{marker}"
            )

        if not found_input:
            print("입력 가능한 마이크 장치를 찾지 못했습니다.")
    finally:
        audio.terminate()


if __name__ == "__main__":
    main()
