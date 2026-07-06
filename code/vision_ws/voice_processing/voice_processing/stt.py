
from openai import OpenAI
import sounddevice as sd
import numpy as np
from scipy.signal import resample_poly
import scipy.io.wavfile as wav
import tempfile

class STT:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        # self.openai_api_key = openai_api_key
        self.duration = 5  # seconds
        self.input_samplerate = 48000  # 마이크 장치 4의 실제 샘플레이트
        self.samplerate = 16000  # Whisper는 16kHz를 선호


    def speech2text(self):
        # 녹음 설정
        print("음성 녹음을 시작합니다. \n 5초 동안 말해주세요...")
        audio = sd.rec(
            int(self.duration * self.input_samplerate),
            samplerate=self.input_samplerate,
            channels=1,
            dtype='int16',
            device=4,
        )
        sd.wait()
        audio = resample_poly(audio[:, 0].astype(np.float32), 1, 3)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        print("녹음 완료. Whisper에 전송 중...")

        # 임시 WAV 파일 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)

            # Whisper API 호출
            with open(temp_wav.name, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", file=f, language="ko")

        #print("STT 결과: ", transcript['text'])
        print("STT 결과: ", transcript.text)
        #return transcript['text']
        return transcript.text