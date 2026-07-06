import io
import os

import sounddevice as sd
from openai import OpenAI
from scipy.io import wavfile


class TTS:
    """OpenAI TTS로 텍스트를 음성으로 변환해서 바로 재생한다."""

    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)

    def speak(self, text, voice="alloy"):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="wav",
        )
        samplerate, audio = wavfile.read(io.BytesIO(response.read()))
        sd.play(audio, samplerate)
        sd.wait()
