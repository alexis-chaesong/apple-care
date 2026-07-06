import os
import numpy as np
try:
    import tflite_runtime.interpreter  # Python 3.11
except ModuleNotFoundError:
    import sys
    import types
    import ai_edge_litert.interpreter as litert_interpreter

    tflite_runtime = types.ModuleType("tflite_runtime")
    tflite_runtime.interpreter = litert_interpreter
    sys.modules["tflite_runtime"] = tflite_runtime
    sys.modules["tflite_runtime.interpreter"] = litert_interpreter

from openwakeword.model import Model
from scipy.signal import resample_poly
from ament_index_python.packages import get_package_share_directory

PACKAGE_NAME = "voice_processing"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

MODEL_NAME = "hello_rokey_8332_32.tflite"
MODEL_PATH = os.path.join(PACKAGE_PATH, f"resource/{MODEL_NAME}")

class WakeupWord:
    def __init__(self, buffer_size):
        self.model = None
        self.model_name = MODEL_NAME.split(".", maxsplit=1)[0]
        self.stream = None
        self.buffer_size = buffer_size

    def is_wakeup(self):
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample_poly(audio_chunk.astype(np.float32), 1, 3)
        audio_chunk = np.clip(audio_chunk, -32768, 32767).astype(np.int16)
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        print("confidence: ", confidence)
        # Wakeword 탐지
        if confidence > 0.3:
            print("Wakeword detected!")
            return True
        return False

    def set_stream(self, stream):
        self.model = Model(wakeword_models=[MODEL_PATH])
        self.stream = stream
