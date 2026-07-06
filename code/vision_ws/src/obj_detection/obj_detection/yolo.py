########## AppleStatusModel ##########
import os
import time

from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


PACKAGE_NAME = "obj_detection"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

YOLO_MODEL_FILENAME = "best_apple_care.pt"
YOLO_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", YOLO_MODEL_FILENAME)

CLASS_NAMES = ["apple_normal", "apple_rotten", "apple_damaged"]

# 이 값보다 낮은 confidence로 감지된 박스는 "unknown"으로 취급한다
# (apple_normal/rotten/damaged로 학습된 3클래스 중 어디에도 확신 없다는 뜻)
MIN_KNOWN_CONFIDENCE = 0.4

# 프레임 수집 단계에서 후보로 남길 최소 confidence (이후 최종 판정은 MIN_KNOWN_CONFIDENCE로 함)
MIN_CANDIDATE_CONFIDENCE = 0.1


class AppleStatusModel:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)

    def get_frames(self, img_node, img_executor, duration=1.0):
        """duration(초) 동안 컬러 프레임을 모아서 리스트로 반환."""
        end_time = time.time() + duration
        frames = {}

        while time.time() < end_time:
            img_executor.spin_once(timeout_sec=0.1)
            frame = img_node.get_color_frame()
            stamp = img_node.get_color_frame_stamp()
            if frame is not None:
                frames[stamp] = frame
            time.sleep(0.01)

        return list(frames.values())

    def get_best_detection(self, img_node, img_executor):
        """카메라에 보이는 것 중 confidence가 가장 높은 박스 하나를 반환.

        반환값: (class_name, confidence, box[x1,y1,x2,y2]) 또는 감지된 게 전혀 없으면
        (None, None, None).
        """
        frames = self.get_frames(img_node, img_executor)
        if not frames:
            return None, None, None

        results = self.model(frames, verbose=False)

        best = None
        for res in results:
            for box, score, label in zip(
                res.boxes.xyxy.tolist(),
                res.boxes.conf.tolist(),
                res.boxes.cls.tolist(),
            ):
                if score < MIN_CANDIDATE_CONFIDENCE:
                    continue
                if best is None or score > best[1]:
                    best = (CLASS_NAMES[int(label)], score, box)

        if best is None:
            return None, None, None
        return best

    def annotate_frame(self, frame):
        """단일 프레임에 YOLO 박스/라벨을 그려서 반환 (디버그 시각화용)."""
        results = self.model(frame, verbose=False)
        return results[0].plot()
