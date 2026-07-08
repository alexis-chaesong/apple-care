########## AppleStatusModel ##########
import os
import time

from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


PACKAGE_NAME = "obj_detection"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

YOLO_MODEL_FILENAME = "best_apple_care.pt"
YOLO_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", YOLO_MODEL_FILENAME)

CLASS_NAMES = ["apple_normal", "apple_rotten", "apple_damaged", "apple_small"]

# 이 값보다 낮은 confidence로 감지된 박스는 "unknown"으로 취급한다
# (apple_normal/rotten/damaged/small로 학습된 4클래스 중 어디에도 확신 없다는 뜻)
MIN_KNOWN_CONFIDENCE = 0.4

# 프레임 수집 단계에서 후보로 남길 최소 confidence (이후 최종 판정은 MIN_KNOWN_CONFIDENCE로 함)
MIN_CANDIDATE_CONFIDENCE = 0.1


class AppleStatusModel:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)

    def get_frames(self, img_node, duration=1.0):
        """duration(초) 동안 컬러 프레임을 모아서 리스트로 반환.

        img_node는 이제 자체 백그라운드 스레드에서 계속 spin되고 있으므로,
        여기서는 그냥 최신 프레임을 주기적으로 읽어오기만 하면 된다."""
        end_time = time.time() + duration
        frames = {}

        while time.time() < end_time:
            frame = img_node.get_color_frame()
            stamp = img_node.get_color_frame_stamp()
            if frame is not None:
                frames[stamp] = frame
            time.sleep(0.01)

        return list(frames.values())

    def get_best_detection(self, img_node):
        """카메라에 보이는 것 중 바운딩 박스 면적이 가장 큰 박스 하나를 반환.

        여러 사과가 동시에 보일 때, 처리 순서를 confidence가 아니라 "화면에
        크게 보이는(=카메라에 더 가깝거나 실제로 더 큰) 사과부터"로 정하기 위함.
        get_apple_status는 매 호출마다 이 함수가 고른 사과 하나만 돌려주고,
        그 사과가 실제로 집어져서 트레이에서 사라지면 다음 폴링에서 자연히
        그 다음으로 큰 사과가 뽑히는 방식으로 동작함.

        다만 면적만으로 고르면, 벽/배경처럼 confidence는 낮지만(MIN_CANDIDATE_
        CONFIDENCE만 겨우 넘는) 박스 자체는 큰 오탐이 실제 사과(면적은 작아도
        confidence는 확실히 높은)를 계속 이겨버리는 문제가 있었음. 그래서
        confidence가 MIN_KNOWN_CONFIDENCE(=detection.py가 "확실히 안다"고 보는
        기준) 이상인 후보가 하나라도 있으면 그 안에서만 면적 최대를 고르고,
        전부 그 기준 미만일 때만(진짜 아무 확실한 것도 없을 때만) 낮은 confidence
        후보들 중 면적 최대로 폴백함 (그래야 detection.py의 depth 기반 "unknown"
        폴백 경로도 계속 동작함).

        반환값: (class_name, confidence, box[x1,y1,x2,y2]) 또는 감지된 게 전혀 없으면
        (None, None, None).
        """
        frames = self.get_frames(img_node)
        if not frames:
            return None, None, None

        results = self.model(frames, verbose=False)

        def _box_area(box):
            x1, y1, x2, y2 = box
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        candidates = []
        for res in results:
            for box, score, label in zip(
                res.boxes.xyxy.tolist(),
                res.boxes.conf.tolist(),
                res.boxes.cls.tolist(),
            ):
                if score < MIN_CANDIDATE_CONFIDENCE:
                    continue
                candidates.append((CLASS_NAMES[int(label)], score, box))

        if not candidates:
            return None, None, None

        confident_candidates = [c for c in candidates if c[1] >= MIN_KNOWN_CONFIDENCE]
        pool = confident_candidates if confident_candidates else candidates

        return max(pool, key=lambda c: _box_area(c[2]))

    def annotate_frame(self, frame):
        """단일 프레임에 YOLO 박스/라벨을 그려서 반환 (디버그 시각화용)."""
        results = self.model(frame, verbose=False)
        return results[0].plot()

    def get_all_detections(self, frame):
        """단일 프레임에서 YOLO가 찾은 모든 후보 박스를 반환 (디버그 시각화용).

        get_best_detection처럼 1초 동안 프레임을 모아 최고 confidence 하나만
        고르는 게 아니라, 화면에 보이는 모든 사과를 depth 디버그 창에 동시에
        표시하기 위해 단일 프레임에 대해 즉시 추론한다.

        반환값: [(class_name, confidence, box[x1,y1,x2,y2]), ...]
        """
        results = self.model(frame, verbose=False)
        detections = []
        for res in results:
            for box, score, label in zip(
                res.boxes.xyxy.tolist(),
                res.boxes.conf.tolist(),
                res.boxes.cls.tolist(),
            ):
                if score < MIN_CANDIDATE_CONFIDENCE:
                    continue
                detections.append((CLASS_NAMES[int(label)], score, box))
        return detections
