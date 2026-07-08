"""
debug_overlay.py
=================
ObjectDetectionNode에 섞어 쓰는(mixin) 디버그 시각화 로직.

- 로컬 뎁스 컬러맵 창 (_depth_debug_loop)
- 프론트(hmi_app)로 나가는 디버그 이미지 토픽 (_publish_debug_image)
- YOLO 박스/미확인 블롭/최근 서비스 판정 결과를 화면에 그리는 함수들

detection.py의 `class ObjectDetectionNode(..., DebugOverlayMixin, ...)`로
섞여서 쓰인다. self.img_node, self.model, self.debug_image_pub,
self.background_distance_mm, self._stop_depth_debug, self._last_debug_info,
self._debug_info_lock은 ObjectDetectionNode.__init__에서 준비되고,
self._box_ring_depths/_real_world_diameter_mm/_find_all_unknown_blobs(depth_utils),
self._preview_size_label(size_classifier)는 다른 mixin이 제공한다는 전제로 작성됨.
"""

import time

import cv2

from obj_detection.depth_utils import HEIGHT_DIFF_MARGIN_MM
from obj_detection.ros_image_utils import cv2_to_imgmsg

# 디버그용 인식 결과 시각화 이미지 퍼블리시 주기(초)
DEBUG_IMAGE_PERIOD_SEC = 0.05

# 뎁스 디버깅 창 이름
DEPTH_DEBUG_WINDOW_NAME = "Depth Debug"

# 디버그 오버레이 글자 크기/굵기 (이전 단일-박스 오버레이 스타일로 복원 - 다중 박스로
# 바꾸면서 한 번 줄였던 걸 다시 키움)
OVERLAY_FONT_SCALE = 0.6
OVERLAY_FONT_THICKNESS = 2


class DebugOverlayMixin:
    """뎁스 디버그 창 + 프론트용 디버그 이미지 토픽을 그리는 mixin."""

    def _publish_debug_image(self):
        """최신 컬러 프레임에 depth 기반 판정 근거(높이차 통과 여부/실제 지름/
        크기 재분류)를 그려서 디버그 토픽으로 퍼블리시.

        예전엔 model.annotate_frame()으로 YOLO 원본 박스/라벨만 그려서 내보냈는데,
        그러면 프론트(hmi_app)의 카메라 화면에는 depth 판정이나 apple_small_* 재분류가
        전혀 안 보였다. 뎁스 디버그 창에서 쓰던 것과 동일한 `_draw_all_detections`를
        컬러 프레임 위에 그대로 적용해서, 로컬 뎁스 창과 프론트 화면이 같은 정보를
        보여주게 한다 (컬러/뎁스 프레임은 aligned_depth_to_color라 픽셀 좌표가 일치).
        """
        frame = self.img_node.get_color_frame()
        if frame is None:
            return
        depth_frame = self.img_node.get_depth_frame()
        annotated = frame.copy()
        detections = self.model.get_all_detections(frame)
        self._draw_all_detections(annotated, depth_frame, detections)
        msg = cv2_to_imgmsg(annotated, encoding='bgr8')
        self.debug_image_pub.publish(msg)


    def _depth_debug_loop(self):
        """뎁스 프레임에 컬러맵을 입히고, 화면에 보이는 모든 사과에 대해 실시간으로
        박스/뎁스값/크기 판정을 오버레이로 그려서 로컬 창으로 띄운다.

        handle_get_status(서비스)는 1초에 한 번, "가장 confidence 높은 박스 하나"만
        판정하므로 트레이에 사과가 여러 개면 나머지는 창에 안 보였다. 여기서는
        디버그용으로 매 주기 YOLO를 다시 돌려 보이는 모든 박스를 그린다.

        executor 콜백 스레드와 분리된 전용 스레드에서 계속 돌며 이 창의
        생성/갱신을 전담한다 (GUI 백엔드가 스레드 하나에 고정되어야 하므로).
        """
        cv2.namedWindow(DEPTH_DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
        while not self._stop_depth_debug.is_set():
            depth_frame = self.img_node.get_depth_frame()
            color_frame = self.img_node.get_color_frame()
            if depth_frame is not None:
                # mm 단위 depth를 8bit로 스케일링 후 컬러맵 적용 (가까울수록 밝게 보이도록)
                depth_8bit = cv2.convertScaleAbs(
                    depth_frame, alpha=255.0 / self.background_distance_mm
                )
                depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

                detections = []
                if color_frame is not None:
                    detections = self.model.get_all_detections(color_frame)
                self._draw_all_detections(depth_colormap, depth_frame, detections)

                with self._debug_info_lock:
                    debug_info = self._last_debug_info
                self._draw_last_service_summary(depth_colormap, debug_info)

                cv2.imshow(DEPTH_DEBUG_WINDOW_NAME, depth_colormap)
            # waitKey가 GUI 이벤트 루프를 돌려줘야 창이 실제로 그려진다.
            cv2.waitKey(1)
            time.sleep(DEBUG_IMAGE_PERIOD_SEC)
        cv2.destroyWindow(DEPTH_DEBUG_WINDOW_NAME)


    def _draw_all_detections(self, colormap, depth_frame, detections):
        """현재 프레임에서 YOLO가 찾은 모든 박스에 대해 depth 기반 판정 근거를
        각 박스 위에 그린다 (높이차 통과 여부, 실제 지름, 크기 재분류 미리보기).

        YOLO가 apple_small을 직접 학습했지만, apple_normal/apple_damaged로 나온
        것들 중 depth 실측 지름이 ABSOLUTE_SMALL_DIAMETER_MM 이하로 확연히 작으면
        여기서도 "apple_small"로 재분류될지 미리 보여준다 (실제 판정은
        handle_get_status의 `_classify_size`가 하고, 여기는 같은 기준의 미리보기).

        YOLO가 아예 후보조차 못 낸 물체도 depth만으로 찾아서(`_find_all_unknown_blobs`)
        자홍색 박스로 같이 표시한다 - 안 그러면 뎁스 상으로는 분명히 뭔가 튀어나와
        있는데 화면에는 아무 박스도 안 그려져서 "왜 이 물체는 안 잡히지?"처럼 보임.
        """
        h, w = colormap.shape[:2]
        yolo_boxes = []  # depth_ok 여부와 무관하게, YOLO가 낸 모든 박스 (자홍 블롭 억제용)

        for label, score, box in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            yolo_boxes.append((x1, y1, x2, y2))

            inner_depth, ring_depth = self._box_ring_depths(depth_frame, box)
            depth_ok = (
                inner_depth is not None and ring_depth is not None
                and (ring_depth - inner_depth) >= HEIGHT_DIFF_MARGIN_MM
            )
            box_color = (0, 255, 0) if depth_ok else (0, 0, 255)  # OK=초록, FAIL=빨강
            cv2.rectangle(colormap, (x1, y1), (x2, y2), box_color, 2)

            if not depth_ok:
                caption = f"{label} {score:.2f} depth-FAIL"
            else:
                diameter_mm = self._real_world_diameter_mm(depth_frame, box)
                preview_label = self._preview_size_label(label, diameter_mm)
                diameter_txt = f"{diameter_mm:.0f}mm" if diameter_mm is not None else "-"
                if preview_label != label:
                    caption = f"{preview_label} {score:.2f} {diameter_txt}"
                else:
                    caption = f"{label} {score:.2f} {diameter_txt}"

            text_y = y1 - 8 if y1 - 8 > 10 else y2 + 18
            cv2.putText(
                colormap, caption, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (0, 0, 0), OVERLAY_FONT_THICKNESS + 2,
            )
            cv2.putText(
                colormap, caption, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (255, 255, 255), OVERLAY_FONT_THICKNESS,
            )

        if not detections:
            cv2.putText(
                colormap, "no YOLO box in current frame", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (255, 255, 255), OVERLAY_FONT_THICKNESS,
            )

        self._draw_unknown_blob(colormap, depth_frame, yolo_boxes)


    def _draw_unknown_blob(self, colormap, depth_frame, yolo_boxes):
        """YOLO가 후보조차 못 낸 곳인데 depth로는 튀어나와 보이는 덩어리를 전부
        자홍색 박스로 표시한다 (여러 개 있으면 다 그림). YOLO가 이미 찾은 박스
        영역은 탐색 전에 미리 제외하므로(`_find_all_unknown_blobs`의
        exclude_boxes), 여기서 다시 겹침을 확인할 필요가 없다."""
        blobs = self._find_all_unknown_blobs(depth_frame, exclude_boxes=yolo_boxes)
        for blob in blobs:
            x1, y1, x2, y2 = blob['bbox']
            cv2.rectangle(colormap, (x1, y1), (x2, y2), (255, 0, 255), 2)  # 자홍색 = depth만으로 감지
            caption = f"unknown(depth-only) {blob['depth_mm']:.0f}mm"

            text_y = y1 - 8 if y1 - 8 > 10 else y2 + 18
            cv2.putText(
                colormap, caption, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (0, 0, 0), OVERLAY_FONT_THICKNESS + 2,
            )
            cv2.putText(
                colormap, caption, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (255, 0, 255), OVERLAY_FONT_THICKNESS,
            )

    def _draw_last_service_summary(self, colormap, debug_info):
        """실제로 backend에 보고되는 값(1초 주기 서비스 호출의 최종 판정)을
        화면 상단에 한 줄 요약으로 보여준다. 위의 실시간 박스 미리보기는 참고용이고,
        실제 로봇 동작에 쓰이는 건 이 값이라는 걸 구분하기 위함."""
        if debug_info is None:
            text = "last service call: (none yet)"
        else:
            text = (
                f"last service call: {debug_info['raw_label']} -> "
                f"{debug_info['final_status']}"
            )
        h = colormap.shape[0]
        y = h - 12
        cv2.putText(
            colormap, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (0, 0, 0), OVERLAY_FONT_THICKNESS + 2,
        )
        cv2.putText(
            colormap, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (255, 255, 255), OVERLAY_FONT_THICKNESS,
        )
