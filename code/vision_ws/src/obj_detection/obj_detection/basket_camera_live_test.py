"""
basket_camera_live_test.py
===========================
바스켓 조망용 웹캠을 켜놓은 상태에서, 바스켓 b1~b4 배정과 사과 점유 판단이
실시간으로 어떻게 나오는지 로컬 창으로 바로 확인하기 위한 스크립트.
basket_camera.py(basket_detection_node)처럼 백엔드용 `/basket_status` 토픽을
계속 publish하는 정식 운영 노드가 아니라, 카메라 화각/위치를 잡는 동안(사용자가
실제로 카메라를 이리저리 옮겨보며 확인하는 단계) 쓰는 눈으로 보는 디버그
도구다. detection.py의 `_depth_debug_loop`와 같은 목적(로컬 cv2 창 실시간
확인)을 바스켓 카메라 쪽에 맞춘 버전이며, ROS는 전혀 필요 없다(rclpy.init 없음).

basket_camera.py와 동일하게 cv2.VideoCapture로 /dev/videoN을 직접 연다.
--device_index로 /dev/videoN의 N을 지정한다 (`v4l2-ctl --list-devices`로 장치와
인덱스 확인 가능, 예: 로지텍 C270이 /dev/video39면 39).

    ros2 run obj_detection basket_camera_live_test --device_index 39

창에서 'q'를 누르면 종료한다. 매 프레임 YOLO 추론 + 바스켓 배정/점유 판단
결과를 basket_camera.py와 동일한 순수 함수(assign_baskets/compute_occupancy/
draw_basket_overlay)로 그대로 그려서, 실제 운영 노드와 판정 로직이 100% 같다.
"""

import argparse
import sys

import cv2

from obj_detection.basket_camera import (
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    assign_baskets,
    compute_occupancy,
    draw_basket_overlay,
)
from obj_detection.yolo import AppleStatusModel

WINDOW_NAME = "Basket Camera Live Test"


def main(args=None):
    parser = argparse.ArgumentParser(
        description="바스켓 카메라(웹캠) 배정/점유 판단을 실시간 창으로 확인"
    )
    parser.add_argument(
        "--device_index", type=int, default=39,
        help="바스켓 조망용 웹캠의 /dev/videoN 인덱스 "
             "(basket_camera.py의 device_index 파라미터와 동일한 값을 쓸 것).",
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_FRAME_WIDTH,
        help="캡처 해상도 너비. 카메라가 지원하지 않는 값이면 검은 화면만 나오므로 "
             "'v4l2-ctl --list-formats-ext'로 지원 해상도를 먼저 확인할 것.",
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_FRAME_HEIGHT,
        help="캡처 해상도 높이 (--width 설명 참고).",
    )
    parsed, _ros_args = parser.parse_known_args(
        args=args if args is not None else sys.argv[1:]
    )

    cap = cv2.VideoCapture(parsed.device_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"/dev/video{parsed.device_index}를 열 수 없습니다. "
            "'v4l2-ctl --list-devices'로 장치/인덱스를 다시 확인하세요."
        )
    # 카메라가 실제로 지원 안 하는 해상도로 열리면 ok=True인데 프레임이 전부
    # 검은색(픽셀 0)으로만 나오는 문제가 있어(basket_camera.py의 WebcamCapture와
    # 동일한 이유), 반드시 지원 해상도로 명시 설정한다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, parsed.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, parsed.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = AppleStatusModel()
    print(
        f"/dev/video{parsed.device_index} 오픈 완료 "
        f"(요청 {parsed.width}x{parsed.height}, 실제 {actual_w}x{actual_h}). 'q'로 종료.",
        flush=True,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    last_occupancy = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
                continue

            detections = model.get_all_detections(frame)
            basket_assignment, valid, basket_labels = assign_baskets(detections)
            occupancy = compute_occupancy(detections, basket_assignment, basket_labels)

            if occupancy != last_occupancy:
                detected = sum(1 for b in basket_assignment.values() if b is not None)
                print(f"바스켓 {detected}/4개 검출(valid={valid}) - 점유 상태: {occupancy}")
                last_occupancy = occupancy

            annotated = draw_basket_overlay(frame, basket_assignment, occupancy)
            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
