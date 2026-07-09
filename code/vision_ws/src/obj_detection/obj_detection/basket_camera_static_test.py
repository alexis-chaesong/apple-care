"""
basket_camera_static_test.py
=============================
세컨 카메라(basket_camera.py)의 판정 로직(바스켓 b1~b4 배정 + 점유 판단)만
정지 이미지 한 장으로 빠르게 확인하기 위한 스크립트. ROS2를 전혀 띄우지 않고
(rclpy.init 없음, 토픽/서비스 없음) YOLO 모델과 basket_camera.py의 순수 함수
(assign_baskets/compute_occupancy/draw_basket_overlay)만 그대로 재사용한다.

실제 카메라를 아직 설치/연결하기 전이라도, 후보 설치 위치에서 찍은 사진
한 장으로 "이 화각에서 바스켓 4개가 다 잡히는지", "좌->우 b1~b4 배정이
의도한 대로 되는지", "사과 유무 판단이 맞는지"를 먼저 확인할 수 있다.

사용법 (vision_ws 빌드/소스 후):
    ros2 run obj_detection basket_camera_static_test --image /path/to/basket.jpg
    ros2 run obj_detection basket_camera_static_test --image basket.jpg --output annotated.jpg

--output을 안 주면 입력 파일명 옆에 "<원본이름>_annotated.jpg"로 저장한다.
"""

import argparse
import json
import os
import sys

import cv2

from obj_detection.basket_camera import (
    assign_baskets,
    compute_occupancy,
    draw_basket_overlay,
)
from obj_detection.yolo import AppleStatusModel


def _default_output_path(image_path: str) -> str:
    root, ext = os.path.splitext(image_path)
    return f"{root}_annotated{ext or '.jpg'}"


def run(image_path: str, output_path: str) -> dict:
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    model = AppleStatusModel()
    detections = model.get_all_detections(frame)

    basket_assignment, valid = assign_baskets(detections)
    occupancy = compute_occupancy(detections, basket_assignment)

    annotated = draw_basket_overlay(frame, basket_assignment, occupancy)
    cv2.imwrite(output_path, annotated)

    result = {name: occupancy[name] for name in ("b1", "b2", "b3", "b4")}
    result["valid"] = valid
    result["basket_count_detected"] = sum(
        1 for b in basket_assignment.values() if b is not None
    )
    result["apple_boxes_detected"] = sum(
        1 for label, _score, _box in detections
        if label in ("apple_normal", "apple_rotten", "apple_damaged")
    )
    return result


def main(args=None):
    parser = argparse.ArgumentParser(
        description="세컨 카메라 바스켓 배정/점유 판단 로직을 정지 이미지로 테스트"
    )
    parser.add_argument("--image", required=True, help="바스켓 4개가 보이는 테스트 이미지 경로")
    parser.add_argument(
        "--output", default=None,
        help="주석(bbox/라벨)이 그려진 결과 이미지를 저장할 경로 (기본: <입력파일명>_annotated.jpg)",
    )
    parsed = parser.parse_args(args=args if args is not None else sys.argv[1:])

    output_path = parsed.output or _default_output_path(parsed.image)
    result = run(parsed.image, output_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        print(
            f"[경고] 바스켓 4개 중 {result['basket_count_detected']}개만 검출됨 - "
            "화각/거리를 조정하거나 조명을 확인하세요.",
            file=sys.stderr,
        )
    print(f"주석 이미지 저장: {output_path}")


if __name__ == "__main__":
    main()
