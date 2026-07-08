# 비전-액션(Vision-Action) 연동 작업 정리

로봇이 카메라(vision)가 인식한 사과 위치/상태를 받아서 실제로 집고 분류해서
박스에 놓기까지, `box_sequence_test.py`를 중심으로 연동한 작업 전체 기록.

## 1. 핵심: 카메라 좌표 → 로봇 베이스 좌표 변환

### 문제
`obj_detection`(비전)이 계산하는 3D 위치는 항상 **카메라 좌표계** 기준
(`obj_detection/depth_utils.py`의 `_pixel_to_camera_coords`). 그런데 이 값이
`vision_bridge.py` → `decision_planner.py` → `/decision/result`를 거쳐
`apple_sorting_cycle.py`까지 변환 없이 그대로 전달되고 있었고, 그걸 로봇 베이스
좌표인 것처럼 바로 `posx()`에 넣고 있었다. 참고 프로젝트
(`cobot2_ws/src/pick_and_place_voice/robot_control/robot_control.py`)의
`transform_to_base()`와 동일한 변환 단계가 빠져 있었던 것.

### 해결: `vision_transform.py` (신규)
- 경로: `code/robot_ws/src/apple_care_robot/apple_care_robot/vision_transform.py`
- `camera_to_base(camera_xyz, robot_posx, depth_offset_mm=None)`:
  eye-in-hand 캘리브레이션(`T_gripper2camera.npy`, 그리퍼-카메라 고정 변환)과
  "카메라가 그 물체를 봤을 때의 로봇 pose"(`get_current_posx()`)를 이용해
  `base2cam = base2gripper @ gripper2cam` 행렬을 만들고, 카메라 좌표(동차좌표)에
  곱해서 로봇 베이스 좌표를 얻는다.
- `T_gripper2camera.npy`를 `vision_ws/src/obj_detection/resource/`에서
  `robot_ws/src/apple_care_robot/resource/`로 복사 (로봇 패키지가 자체적으로
  갖고 있어야 함) → `setup.py`의 `data_files`에 등록.
- 튜닝 상수 (실측 기반, 계속 조정 중):
  - `DEPTH_OFFSET_MM = -80.0` — 카메라 depth 원점과 실제로 잡아야 할 지점(사과
    표면) 사이 z 보정값.
  - `SMALL_APPLE_DEPTH_OFFSET_MM = DEPTH_OFFSET_MM + 20.0` — 작은 사과는
    몸통이 낮아서 일반 오프셋만큼 내려가면 트레이/받침대와 충돌하므로, 덜
    내려가는 전용 오프셋을 별도로 둠. `box_sequence_test.py`가 감지된
    `status`에 따라 둘 중 하나를 골라서 넘김.
  - `MIN_Z_MM = 2.0` — 변환 후 z가 이보다 낮으면 클램프 (테이블 관통 방지).

### 적용된 곳
- `apple_sorting_cycle.py`: `/decision/result`로 받은 `pose`를
  `camera_to_base()`로 변환 후 `pick_pos` 생성.
- `box_sequence_test.py`: `get_apple_status` 서비스 응답을 직접
  `camera_to_base()`로 변환 (아래 2번 항목 참고).

## 2. `box_sequence_test.py` — 비전 연동 + 재시도/우선순위 로직

원래는 박스 4개(`b1~b4`)를 고정 순서로 방문하며 하드코딩된 좌표로 사과를 집는
하드웨어 시퀀스 테스트 스크립트였음. 다음 순서로 실제 비전 연동 파이프라인으로
바꿈.

1. **비전 좌표 연동**: `APPLE_PICK_POS` 하드코딩 제거 →
   `/get_apple_status` 서비스를 CAMERA 위치에서 직접 호출해서 실측 위치 사용.
   - 서비스 이름은 반드시 절대 경로(`"/get_apple_status"`)여야 함 — 이 노드가
     `namespace="dsr01"`로 생성되어 있어서, 상대 이름을 쓰면 rclpy가
     `/dsr01/get_apple_status`로 잘못 리졸브해서 서비스를 영영 못 찾는 버그가
     있었음.
2. **status → 박스 매핑**: 더 이상 `b1→b2→b3→b4` 고정 순서가 아니라, 감지된
   `status`로 목적지를 정함 (`STATUS_TO_BOX_NAME`):
   `apple_small→b1`, `apple_damaged→b2`, `apple_normal→b3`,
   `apple_rotten(폐기)→b4`. `unknown`/`empty`나 매핑에 없는 값은 "확실하지
   않음"으로 보고 집지 않고 재시도(`wait_for_confident_detection`,
   `MAX_DETECTION_ATTEMPTS=5`).
3. **depth 측정 실패 sentinel 버그 수정**: `detection.py`는 depth 읽기 실패
   시 `position=[0.0, 0.0, 0.0]`을 반환하는데(파이썬에서 `[0,0,0]`은
   `not position`으로 안 걸러짐), 이걸 진짜 카메라 좌표로 착각해서
   변환하면 "카메라 렌즈 자체의 위치"로 계산되어 엉뚱한 곳으로 이동하는
   버그가 실제로 재현됨 → `all(v == 0.0 for v in position)` 체크 추가.
   (같은 버그가 `backend/vision_bridge.py`의 `_ros_response_to_vision_feature`
   에도 있어서 같이 고침 — `_is_zero_position()` 헬퍼 추가.)
4. **그립 실패 재시도**: `pick_apple()`의 반환값(force feedback 기반 그립
   성공 여부)을 지금까지 버리고 있어서, 그립에 실패해도 빈 그리퍼로 그대로
   박스까지 이동하던 문제가 있었음 → `try_pick_apple()` 추가: 그립 실패 시
   그리퍼를 다시 열고 CAMERA로 복귀해서 위치를 재감지, 최대
   `MAX_PICK_ATTEMPTS=3`번 재시도.
5. **불필요한 HOME 왕복 제거**: 기존엔 매 사이클마다
   `... → CAMERA → HOME → (다음 사이클) HOME → CAMERA → ...`처럼 중복
   왕복이 있었음. HOME은 전체 시퀀스 맨 처음 1회, 맨 마지막 1회만 거치고,
   사이클 사이에는 CAMERA에 머무르며 바로 다음 감지로 이어지도록 변경
   (박스로 가기 전의 안전 경유 HOME은 충돌 방지 목적이라 유지).
6. **시작 시 그리퍼 오픈**: 이전 실행이 그립한 채로 끝났거나 하드웨어
   전원이 새로 들어온 경우를 대비해, 시퀀스 시작 직후 항상 `gripper_open()`
   호출.

## 3. `obj_detection` (비전) 쪽 개선

### `yolo.py` — 감지 우선순위: confidence 우선 후 면적 최대
- 요청: "여러 개의 bbox 중 큰 것부터 처리".
- 1차 구현: 단순히 면적이 가장 큰 박스를 선택 → 벽/배경처럼 confidence는
  낮지만 박스 자체는 큰 오탐이 실제 사과를 계속 이겨버리는 부작용 발생
  ("벽을 unknown으로 인식해서 small이 묻힘").
- 최종: `MIN_KNOWN_CONFIDENCE`(0.4) 이상인 후보가 있으면 그 안에서만 면적
  최대를 고르고, 전부 그 미만일 때만(정말 애매한 것 하나만 있을 때) 낮은
  confidence 후보 중 면적 최대로 폴백.

### `detection.py` — YOLO confidence가 높으면 depth 검증 생략
- `_box_ring_depths`의 높이차 검증(박스 안쪽 depth가 바로 바깥 배경보다
  `HEIGHT_DIFF_MARGIN_MM` 이상 튀어나와야 통과, 사진/그림자 오탐 방지용
  안전장치)이 작은 사과(배경 대비 튀어난 높이 자체가 작음)에서 자주
  실패해서 `unknown`으로 새는 문제가 있었음.
- `YOLO_TRUST_CONFIDENCE = 0.7` 추가: YOLO confidence가 이 값 이상이면 depth
  높이차 검증을 건너뛰고 YOLO 결과를 그대로 신뢰. 그 미만은 기존처럼 depth
  검증이 안전장치로 계속 동작 (기존 로직은 그대로 유지, 우선순위만 추가).

### `size_classifier.py` — apple_small 재분류 기준 이원화
- 기존: 정상 사과 표본 평균 대비 비율(`SMALL_APPLE_SIZE_RATIO=0.8`)로만 판단
  (표본이 1개뿐이어도 그 표본 자체를 "평균"으로 써서 비교가 무의미했음).
- 변경:
  - 정상 사과 표본이 **2개 이상**이면 기존처럼 `평균 * 0.8` 비율로 판단.
  - **0~1개**뿐이면(비교 대상이 통계적으로 불충분) 절대 지름 기준
    `ABSOLUTE_SMALL_DIAMETER_MM = 60`(mm) 이하면 `apple_small`로 재분류.
  - 버그: override 로그가 `avg_diameter`를 `.1f`로 무조건 포맷하다가,
    표본 0개(`avg_diameter=None`)인 상태에서 절대 기준으로 override가
    발생하면 `TypeError`로 서비스 콜백이 통째로 죽는 문제가 있었음 → 로그
    포맷을 `None` 방어하도록 수정.
- `_avg_normal_diameter()` → `_normal_size_stats()`로 이름/시그니처 변경
  (표본 개수 + 평균을 함께 반환). `debug_overlay.py`의 호출부도 같이 갱신.

### `depth_utils.py`
- `HEIGHT_DIFF_MARGIN_MM`: 30 → 20 (작은 사과 오탐지 완화, 실측 튜닝).

## 4. 그리퍼(gripper) — pymodbus 버전 호환

- 문제: ROS 2 Humble(구버전 pymodbus 2.x대)과 Jazzy(pymodbus 3.x대)에서
  `python3-pymodbus`의 API가 다름.
  - import 경로: `pymodbus.client.sync.ModbusTcpClient`(< 3.0) vs
    `.sync` 서브모듈 없이 `pymodbus.client.ModbusTcpClient`(>= 3.0).
  - `write_registers()`의 슬레이브 ID 인자명: `unit`(구버전) → `slave`(3.x)
    → `device_id`(최신 3.7+)로 계속 바뀜.
- `openclose.py` (`code/robot_ws/src/apple_care_robot/apple_care_robot/`):
  `try/except ImportError`로 import 경로 둘 다 시도, `inspect.signature()`로
  실제 설치된 `write_registers`가 어떤 인자명을 쓰는지 런타임에 판별해서
  코드 수정 없이 두 배포판 모두 동작하게 함.
- `gripper_control.py` (신규, `code/vision_ws/src/obj_detection/obj_detection/`):
  `prepare_camera.py`가 `apple_care_robot`을 import하지 않고 독립적으로
  동작하도록, 동일한 로직을 이 패키지 안에 복제. (수정 시 두 파일 다 같이
  고쳐야 함.)

## 5. `prepare_camera.py` (신규) — 카메라 노드 실행 전 준비 스크립트

- 경로: `code/vision_ws/src/obj_detection/obj_detection/prepare_camera.py`
- `object_detection`(카메라 인식 노드)을 띄우기 전에: 그리퍼 오픈(시야 확보)
  → `HOME` → `CAMERA` 이동. 로봇 팔이 시야를 가리거나 그리퍼가 닫힌 채로
  배경 캘리브레이션(`_calibrate_scene`)이 이루어지는 걸 방지.
- `CAMERA` 좌표는 `box_sequence_test.py`와 반드시 동일해야 함 (다르면
  `camera_to_base()`의 좌표 변환 전제가 깨짐).
- 실행 순서 (`실행.txt`에도 반영):
  ```
  ros2 run obj_detection prepare_camera     # 그리퍼 오픈 + CAMERA 이동
  ros2 run obj_detection object_detection   # obd
  ros2 run apple_care_robot box_sequence_test  # acr
  ```

## 6. 힘제어 하강(`force_place.py`) — z-스톨 감지 추가

- 문제: 힘 변화량이 `force_threshold`를 못 넘는 경우(사과가 물러서 충격이
  완만하거나 센서 노이즈에 묻힘)가 있어서, 실제로는 바닥/사과에 닿아
  멈춰있는데도 타임아웃까지 기다리다 "접촉 실패"로 처리되는 문제가 있었음.
- 해결: 힘 조건과 별개로, z가 `STALL_TIME_SEC`(2초) 이상
  `STALL_Z_EPSILON_MM`(0.5mm) 넘게 안 움직이면 접촉으로 간주하고 그리퍼를
  염. 하강 루프 안에서 매 tick마다 `get_current_posx()`로 z를 읽어서, 기준
  대비 변화 없으면 스톨 타이머 누적, 움직이면 기준을 그 시점으로 갱신.

## 7. 패키지 빌드 관련 버그 (colcon/ROS2)

`apple_care_robot` 패키지가 `ros2 run`/`ros2 pkg list`에서 아예 안 잡히던
문제:

1. **`package.xml`의 XML 문법 오류**: `<description>...& Robot...</description>`의
   `&`가 이스케이프(`&amp;`) 안 되어 있어서 `catkin_pkg`가 파싱 실패 →
   colcon이 이 패키지를 ROS 패키지(`ros.ament_python`)로 인식 못 하고 그냥
   일반 파이썬 패키지로 분류 → `ament_prefix_path` 훅이 안 만들어짐.
2. **`setup.cfg` 부재**: ROS2 ament_python 패키지는 보통
   `install_scripts=$base/lib/<패키지명>`을 지정해야 실행 스크립트가
   `ros2 run`이 찾는 `lib/<pkg>/`에 설치되는데, 이 파일 자체가 없어서
   스크립트가 `bin/`에 설치되고 있었음 → `setup.cfg` 신규 추가.
3. **flat import 구조적 문제**: `box_sequence_test.py`, `apple_sorting_cycle.py`,
   `grasp_force.py`가 형제 모듈을 `from openclose import ...`처럼 상대
   경로 없이 import하고 있었는데, 이 방식은 `python3 xxx.py` 직접 실행할
   때만 되고(`sys.path[0]`에 그 디렉토리가 자동으로 들어감) `ros2 run`
   (entry_point) 방식으로는 실행 안 됨 → 전부
   `from apple_care_robot.xxx import ...` 절대 import로 변경.

## 8. 남은 튜닝 포인트 / TODO

- `vision_transform.py`의 `DEPTH_OFFSET_MM`/`SMALL_APPLE_DEPTH_OFFSET_MM`은
  실측 기반으로 계속 조정 중 (현재 -80.0 / -60.0).
- `detection.py`의 `YOLO_TRUST_CONFIDENCE`(0.7), `size_classifier.py`의
  `ABSOLUTE_SMALL_DIAMETER_MM`(60mm), `depth_utils.py`의
  `HEIGHT_DIFF_MARGIN_MM`(20mm)도 실측 데이터가 쌓이면 재튜닝 필요.
- `apple_normal`이 45mm로 측정되는데도 재분류가 안 되는 등, 표본 개수/평균
  오염 여부에 따라 여전히 애매한 경계 케이스가 있을 수 있음 - 필요하면
  `_classify_size`에 `avg_diameter`/`sample_count`를 매 호출 로그로 남기는
  디버그 옵션 추가 고려.
- `box_sequence_test.py`의 `BOXES_BY_NAME`(박스 좌표/기존 사과 존재 여부)은
  여전히 하드코딩 - 비전이 박스 내부를 직접 인식하게 확장하는 건 이번
  범위 밖.
