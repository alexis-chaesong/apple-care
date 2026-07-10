# 사과 1개 순차 실물 드라이런 — 실행 전 체크리스트

> **범위**: 이 체크리스트와 Task2 하네스(`scripts/dry_run_monitor.py`,
> `scripts/dry_run_verify.py`)는 **사과를 한 번에 하나씩 순차로 투입하는
> 드라이런** 전용입니다. 여러 개를 동시에 투입하는 §5.1 본 실험(6개 동시
> 투입, 5세트)의 동시성/경합(예: `hitl_state_machine._pending` 큐 처리,
> congestion 신호 `n(t)`가 실제로 여러 개일 때의 동작)은 **이번 검증
> 범위 밖**입니다 — 그건 로봇팀 재개(resume) 기능이 끝나고 본 실험에서
> 별도로 확인합니다.

## 0. 사전 확인 (한 번만, 매번 안 해도 됨)

- [ ] `code/.env`에 `OPENAI_API_KEY`가 유효하게 설정돼 있는가 (unknown 물체가
      한 번이라도 나오면 Track6 VLM Gate가 실제로 OpenAI API를 호출함 - 비용
      발생)
- [ ] `apple_care_msgs`, `obj_detection`, `apple_care_robot` 세 패키지가 최신
      상태로 빌드돼 있는가 (Task1에서 `SrvCaptureFrame` 인터페이스가 추가됐으니
      오래된 빌드를 쓰고 있다면 `capture_frame` 서비스가 없다고 나올 수 있음):
      ```bash
      cd code/vision_ws && colcon build --symlink-install
      cd code/robot_ws && colcon build --symlink-install
      ```
- [ ] `data/robot_system.db`를 이번 드라이런에서 **이어서 학습**시킬지,
      **깨끗한 상태로 시작**할지 결정 (후자면 드라이런 시작 전에 백업 후 삭제 -
      `init_db()`가 재생성함). 기본적으로는 이어서 쓰는 걸 권장 - 정책 학습이
      누적되는 게 이 시스템의 핵심 동작이므로.

## 1. 기동 순서 (매번)

기존 팀 런북(`실행.txt`)을 그대로 따르되, 아래에서 **한 가지 의심되는 오류**를
바로잡았습니다: `실행.txt`의 `pos_c` 별칭이 `object_detection`과 완전히 같은
명령(`ros2 run obj_detection object_detection`)으로 정의돼 있는데, 주석은
"카메라 위치로 이동"이라고 돼 있습니다. `prepare_camera.py`의 자체 docstring이
"obj_detection 카메라 노드보다 먼저 실행"해서 그리퍼를 열고 로봇을 CAMERA
자세로 이동시키는 스크립트라고 명시하므로, **`pos_c`는 `ros2 run obj_detection
prepare_camera`의 오타로 추정됩니다.** 아래 순서는 `prepare_camera`로
바로잡아서 적었습니다 — 실제 로봇 자세가 이상하면 이 부분부터 의심해보세요.

1. [ ] **로봇 드라이버 기동** (Doosan 로봇팔 ROS2 드라이버, `roboton`/`bring_rg`
       등 팀 자체 별칭 - 이 저장소 밖의 로봇팔 벤더 절차)
2. [ ] **RealSense 카메라 드라이버 기동** (`realsense` 별칭). 이후
       `rs-enumerate-devices`로 카메라가 실제로 잡히는지 확인
3. [ ] **ROS2 워크스페이스 소스** (`p2` 함수 또는 수동으로):
       ```bash
       source /opt/ros/humble/setup.bash
       source code/vision_ws/install/setup.bash
       source code/robot_ws/install/setup.bash
       ```
4. [ ] **카메라 준비 자세로 이동** (그리퍼 열기 + HOME→CAMERA 이동):
       ```bash
       ros2 run obj_detection prepare_camera
       ```
       완료 후 프로세스가 종료됨 (상시 실행 노드 아님)
5. [ ] **Vision 노드 기동** (`obd` 별칭, Task1의 `capture_frame` 서비스도 이
       프로세스 안에서 같이 뜸):
       ```bash
       ros2 run obj_detection object_detection
       ```
       로그에서 "ObjectDetectionNode initialized." 확인. 뎁스 디버그 창(로컬
       cv2 창)이 뜨면 배경 캘리브레이션이 정상 진행된 것
6. [ ] **(필요시) 바스켓 카메라 기동** - B1~B4 점유 확인이 필요한 경우만:
       ```bash
       ros2 launch obj_detection basket_camera.launch.py device_index:=<실제 인덱스>
       ```
7. [ ] **Backend 기동**:
       ```bash
       cd code/backend && uvicorn main:app --reload
       ```
       시작 로그에서 다음을 확인:
       - `apple_care_msgs를 찾을 수 없어...` 경고가 **안 떠야 함** (뜨면 6단계
         전에 ROS 소스가 안 됐거나 빌드가 안 된 것)
       - `[DB] SQLite 데이터베이스 초기화 및 테이블 생성 성공`
       - `Vision Bridge 시작: service=get_apple_status ...`
8. [ ] **로봇 실행 노드 기동** (`acr` 별칭 - `box_sequence_test.py`가 실제
       pick&place를 수행하는 운영 진입점, `apple_care.launch.py`의
       `motion_planner_node`가 아님에 주의 - 그건 아직 미완성 TODO 스텁):
       ```bash
       ros2 run apple_care_robot box_sequence_test
       ```
       로그에서 "Box sequence test: /robot/command의 START 대기 중" 확인 -
       아직 사이클을 시작 안 하고 대기만 함
9. [ ] **HMI 대시보드 기동** (선택 - 화면으로 직접 보고 싶으면):
       ```bash
       cd code/hmi_app && python3 view/dashboard.py
       ```
10. [ ] **로봇 사이클 시작 트리거** (`box_sequence_test.py`가 START를 받아야
        실제로 카메라를 보기 시작함):
        ```bash
        curl -X POST http://localhost:8000/api/robot/start
        ```

## 2. 모니터링 시작 (사과 투입 직전)

```bash
cd code/backend && python3 scripts/dry_run_monitor.py --log-file /tmp/dry_run_$(date +%Y%m%d_%H%M%S).log
```
- `tb_decision_audit`에 새 행이 쌓일 때마다 실시간 출력
- rclpy를 쓸 수 있는 환경이면 `/decision/result`, `/robot/command` 토픽도 함께 echo
- Ctrl+C로 종료 가능 (로그 파일은 그대로 보존됨)

## 3. 확인해야 할 초기 상태 요약

- [ ] 카메라가 실제로 연결/인식됨 (`rs-enumerate-devices`)
- [ ] `data/robot_system.db` 존재 및 `tb_policy_memory`/`tb_decision_audit`
      테이블 생성 확인 (backend 시작 로그로 확인 가능)
- [ ] `obd`(vision) 프로세스가 떠 있고 `get_apple_status`/`capture_frame`
      서비스 둘 다 응답 가능 상태 (`ros2 service list | grep -E "apple_status|capture_frame"`)
- [ ] backend가 vision 폴링을 시작했음 (backend 로그에 "Vision 감지 결과 큐
      적재" 라인이 주기적으로 찍히는지)
- [ ] `acr`(로봇) 프로세스가 START 대기 중이었다가, `/api/robot/start` 호출
      후 카메라 위치로 이동해 대기하는지
- [ ] (unknown 케이스를 일부러 테스트한다면) `code/.env`의 OpenAI 키가
      유효한지 - 실패하면 `tb_decision_audit`의 `stage2_vlm_call` 행에서
      `vlm_response`가 비어있거나 로그에 타임아웃/인증 에러가 남음

## 4. 드라이런 실행 (사과 1개 투입)

사용자가 직접 사과 1개를 트레이에 넣습니다. 아래를 순서대로 관찰:

1. `dry_run_monitor.py` 콘솔에 새 `tb_decision_audit` 행이 뜨는지
2. 정상/손상/작은 사과처럼 이미 학습된 케이스면 → `stage3_risk_accept_execute`
   또는 `stage3_query_human` 중 하나로 곧장 귀결
3. 완전히 낯선 물체(unknown)를 일부러 넣어본다면 → `stage2_vlm_call` 행이
   먼저 뜨고(VLM 식별 시도), 뒤이어 `stage3_query_human`(항상 사람에게
   물음), 사람이 HMI/음성으로 답한 뒤 `stage3_human_resolved` 행
4. 로봇이 실제로 해당 박스로 사과를 옮기는지 육안 확인 (HOLD 상태에서
   멈추는지도 함께 - HOLD 커맨드 자체는 no-op이므로 로봇이 멈추는 건
   `/decision/result`가 아예 발행되지 않아서라는 점 참고,
   `BEFORE_AFTER_INTEGRATION.md`의 "0단계 발견 구조 변경사항" 2/3번 참고)

## 5. 드라이런 종료 후

```bash
cd code/backend && python3 scripts/dry_run_verify.py --min-audit-id <1단계에서 기록해둔 시작 audit_id>
```
(또는 `--since "2026-07-15T10:00:00"`처럼 시각 기준으로) 이번 세션에서 쌓인
로그의 스키마 완전성과 Stage 1/2/3 경로 요약을 확인합니다.

## 알려진 제약 (재확인)

- **동시 처리 검증 범위 밖**: 사과를 한 번에 하나만 넣습니다. 여러 개를
  동시에/연속으로 빠르게 넣었을 때의 `_pending` 큐 처리, congestion `n(t)`
  변화 등은 이 드라이런으로 검증되지 않습니다.
- **이미지 캡처 인터페이스는 이번이 첫 실카메라 검증**입니다 (Task1은
  RealSense 없는 개발 환경에서 구현만 하고 실카메라로는 못 띄워봤음) -
  unknown 케이스를 테스트할 때 `capture_frame` 서비스가 실제로 이미지를
  주는지 특히 주의 깊게 봐주세요.
- **HOLD 명령은 로봇 쪽에서 no-op**입니다 - 로봇이 실제로 멈추는 것도 함께
  확인해야 안전 관련 회귀가 없는지 알 수 있습니다.
