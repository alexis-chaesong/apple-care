# 작업 요약 — E-STOP 복구 버그 수정 + 바스켓 클래스 인식 정리 + 트레이 범위 제한

**브랜치:** `feat--vision_basket` (병합 기준: `c6c2646`)
**커밋:** `b0c280e`, `09272e0` (+ 아래 3번 항목은 이 문서 작성 시점 기준 아직 미커밋)

## 1. `b0c280e` — E-STOP 복구 후 상승 안 되는 문제 + 프론트 재개 버튼

**증상:** 비상정지 해제 후 복구 절차에서 충돌 회피용 상승(lift)이 로그상 "정지 확인
완료"로 찍히는데도 로봇이 실제로는 안 움직임. 여러 가설(HOLD stop_mode,
check_motion 타이밍 레이스, 절대/상대좌표, z축 부호)을 실기로 검증해가며 좁힘.

- **`estop_handler.py`**
  - lift 단계 실제 z 변화량 검증 + 재시도(3회), 실패 시 `RuntimeError`로 나머지
    복구 이동 중단 (기존엔 실패해도 진행해서 충돌 위험 있었음).
  - lift를 절대좌표 대신 **상대이동(REL)**으로 전환.
  - `wait_for_resume` 옵션 추가: 물리 복구가 끝나도 곧바로 READY로 안 넘어가고,
    운영자가 프론트 재개 버튼을 눌러야만(RESUME 명령) 다음 사이클(비전 재탐지)이
    재개되도록 게이트 추가.
- **`safe_motion.py`**: `stop_motion()` 기본 stop_mode를 `DR_HOLD`에서
  `DR_QSTOP`으로 변경 (자매 프로젝트 auto_dump_robot_pkg에서 검증된 방식).
- **`box_sequence_test.py`**
  - 최초 1회 `HOME→CAMERA` 시작 시퀀스, 종료 시 마지막 `HOME` 이동에
    `try/except EmergencyStopError` 추가 (이 구간은 메인 루프 밖이라 기존
    예외처리로 안 잡혀서 E-STOP 시 프로세스 자체가 죽는 별개 버그였음).
  - `/robot/command`에 `RESUME` 명령 처리 추가.
  - `_recovery_movel`에서 REL 모드일 때 z축 부호 처리 + `posx(...)` 타입
    재포장(DSR `Invalid type: pos` 방지).
- **`force_place.py`**: `release_force()`/`release_compliance_ctrl()`도 성공/실패
  반환값이 있는데 확인 안 하고 있던 것 발견 → 확인 + 3회 재시도 + 실패 시 에러
  로그 추가.
- **백엔드/프론트**: `POST /api/robot/resume` 엔드포인트 추가, 대시보드 긴급정지
  팝업 버튼을 실제 재개 트리거로 변경 (기존엔 화면만 닫고 로봇엔 아무 영향 없었음).
- **`test_estop_handler.py`**: 위 변경 전부에 대한 회귀 테스트 추가, 32개 전부 통과.

## 2. `09272e0` — 바스켓 클래스 인식 수정 + 로봇/비전 판단 로직 정리

**증상:** 바스켓을 새로 학습시켰는데 `apple_small`로 인식됨.

**원인:** 실제 배포 모델(`best_apple_care.pt`)을 직접 열어 확인한 결과, basket이
5번째 클래스로 추가된 게 아니라 `apple_small` 자리(인덱스 3)를 **대체**했음
(`{0: apple_normal, 1: apple_rotten, 2: apple_damaged, 3: basket}`). 코드의
`CLASS_NAMES`가 옛 매핑을 쓰고 있어서 발생.

- **`yolo.py`**: `CLASS_NAMES`를 실제 모델과 일치하도록 수정.
- **`debug_overlay.py`**: basket bbox는 파란색으로 고정 표시 (인식/시각화만 유지).
- **로봇/비전 판단 로직은 이후 요청으로 다시 제거함** — 바스켓 인식은
  "탐지 + bbox 표시"만 남기고:
  - `box_sequence_test.py`: `detect_box_occupancy()`(비전으로 박스 점유 확인)
    완전 삭제. `place_apple_in_box()`는 이제 항상 동일하게 고정 절대좌표
    (box_pos)로 이동 후 힘제어 하강 — 점유 확인 분기 없음. ("세컨카메라로 할
    예정"이라는 커밋 메시지대로 이 부분은 추후 별도 카메라로 재구현 예정)
  - `force_place.py`: 이제 안 쓰는 `CAREFUL_APPROACH_*`, `EXISTING_APPLE_*`
    상수 제거.
  - `detection.py`/`yolo.py`: basket 감지 시 상태를 특별 판정하던 로직
    ("empty" 단락회로, 사과 우선순위 선택) 전부 제거 — basket은 그냥 인식되는
    클래스 중 하나일 뿐, 판정 파이프라인에는 관여 안 함.

## 3. (미커밋) 픽업 좌표를 트레이 작업 영역 안으로 제한

**증상:** discard_box행 사과 파지가 3회 연속 실패. 로그 분석 결과:
- pick 좌표(x≈723~745)가 `wall_avoidance.py`의 `TRAY_X_MAX_MM(607.58)`을
  120~140mm 벗어난 위치로 나왔음.
- 그런데도 `_nearest_wall_escape_dir()`가 "벽 근처"로 오판해서 대각선 접근을
  시도함 — 경계를 벗어나면 거리 계산값이 **음수**가 되는데,
  `dist <= WALL_PROXIMITY_MARGIN_MM` 조건이 음수도 그냥 통과시켜버려서
  "벽에 딱 붙어있음"과 "범위를 한참 벗어남"을 구분 못 하는 버그였음.
- z는 `vision_transform.py`의 `MIN_Z_MM=2.0` 클램프에 정확히 걸림 — 원래
  계산된 깊이가 테이블을 뚫고 들어갈 정도로 낮게 나왔다는 뜻. grasp 힘 로그도
  400까지 힘을 올려도 반발력 변동이 0.5N 미만(정상 파지 때는 ~6N대)이라,
  그리퍼가 사과를 아예 못 물고 허공/테이블 근처에서 닫힌 것으로 보임.

**수정:**
- `wall_avoidance.py`: `is_within_tray_bounds(x, y)` 함수 추가 —
  `TRAY_X_MIN_MM~MAX_MM`, `TRAY_Y_MIN_MM~MAX_MM` 범위 안인지 정확히 확인.
- `box_sequence_test.py`:
  - vision이 준 최초 pick 좌표가 이 범위 밖이면, 접근 시도 자체를 하지 않고
    그 사과를 건너뜀 (`ERR_PICK` 안전 이벤트 발행 + 로그).
  - 그립 실패 후 재획득한 좌표도 동일하게 검증 — 범위 밖이면 무시하고 이전
    좌표로 재시도.
- `wall_avoidance.py`: `_nearest_wall_escape_dir()`의 근본 버그도 같이 수정함 —
  margin 조건을 `dist <= WALL_PROXIMITY_MARGIN_MM`에서
  `0 <= dist <= WALL_PROXIMITY_MARGIN_MM`로 하한을 둬서, 경계를 이미 벗어난
  축(음수 margin)은 "벽 근처"로 안 잡히도록 함. `is_within_tray_bounds()`가
  앞단에서 걸러주는 정상 흐름에서는 이 함수까지 범위 밖 좌표가 들어올 일이
  없어야 정상이지만, 함수 자체의 불변조건을 명확히 하기 위해 방어적으로 고침.
- `test_wall_avoidance.py` 신규 추가: 경계값(포함)/범위 밖(x/y 각각) 케이스,
  `_nearest_wall_escape_dir`/`compute_wall_aware_approach`의 정상 벽-근접 감지 +
  범위 밖 무시 회귀 테스트 총 9개.
- `pytest test/` 41개 전부 통과 (기존 32 + 신규 9).
