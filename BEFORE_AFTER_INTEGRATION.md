# 베이지안 의사결정 시스템(Track 1~6) 재구현 — Vision/Robot 팀 공유용

> 이전 세션에서 구현했던 Track 1~4(loss_matrix, decision_audit, human_query_gate,
> 계층적 prior)가 커밋 전에 유실되어 재구현했습니다. `services/`, `state/`,
> `test_*.py`, `replay_session.py`, `synthetic_pipeline_run.py`가 대상입니다.
> Track 6(VLM Gate, GPT-4o 실연동)은 이후 세션에서 이 문서의 재구현 위에
> 추가됐습니다.

## 재구현이 실제로 검증한 것

- **Track 1 (`services/loss_matrix.py`)**: §4.1 손실 행렬, §4.3.1 condition 분류,
  §4.3.2 오분류 손실 근사. 지배조건(`assert L_value.max() < 1000`)이 모듈 임포트
  시점에 즉시 실행됨. 17개 pytest로 행렬 수치·condition 분류·`small→normal_box`
  문서 예시(=10) 검증 완료.
- **Track 4 (`services/decision_audit.py` + `tb_decision_audit` 테이블)**:
  append-only 보장(INSERT만, UPDATE/DELETE 문자열이 소스에 없음을 정적으로도
  확인)을 실제 반복 INSERT로 검증 — 같은 조합을 두 번 기록해도 새 행만
  추가되고 이전 행은 그대로임을 확인. 9개 pytest 통과.
- **Track 2/3 (`services/human_query_gate.py`, 하드코딩 threshold 0.65 완전
  제거)**: 동적 EVPI 게이트가 4가지 필수 시나리오(정책 없음/고신뢰 전환/혼잡
  억제/`dstored=ask_human` 강제)와 안전조건·normal gap-fill 등 추가 6가지를
  포함해 10개 pytest로 검증. `services/decision_planner.decide()`와의 실제 결합도
  7개 pytest(`test_decision_planner_stage3.py`)로 end-to-end 확인.
- **계층적 Prior (§5.4, `USE_HIERARCHICAL_PRIOR` 기본 False)**: informative
  prior 전환, 폴백(표본 0개), llm_policy/자기 자신 fruit_type 풀링 제외,
  플래그 False일 때 기존 Beta(1,1) 완전 보존을 8개 pytest로 검증.
- **Track 5-prep (`replay_session.py`)**: 고정 threshold vs 동적 EVPI 재생
  로직을 17개 pytest(synthetic 로그)로 검증. `apply_feedback_update()`를 실제
  posterior 갱신 경로(`bayesian_policy.record_human_feedback`)와 완전히
  공유하도록 순수함수로 분리해서, 시뮬레이션과 실제 갱신이 같은 코드를 탄다.
- **통합 드라이런 (`synthetic_pipeline_run.py`)**: mock Vision 관측치를
  `decide()` → `should_query_human()` → `decision_audit` 로깅까지 실제
  프로덕션 코드 경로 그대로 흘려 5개 필수 케이스(정상과일 즉시실행 / 낮은
  confidence+고신뢰 정책이어도 Stage1이 우선 / 완전 신규 조합 / safety
  condition은 고신뢰(0.75)로도 안 뚫림 / 동일 정책이 congestion만으로
  ask→execute 전환)를 확인. 이 드라이런이 남긴 감사 로그를 그대로
  `replay_session.py` CLI에 넣어 리포트가 생성되는 것도 확인.
- **§4.4 Stage2 순수성**: `log_stage2_vlm_call()` 호출 전후로 `tb_policy_memory`
  스냅샷이 완전히 동일함을 assert로 확인 — VLM 응답이 posterior를 직접
  갱신하지 않는다는 원칙이 지금 있는 코드(감사 로그 함수) 레벨에서는 지켜짐.
- **Track 6 (`services/vlm_gate.py`, GPT-4o Vision 실연동)**: `condition=="unknown"`을
  Stage1(vision.confidence) 게이트보다 먼저 분기시켜 Stage2(VLM 식별)로 보낸 뒤,
  식별된 이름(또는 실패 시 `UNIDENTIFIED_OBJECT_FRUIT_TYPE` 폴백 라벨)을
  `fruit_type`으로 삼아 **기존 Stage3 게이트(`should_query_human`)를 코드 수정 없이
  그대로** 통과시킴 - "unknown은 무조건 사람에게 물어야 한다"는 요구사항이 새
  분기 없이 `policy=None → 무조건 True` 규칙만으로 충족됨을 확인. 이 발견이
  이번 Track의 핵심 설계였음(진행 전 검증 단계에서 확인, 아래 "알려진 사소한
  특이사항" 아님). `call_gpt4o_vlm()`은 **mock이 아니라 실제 OpenAI API를
  호출하는 프로덕션 코드**이며(테스트에서만 클라이언트 patch), config의
  `openai_api_key`/`openai_model`을 `llm_service.py`와 동일한 방식으로 재사용함.
  7개(`test_vlm_gate.py`) + 7개(`test_unknown_object_vlm_integration.py`) pytest로
  검증: 이미지 있음/없음/식별 실패 3가지 경로 모두 결국 Stage3 질문까지
  도달하는지, VLM 호출 직후엔 posterior가 그대로였다가 사람 답변 후에만
  (§4.4) `tb_policy_memory`가 식별된 이름으로 기록되는지, 그 정책이 다음
  동일 물체 재등장 시 실제로 재조회되는지(= "다음번엔 자동 학습" 동작의
  근거)까지 확인함.
- **`decide()`가 동기 → 비동기로 전환됨**: Track6에서 unknown 경로가 실제
  네트워크 호출(GPT-4o)을 할 수 있게 되면서, 동기 함수로 남겨두면 `main.py`의
  이벤트 루프 전체가 그 호출이 끝날 때까지 멈추는 문제가 있어 `async def`로
  바꿈. 호출부 4곳(`main.py`, 테스트/스크립트 3곳)을 `await`/`asyncio.run()`으로
  갱신하고 전체 회귀로 재확인함.
- **전체 회귀**: 86개 pytest 전부 통과 (`test_vision_service_server.py`,
  `test_mic.py`는 ROS/마이크 하드웨어 의존이라 이번 검증 범위에서 제외).

## 여전히 mock/추정에 의존하는 것

- **Stage1 fast-path**: 여전히 코드 경로 자체가 없습니다 (`decision_audit.py`
  모듈 docstring에 명시). `log_stage1_auto_execute`는 함수만 존재하고
  프로덕션에서 아무도 호출하지 않습니다.
- **Stage2(VLM) 트리거 조건은 근사입니다**: 문서의 `EVSI_VLM(x_t) > Cost_VLM`은
  구체 계산식이 없어(다이어그램 레이블뿐) `condition == "unknown"이면 무조건
  시도`로 단순화했습니다 (`services/vlm_gate.py` 모듈 docstring 명시). 호출
  자체(Track 6)는 이제 mock이 아니라 실제 OpenAI API를 부릅니다 - 트리거
  "조건식"만 근사이지, 호출 "코드"는 실제입니다.
- **이미지 캡처는 Task1부터 실제 ROS2 서비스로 연결됐지만, 실제 카메라로
  end-to-end 검증은 아직 못 했습니다** (아래 Task1 섹션 참고 — 개발 환경에
  RealSense가 연결돼 있지 않아 물리 드라이런 때 확인 예정).
- **`k1, k2` (Cost_human 계수), `K`(계층적 prior pseudo-count), `tau_yolo=0.9`
  (감사 로그 전용, `settings.confidence_threshold`와 다름), `VLM_CALL_TIMEOUT_SECONDS`
  (Track6, 기본 10초), `IMAGE_CAPTURE_TIMEOUT_SECONDS`(Task1, 기본 3초)**:
  전부 코드 주석에 "잠정치, 물리 통합 이후 재보정 필요"라고 명시된
  placeholder입니다. 실측 근거가 전혀 없습니다.
- **`replay_session.py`의 시간 근사 (`estimate_ask_duration_sec`,
  `estimate_execute_interval_sec`)**: 실제 사과 5세트 로그가 아직 없어서
  synthetic 로그로만 검증했습니다. 실제 로그가 쌓이면 그 값으로 재검증
  필요합니다.
- **`"rotten"` condition**: 여전히 도달 불가능합니다 (`vision_bridge.py`가
  YOLO의 `apple_rotten` 클래스를 `defect_type="mold"`로 정규화해버려서
  `condition="rotten"`이라는 문자열 자체가 안 나옵니다). `loss_matrix.py`에는
  연구문서 §4.3.1과의 일치를 위해 그대로 남겨뒀습니다.
- **YOLO 오분류 구조적 한계 (Track6 범위 밖)**: YOLO가 닫힌 5-클래스 분류기
  (`apple_normal`/`apple_rotten`/`apple_damaged`/`basket`/`full_basket`)라,
  완전히 새로운 물체(예: 망치)가 confidence ≥ 0.4로 우연히 기존 클래스 중
  하나로 오분류되면 애초에 `condition="unknown"`이 되지 않아 Track6의 VLM
  게이트 자체가 트리거되지 않습니다. 이 경로는 YOLO 재학습 또는 별도 OOD
  (분포 밖) 탐지가 필요한 범위 밖 과제로 남겨둡니다 (0단계 vision_ws 조사
  결과, `code/vision_ws/src/obj_detection/obj_detection/yolo.py`).

## Task 1 — 이미지 캡처 인터페이스 연결 (vision_ws + backend)

Track6에서 스텁으로 남겨뒀던 `get_captured_frame_for_vlm()`을 실제 ROS2
인터페이스로 연결했습니다. §5.1 물리 실험 직전 세션에서 진행했습니다.

- **새 ROS2 서비스 `capture_frame`** (`apple_care_msgs/srv/SrvCaptureFrame.srv`,
  요청 필드 없음 — 호출 자체가 트리거, 응답은 `bool success / uint8[]
  image_data / string encoding`). 기존 `get_apple_status`(1Hz 정기 폴링)와는
  완전히 별개 경로로, `condition=="unknown"`일 때만 온디맨드로 호출됩니다.
  `apple_care_msgs` 패키지를 `colcon build`해서 인터페이스 생성까지 확인했습니다.
- **Vision 쪽(`detection.py`)**: `ImgNode.get_color_frame()`이 들고 있는
  최신 컬러 프레임을 그 자리에서 JPEG(품질 85, `CAPTURE_FRAME_JPEG_QUALITY`)로
  인코딩해 응답합니다. `get_apple_status`와 별도 `MutuallyExclusiveCallbackGroup`을
  줘서, 느린 폴링 서비스(~1초) 처리 중에도 `capture_frame`이 뒤에서 기다리지
  않게 했습니다.
- **동기화 방식 — 온디맨드(캐싱 아님)로 결정**: "unknown 판정 시점의 프레임을
  캐싱해뒀다가 반환" 방식도 검토했지만, (1) `SrvCaptureFrame.srv`에 요청
  필드가 없어(사용자 확정 스펙) 애초에 "어느 판정 건에 대한 캡처인지"
  상관관계를 전달할 방법이 없고, (2) 이 시스템은 한 번에 사과 하나만 CAMERA
  위치에서 처리하고 그 사과가 판정 완료 전까지 로봇이 움직이지 않으므로
  두 방식이 실질적으로 같은 프레임을 가리킬 확률이 매우 높으며, (3) 캐싱은
  유효기간/무효화 로직이라는 새 복잡도를 요구합니다. 그래서 **호출 시점의
  최신 프레임을 그대로 쓰는 온디맨드 방식**을 택했습니다 — 기존
  `debug_image` 스트림이 이미 쓰고 있던 것과 같은 종류의 근사입니다. 판정
  시점과 캡처 시점 사이의 시간차가 실제로 문제가 되는 사례가 물리
  드라이런에서 관찰되면 캐싱 방식으로 전환을 재검토할 수 있도록 이 결정과
  근거를 `detection.py`의 `handle_capture_frame()` docstring에 그대로
  남겨뒀습니다.
- **Backend 쪽(`vision_bridge.py`)**: `SrvAppleStatus` 클라이언트와 같은
  노드/executor/스레드를 재사용해 `capture_frame_client`를 추가했습니다(새
  ROS2 노드를 따로 안 띄움). 새 메서드 `request_captured_frame()`은 rclpy
  Future(ROS2 스레드에서 완료)를 `call_soon_threadsafe`로 asyncio.Future에
  중계해서 `decide()`(FastAPI 이벤트 루프)가 직접 `await`할 수 있게
  합니다 — 기존 콜백 브리지 원칙(robot_bridge.py 등)과 hitl_state_machine.py의
  `_answer_future` 패턴을 결합한 형태입니다. `services/vlm_gate.py`의
  `get_captured_frame_for_vlm()`도 스텁에서 이 함수를 호출하는 실제
  구현으로 바뀌면서 `async def`가 됐습니다(`decision_planner.py`가 이미
  `await`로 부르고 있었음).
- **검증 범위**: `apple_care_msgs`/`obj_detection` colcon build 성공, 새
  인터페이스 필드 타입 직접 확인, 전체 pytest 86개 통과(get_captured_frame_for_vlm이
  이제 실제 서비스를 시도하되 ROS 브리지 미기동 상태에서 예외 없이 None
  폴백하는 것까지 포함). **개발 환경에 RealSense D455가 연결돼 있지 않아
  (`rs-enumerate-devices`가 장치 없음을 보고함) 실제 카메라로 이미지가
  오는지는 검증하지 못했습니다** — 가짜 ROS2 서비스 서버로 배선 자체를
  검증하려 시도했으나 이 개발 환경의 백그라운드 프로세스 샌드박스 제약으로
  rclpy 노드 초기화가 실패해(포그라운드 실행은 정상 동작 확인) 완료하지
  못했습니다. §5.1 물리 드라이런 때 실제 카메라로 함께 확인이 필요합니다.

### Task 1 후속 — `ros2 pkg list`/`interface show`가 `apple_care_msgs`를 못 찾던 문제

Task 1에서 "colcon build 성공"까지 확인했지만, 그 뒤 실제로
`source install/setup.bash`를 해도 `ros2 pkg list`에 `apple_care_msgs`가
안 뜨고 `ros2 interface show`/`ros2 service call`도 "Unknown package"·"The
passed service type is invalid"로 실패하는 문제가 있었습니다. 원인과 고친
내용을 팀 전체가 같은 문제를 겪을 수 있어 남겨둡니다.

- **원인**: 이 개발 머신에 깔린 `colcon-ros 0.5.0`의 `ament_cmake` 빌드
  태스크(`AmentCmakeBuildTask`)가 `AMENT_PREFIX_PATH` 환경 훅을 등록하지
  않는 결함이 있습니다. 같은 패키지의 `ament_python` 빌드 태스크는 이 훅을
  직접 만들어 넣는데(`colcon_ros/task/ament_python/build.py`의
  `create_environment_hook('ament_prefix_path', ...)`), `ament_cmake`
  쪽에는 이에 대응하는 코드가 없습니다. CMake/`ament_cmake_core` 자체는
  `share/<pkg>/environment/ament_prefix_path.sh`를 정상적으로 만들지만,
  워크스페이스 최상위 `install/setup.bash`가 실제로 읽는 `package.dsv`
  목록에는 이 훅이 빠진 채로 colcon이 다시 써버립니다. 그 결과
  `rosidl_generate_interfaces`만 쓰는 순수 메시지 패키지(`apple_care_msgs`,
  그리고 별개 워크스페이스인 `cobot_ws`의 `od_msg`에서도 동일하게 재현)는
  빌드는 성공해도 `AMENT_PREFIX_PATH`에 전혀 추가되지 않아 `ros2` CLI가
  찾지 못합니다. 이 머신 전체의 colcon 툴체인 문제이지, `apple_care_msgs`
  자체 설정 문제가 아닙니다.
- **고친 방법**: `tools/colcon_ament_prefix_path_fix/`에 작은 로컬 colcon
  확장을 추가했습니다. `colcon_core.environment` 확장 포인트에
  `ament_prefix_path`라는 이름으로 등록되어(정확히 colcon-ros가
  `ament_python`에 하던 것과 같은 방식), `share/ament_index/resource_index/
  packages/<pkg>` 마커가 있는 모든 패키지에 대해 `AMENT_PREFIX_PATH` prepend
  훅을 만들어줍니다. **한 번만** `pip3 install --user -e
  tools/colcon_ament_prefix_path_fix`로 설치하면 이 사용자 계정에서 실행하는
  이후의 모든 `colcon build`(vision_ws뿐 아니라 cobot_ws 등 다른
  워크스페이스도 포함)에 자동 적용됩니다. 시스템 패키지(`/opt/ros/...`,
  `/usr/lib/python3/dist-packages/...`)는 전혀 건드리지 않으므로 `sudo`가
  필요 없고 apt 업그레이드에도 안전합니다.
- **팀원들도 같은 증상을 겪으면**: `pip3 install --user -e
  tools/colcon_ament_prefix_path_fix` 한 번 실행 후 해당 패키지를 다시
  `colcon build`하면 됩니다(`rm -rf build install log` 클린 빌드까지는
  필요 없고, 그냥 재빌드하면 `package.dsv`가 갱신됩니다).
- **검증**: `apple_care_msgs`/`obj_detection` 클린 리빌드 후
  `source install/setup.bash` → `ros2 pkg list | grep apple_care` →
  `apple_care_msgs` 출력, `ros2 interface show
  apple_care_msgs/srv/SrvCaptureFrame` → `success`/`image_data`/`encoding`
  필드 정상 출력 확인. 동일한 방식으로 `cobot_ws/od_msg`도 재현·수정
  확인했습니다.
- **별개로 발견한 사소한 이슈**: `~/.bashrc`에 `ROS_DOMAIN_ID`가 두 번
  정의돼 있습니다(100, 그다음 99 — 나중 것이 최종 적용됨). 이번 패키지
  탐지 문제와는 무관함을 확인했습니다(`AMENT_PREFIX_PATH` 기반 파일
  시스템 탐색은 DDS 도메인 ID와 별개 메커니즘)만, 의도치 않은 중복 정의라
  정리가 필요해 보입니다.

## Task 2 — 사과 1개씩 순차 실물 드라이런 하네스

물리 실험 준비물만 만들었고, 실제 사과 투입은 아직 하지 않았습니다.

- **`DRY_RUN_CHECKLIST.md`**(저장소 루트): 기동 순서 체크리스트. 기존
  팀 런북(`실행.txt`)을 따르되, `pos_c` 별칭이 `object_detection`과 동일한
  명령으로 잘못 정의돼 있어(주석은 "카메라 위치로 이동"인데 실제로는
  `prepare_camera`가 아니라 `object_detection`을 또 실행함) `prepare_camera`로
  바로잡아 적었습니다 — 실물 드라이런 때 실제 별칭 정의도 함께 확인 필요합니다.
- **`code/backend/scripts/dry_run_monitor.py`**: `tb_decision_audit`를 폴링해
  새 행이 생길 때마다 branch별 핵심 필드를 콘솔에 출력. rclpy를 쓸 수 있는
  환경이면 `/decision/result`/`/robot/command` 토픽도 함께 echo(스레드 없이
  구독 콜백 + 타이머 콜백만으로 구현, 이 코드베이스의 다른 노드들과 동일한
  관례). synthetic 데이터로 폴링/포맷 로직 동작을 확인했습니다.
- **`code/backend/scripts/dry_run_verify.py`**: 세션 종료 후 로그를 모아
  branch별 필수 필드 null 여부, `query_human`/`final_action` 논리적 일관성을
  점검하고 Stage 1/2/3 경로를 요약. `services.decision_audit.VALID_BRANCHES`를
  동적으로 참조해서, 스키마가 나중에 바뀌면 이 검증 스크립트가 자동으로
  "알 수 없는 branch"를 잡아내도록 했습니다. synthetic_pipeline_run.py의
  6개 케이스로 실제 실행해 스키마 완전성 검사(이상 0건)와 Stage 경로 요약이
  올바르게 나오는 것을 확인했습니다.
- **명시적 범위 제한**: 체크리스트와 두 스크립트 모두 "사과 1개 순차 처리"
  전제를 문서/주석에 반복해서 명시했습니다 - 여러 물체 동시 투입 시나리오의
  동시성/경합(§5.1 본 실험)은 이번 하네스의 검증 범위 밖입니다.

## 0단계에서 발견한 구조 변경사항 (Vision/Robot 팀 확인 필요)

1. **`/decision/result` 토픽 구독자가 2곳으로 늘어났습니다.** 기존 실제
   구동 노드(`box_sequence_test.py`, 구 `apple_sorting_cycle.py`)는 그대로
   pick&place를 즉시 실행하지만, 새로 추가된 `motion_planner_node.py`는
   같은 토픽을 구독하되 4개씩 모아 `/motion/queue`로 재발행만 하는 미완성
   TODO 스텁입니다. 백엔드 로직에는 영향이 없지만, 두 노드가 동시에
   기동되면 판정 하나가 중복 처리될 위험이 있어 별도 이슈로 분리해뒀습니다.
2. **HOLD 커맨드는 여전히 로봇 쪽에서 no-op입니다.** `box_sequence_test.py`의
   `robot_command_callback`에 "HOLD/MANUAL_PAUSE 등 사이클 도중 정지/재개는
   이번 연동 범위 밖"이라는 명시적 주석과 함께 처리되지 않습니다.
   `query_human=True`일 때 로봇이 멈추는 실제 메커니즘은 여전히 "다음
   `/decision/result`가 발행되지 않아서"(수동적 대기)이며, HOLD 명령 자체가
   능동적으로 로봇을 세우지는 않습니다. 안전 관련이라 재확인이 필요합니다.
3. **`decide()`에는 Stage1 fast-path가 없습니다.** vision.confidence 게이트를
   통과한 뒤에도 항상 Stage3 EVPI 게이트를 거치며, 유일한 execute 분기는
   Stage3 risk-accept입니다 (Track4 재발방지 항목 그대로 재확인됨).
4. **`DecisionResult.reason`이 `"rule_match"/"memory_match"`에서
   `"risk_accept_execute"`로 바뀌었습니다.** execute 분기는 이제 정책
   출처(llm_policy/human_feedback)와 무관하게 항상 "위험 감수 실행"이라는
   근거로 통일됩니다 (Track2/3 요구사항 - 명시적 로깅). HMI 등 외부
   소비자가 이 문자열을 하드코딩해서 비교하는 곳은 없는지 확인했고, 현재는
   없습니다.

## Track 6 구현 중 발견해서 함께 고친 것

- `hitl_state_machine.py`의 TTS 질문 생성 로직(`_ask_and_wait`)이
  `session.fruit_type != "unknown"`만 확인하고 있었는데, Track6부터
  `session.fruit_type`이 더 이상 `"unknown"` 리터럴이 아니라 VLM 식별
  결과이거나 실패 시 `UNIDENTIFIED_OBJECT_FRUIT_TYPE`("unidentified_object")
  폴백 라벨이라, 고치지 않았으면 "unidentified_object 처리 방법을 확인해
  주세요"처럼 내부 기술 라벨이 그대로 TTS/LLM 프롬프트에 새어나갈 뻔했습니다.
  `UNIDENTIFIED_OBJECT_FRUIT_TYPE`도 함께 걸러서 그 경우엔 "미확인 물체"라는
  자연스러운 한국어로 치환하도록 고쳤습니다.
- `decision_audit.py`의 `_insert_audit_row`가 `vlm_response`를
  `json.dumps(..., ensure_ascii=False)` 없이 저장하고 있어서, GPT-4o가
  한국어로 식별한 물체명이 `tb_decision_audit`에 `\uXXXX` 이스케이프로
  저장되고 있었습니다(테스트 작성 중 발견). DB를 직접 조회할 때 가독성이
  떨어지는 문제라 고쳤습니다 - 스키마/동작은 변경 없음.

## 알려진 사소한 특이사항

- Stage3 EVPI 게이트로 인한 `ask_human`은 실제로는 정책 신뢰 부족 때문이어도
  `reason` 필드가 `"unknown_object"`(condition="unknown"일 때만) 또는
  `"low_confidence"`로만 나옵니다 — Stage1 confidence 실패든 Stage3 EVPI
  게이트든 구분 없이 `"low_confidence"`로 뭉뚱그려집니다. 이건 유실 전
  원본 코드(디컴파일로 복구한 바이트코드)에도 동일하게 있던 특성이라 그대로
  유지했습니다. 로그/디버깅 시 `tb_decision_audit.branch` 컬럼(`stage3_query_human`
  등)을 보면 정확한 분기를 알 수 있으니, 필요하면 이 `reason`을 세분화하는
  건 별도 작업으로 진행하는 걸 권장합니다.
