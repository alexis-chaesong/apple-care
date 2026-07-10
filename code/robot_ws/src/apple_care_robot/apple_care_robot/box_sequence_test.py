"""
box_sequence_test.py (유일한 실행 진입점 - pick_helpers.py에서 공용 함수 제공)
=======================================================================================

백엔드 연동 (apple-care/code/backend 참고):
    - /decision/result (String/JSON) 구독: 백엔드의 decision_planner.decide()가
      action="execute"로 판단한 결과가 여기로 들어옴.
      형식: {"fruit": str, "destination": "normal_box"|"processing_box"|
             "discard_box"|"ugly_box", "pose": [x,y,z] 또는 None,
             "condition": "small"|"unknown"|그 외, "reason": str}
      Vision이 아직 판정을 못 내렸거나 확신이 낮은 사과는 ask_human 경로로 빠져서
      여기로 아예 들어오지 않음 (백엔드 HITL 흐름, 로봇 쪽은 신경 쓸 필요 없음).
    - /robot/command (String/JSON) 구독: {"command": "START"|"EMERGENCY_STOP"|...}
      START가 와야 사이클을 시작하고, EMERGENCY_STOP이 오면 즉시 정지 후
      estop_handler.py의 복구 절차를 밟음.
    - StatusBus로 /robot/process_state, /robot/motion_status, /gripper/status,
      /robot/safety_event를 발행해서 백엔드/HMI가 현재 상태를 알 수 있게 함.

박스 매핑 (destination -> 실제 박스, 실제 배치로 확정됨):
    processing_box -> b1 (경유점: way1)
    ugly_box       -> b2 (경유점: way1)
    normal_box     -> b3 (경유점: way1)
    discard_box    -> b4 (경유점: way2)

박스 안에 이미 사과가 있는지는 비전으로 확인하지 않음 - place_apple_in_box는
항상 동일하게 box_pos(고정 절대좌표)까지 이동한 뒤 force_controlled_place로
접촉을 감지하며 내려가므로, 이미 사과가 있어도 접촉 시점에 멈춰서 안전함.

그립 자체가 실패하는 경우(사과를 놓침)에도 pick_pos 하나로 계속 헛손질하지 않도록,
CAMERA로 돌아가 get_apple_status를 다시 호출해 최신 위치를 받아온 뒤 재시도함
(refresh_pick_pos, MAX_PICK_ATTEMPTS번까지).

동작 순서 / HOME 관련:
    - HOME은 사이클 전체에서 맨 처음(1회)과 맨 마지막(1회)에만 거침.
      그 사이에는 매 사과마다 CAMERA -> (집기) -> CAMERA -> 경유점 -> 박스
      -> 경유점 -> CAMERA 순으로만 돌고, 박스로 가기 전/집은 뒤에 HOME을
      다시 거치지 않음.

연혁: 원래 이 백엔드 연동 사이클은 apple_sorting_cycle.py(현 pick_helpers.py)의 main()이었고,
box_sequence_test.py는 백엔드 없이 로컬 vision 호출만으로 도는 독립 하드웨어
검증 스크립트였음. 두 파일이 거의 같은 ~500줄짜리 루프를 따로 유지하다가 머지할
때마다 코드가 꼬이는 문제가 반복돼서(PR #18 머지 후 두 파일 다 깨져서 실행 자체가
안 됐음), 실행 진입점을 이 파일 하나로 통합함. pick_helpers.py에는 이제
pick_apple()/_depth_offset_for_condition() 같은 공용 함수만 남아있음.
"""

import json
import queue
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor
import DR_init
from std_msgs.msg import String
from apple_care_msgs.srv import SrvAppleStatus, SrvBasketStatus
from apple_care_robot.openclose import gripper_open
from apple_care_robot.pick_helpers import (
    pick_apple, _depth_offset_for_condition, DEFAULT_PICK_ORIENTATION,
)
from apple_care_robot.vision_transform import camera_to_base
from apple_care_robot.wall_avoidance import is_within_tray_bounds, compute_wall_aware_approach
from apple_care_robot.status_bus import StatusBus
from apple_care_robot.estop_handler import EstopRecoveryTracker, check_and_recover
from apple_care_robot.safe_motion import (
    safe_movel, safe_movej, EmergencyStopError, is_controller_in_hardware_safety_stop,
    get_controller_robot_state, ROBOT_STATE_NAMES, reset_controller_safety_stop,
)

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

TOPIC_DECISION_RESULT = "/decision/result"
TOPIC_ROBOT_COMMAND = "/robot/command"

# 반드시 앞에 '/'를 붙인 절대 이름이어야 함. 이 노드가 namespace="dsr01"로 생성되기
# 때문에("box_sequence_test" 아래 참고), 상대 이름("get_apple_status")을 쓰면
# rclpy가 자동으로 "/dsr01/get_apple_status"로 리졸브해버려서 obj_detection이
# 네임스페이스 없이 여는 실제 서비스(/get_apple_status)를 영영 못 찾음
# (pick_and_place_voice/robot_control.py의 "/get_3d_position"과 동일한 이유).
VISION_SERVICE_NAME = "/get_apple_status"

# backend/basket_bridge.py가 네임스페이스 없이 여는 실제 서비스(get_basket_status)를
# 찾기 위해 VISION_SERVICE_NAME과 동일한 이유로 절대 이름을 씀.
BASKET_SERVICE_NAME = "/get_basket_status"

# 그립 자체가 실패하는 경우(사과를 놓침 - grasp_apple_with_force_feedback이
# picked_ok=False)를 대비한 재시도 횟수. 매번 CAMERA로 돌아가서 위치를 다시
# 받아온 뒤 집기를 재시도함.
MAX_PICK_ATTEMPTS = 3

# place_apple_in_box가 힘제어 하강(force_controlled_place)을 시작하는 높이를
# box_pos(고정 절대좌표) z보다 이만큼(mm) 더 높게 잡음. 실측으로 확인된 문제:
# box_pos까지 바로 movel한 뒤 그 자리에서 곧바로 힘제어를 시작하면 하강 시작
# 높이가 너무 낮아서(박스 바닥/이미 쌓인 사과와 거의 붙어있는 상태) 부드러운
# 접촉 감지 여유가 부족했음. force_controlled_place는 어차피 최대
# MAX_DOWN_DISTANCE(force_place.py, 165mm)까지 내려가며 접촉을 찾으므로, 이만큼
# 더 높은 지점에서 시작해도 정상 범위 안에서 접촉을 찾음.
#
# 원래 30mm였는데, 시작점을 높일수록 그만큼 MAX_DOWN_DISTANCE 중 "실제 접촉면
# 도달 전 그냥 내려가기만 하는 구간"이 늘어나서 접촉면 기준 실제 탐색 가능
# 깊이가 줄어드는 부작용이 있었음(30mm 클리어런스면 150mm 중 120mm만 접촉면
# 아래 탐색 가능). 그래서 10mm로 줄임 - force_place.MAX_DOWN_DISTANCE(165mm)는
# 그대로 둬서, 오히려 접촉면 기준 탐색 깊이가 원래(150mm)보다 넉넉한 155mm로
# 확보됨(시간 여유 TIMEOUT_SEC도 MAX_DOWN_DISTANCE 기준 그대로 유지).
#
# 이 값은 박스 배치(place_apple_in_box) 전용임 - 작업대 재배치
# (place_apple_back_on_worktable)는 별도로 WORKTABLE_FORCE_START_CLEARANCE_MM을
# 씀. 원래 place_apple_back_on_worktable을 추가할 때 이 상수를 그대로 재사용했는데,
# 그 뒤 박스 배치 튜닝(30 -> 15 -> 10)이 의도치 않게 작업대 재배치에도 같이
# 적용돼버렸던 문제가 있어서 분리함.
PLACE_FORCE_START_CLEARANCE_MM = 10

# 바스켓이 occupied로 판단됐을 때 way_pos -> force_start_pos로 접근하는 movel의
# 속도/가속도. 실측으로 확인된 문제: force_start_pos는 실제 쌓인 사과 높이를 측정한
# 값이 아니라 PLACE_FORCE_START_CLEARANCE_MM_OCCUPIED(고정 추정 오프셋)일 뿐이라,
# 더미가 그보다 높으면 force_controlled_place(컴플라이언스 진입)가 시작되기도 전인
# 이 구간(rigid position-control movel)에서 그대로 사과더미에 부딪혀 충돌 오류로
# 이어짐 - 바스켓이 비어있을 때는 같은 구간에 아무것도 없어서 기본 순항 속도
# (set_velx(45,30))로도 문제가 없었지만, occupied일 때는 그대로 쓰면 안 됨.
# 접근 속도를 낮춰서 예상보다 일찍 걸리는 저항/충돌에 대한 반응 여유와 실제
# 충돌 시 충격/초과 진입 거리를 줄인다. 기본 순항(45/30)과 최초값(15/15) 중간인
# 25/20으로 조정. force_start_pos 도달 이후(force_controlled_place의 컴플라이언스
# 하강 속도, force_place.py의 DOWN_SPEED_VEL/ACC)는 이 값과 무관하게 기존 그대로 둠.
OCCUPIED_APPROACH_VEL = 25
OCCUPIED_APPROACH_ACC = 20

# 바스켓에 이미 사과가 있으면(get_basket_status로 조회) 그 위에서 접촉을
# 감지해야 하므로, 빈 바스켓 기준(10mm)보다 시작 높이 여유를 더 둔다. 아직
# 실기 튜닝 전 새 기능이라, WORKTABLE_FORCE_START_CLEARANCE_MM(작업대 재배치 -
# 역시 "바닥이 아니라 뭔가 위에 착지"하는 동일한 상황)과 같은 값을 초깃값으로
# 사용함 - 실기 테스트 후 조정할 것.
PLACE_FORCE_START_CLEARANCE_MM_OCCUPIED = 70

# place_apple_back_on_worktable 전용 힘제어 시작 높이 클리어런스(mm). 박스 배치와
# 달리 아직 실기로 튜닝된 적 없는 새 기능이라, PLACE_FORCE_START_CLEARANCE_MM의
# 박스 전용 튜닝(10mm)에 영향받지 않도록 원래 값(30mm)을 그대로 씀 - 실기 테스트
# 후 필요하면 이 값만 따로 조정할 것.
WORKTABLE_FORCE_START_CLEARANCE_MM = 60


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("box_sequence_test", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    # DSR 제어용 메인 node와 완전히 분리된 comm_node. /decision/result,
    # /robot/command 구독 + StatusBus 발행을 이 node에서 spin하면, movel/wait이
    # 메인 스레드를 블록하는 동안에도 START/EMERGENCY_STOP을 받을 수 있음.
    comm_node = rclpy.create_node("box_sequence_test_comm", namespace=ROBOT_ID)
    status_bus = StatusBus(comm_node)
    status_bus.set_state("INIT")

    from DSR_ROBOT2 import (
        movel, movej, posx, posj, wait,
        set_velx, set_accx, set_velj, set_accj, get_current_posx,
        amovel, amovej, check_motion, DR_BASE,
        DR_MV_MOD_ABS, DR_MV_MOD_REL,
    )
    from apple_care_robot.force_place import force_controlled_place, TIMEOUT_SEC as BASE_PLACE_TIMEOUT_SEC

    # 바스켓 occupied라 접근 속도를 낮춘 경우(OCCUPIED_APPROACH_VEL/ACC) 전용
    # 힘제어 타임아웃. 기본 타임아웃(BASE_PLACE_TIMEOUT_SEC)을 그대로 쓰면 감속
    # 접근 때문에 실기에서 접촉 판정 여유(4.5초) 안에 못 끝나 작업대로 재배치되는
    # 사례가 재현됨 - 이 케이스에만 3초 더 넉넉하게 줌. 비어있는 바스켓(기본 속도)
    # 배치나 작업대 재배치까지 전부 늘리면 실제로 필요 없는 곳까지 실패 판정이
    # 느려지므로 occupied일 때만 force_controlled_place에 넘긴다.
    OCCUPIED_PLACE_TIMEOUT_SEC = BASE_PLACE_TIMEOUT_SEC + 3.0

    # 좌표 정의
    HOME = posj(0, 0, 90, 0, 90, 0)
    CAMERA = posx(493.30, -104.81, 247.48, 26.09, 175.22, 23.18)
    WAY1 = posx(446.54, 244.89, 206.64, 79.38, 177.62, 172.06)
    WAY2 = posx(740.15, 189.44, 182.73, 170.37, -162.55, -103.95)

    B1 = posx(271.57, 248.16, 40.72, 90.88, -177.84, 89.87)
    B2 = posx(466.13, 239.03, 40.71, 39.87, -175.85, 55.2)
    B3 = posx(635.23, 246.09, 40.71, 169.25, -179.16, -169.02)
    B4 = posx(786.27, 185.37, 40.23, 8.77, 158.32, 91.18)

    # destination -> (박스 이름, 박스 좌표, 경유점). 실제 물리 배치 확정:
    # b1=가공용, b2=못난이, b3=정상, b4=폐기.
    DESTINATION_TO_BOX = {
        "processing_box": ("b1", B1, WAY1),
        "ugly_box":       ("b2", B2, WAY1),
        "normal_box":     ("b3", B3, WAY1),
        "discard_box":    ("b4", B4, WAY2),
    }

    emergency_stop = threading.Event()
    estop_tracker = EstopRecoveryTracker()

    # E-STOP 물리 복구(상승/홈/카메라) 후, 운영자가 프론트엔드에서 재개 버튼을 눌러
    # /robot/command로 RESUME을 보내야만 다음 사이클(비전 재탐지)을 진행하도록 하기
    # 위한 게이트. 이전에는 물리 복구가 끝나면 무조건 자동으로 READY까지 갔는데,
    # 안전정지 상황에서는 운영자가 현장을 확인한 뒤 명시적으로 재개시켜야 한다는
    # 요구사항으로 추가함 (estop_handler.check_and_recover의 wait_for_resume 참고).
    resume_requested = threading.Event()

    def wait_for_resume_signal() -> None:
        # 대기 시작 직전에 clear해서, 이번 E-STOP 이전에 눌렸을 수 있는 낡은
        # RESUME 신호 때문에 이번 게이트가 곧바로(운영자 확인 없이) 통과되는 걸
        # 막음 - 여기서 clear한 뒤에 오는 set()만 이 wait()를 깨움.
        resume_requested.clear()
        resume_requested.wait()
        resume_requested.clear()

    # 프론트엔드 "PAUSE ROBOT MOTION" 버튼(/api/robot/pause -> MANUAL_PAUSE)을 위한
    # 게이트. E-STOP과 달리 물리 복구(상승/홈 이동)가 필요 없는 상황이라 - 그냥
    # "다음 사과 집기를 새로 시작하지 않고 대기"만 하면 됨. 이미 진행 중인
    # 이동/파지 한 사이클은 안전하게 끝까지 마친 뒤(중간에 갑자기 멈추면 오히려
    # 위험할 수 있음), 다음 while 루프 최상단에서만 이 플래그를 확인함.
    # RESUME 명령(같은 /robot/command 토픽)이 오면 이 일시정지도 함께 풀림 -
    # 프론트엔드의 PAUSE 버튼이 두 번째 클릭에서 RESUME을 보내도록 되어 있음.
    pause_requested = threading.Event()

    # 이동/힘제어 폴링 루프 안에서 공용으로 쓸 하드웨어 안전정지 조회 콜백.
    # comm_node를 넘기는 이유: stop_motion()과 동일하게 DSR 제어용 메인 node를
    # 여기서 또 spin하면 충돌 위험이 있음 (comm_node는 이미 별도 스레드에서 spin 중).
    def check_hw_safety_stop():
        return is_controller_in_hardware_safety_stop(comm_node)

    def describe_robot_state() -> str:
        # lift movel이 서비스 응답은 성공으로 받으면서도 실제로는 z가 안 바뀌는
        # 문제의 원인을 좁히기 위한 진단용 - 그 순간 컨트롤러가 실제로 어떤 상태
        # (STANDBY/MOVING/SAFE_STOP 등)를 보고하는지 로그로 남김.
        state = get_controller_robot_state(comm_node)
        if state is None:
            return "UNKNOWN(조회 실패)"
        return f"{ROBOT_STATE_NAMES.get(state, 'UNKNOWN')}({state})"

    def do_safe_movel(target, vel=None, acc=None, ref=None):
        safe_movel(
            comm_node, target, emergency_stop,
            amovel=amovel, check_motion=check_motion, wait=wait,
            vel=vel, acc=acc, ref=ref,
            check_hw_safety_stop=check_hw_safety_stop,
        )

    def do_safe_movej(target, vel=None, acc=None):
        safe_movej(
            comm_node, target, emergency_stop,
            amovej=amovej, check_motion=check_motion, wait=wait,
            vel=vel, acc=acc,
            check_hw_safety_stop=check_hw_safety_stop,
        )

    def recover_from_estop_if_needed() -> bool:
        # get_current_pos/movel 둘 다 ref를 명시적으로 DR_BASE로 고정함 - _g_coord
        # 전역 기본값에 의존하면, 다른 코드가 나중에 set_ref_coord()로 그 값을
        # 바꿔도 이 복구 로직은 항상 확실한 BASE 기준으로 동작하도록 하기 위함
        # (force_place.py가 자기 이동에 항상 ref=DR_BASE를 명시하는 것과 동일한 이유).
        #
        # 동기(movel/movej) 대신 반드시 비동기(amovel/amovej)를 넘겨야 함.
        # execute_recovery()는 매 구간(상승->홈->원위치/카메라)마다 이동 직후
        # _wait_until_fully_stopped()로 check_motion()이 "정지"를 확인해줄 때까지
        # 폴링한 뒤 추가로 settle_sec만큼 더 기다리는데, 동기 함수는 호출 시점에
        # 이미 이동이 다 끝난 뒤 리턴하므로 이 폴링이 사실상 0번 돌고 그냥
        # settle_sec(1초) 고정 대기 하나만 남아 - 구간 사이 정지가 육안으로 잘
        # 안 보이는 원인이었음 (실제로 겪은 문제). amovel/amovej로 이동만
        # "시작"시키고 바로 리턴하게 하면, 그 뒤의 폴링이 실제로 "물리적으로
        # 멈췄는지"를 확인하는 의미 있는 대기가 됨.
        #
        # do_safe_movel/do_safe_movej(safe_motion.safe_movel/safe_movej)를 쓰지
        # 않는 이유: 그 함수들은 이동 중에 emergency_stop 이벤트를 다시 확인해서
        # 걸려 있으면 즉시 EmergencyStopError를 던지는데, 지금은 이미 그 이벤트가
        # 걸려서 복구 중인 상황이라 이벤트가 아직 clear()되지 않은 상태임 -
        # 그대로 쓰면 복구 이동 자체가 시작하자마자 다시 예외를 던져 복구가 영원히
        # 끝나지 않음. 그래서 emergency_stop을 재확인하지 않는 순수 이동
        # (amovel/amovej + check_motion 폴링)만 씀.
        def _recovery_movel(pos, mod=DR_MV_MOD_ABS):
            # ref=DR_BASE(절대좌표계 기준) + mod=REL(상대이동)이므로 z축 부호는
            # 뒤집을 필요 없음 - "양수 lift_mm=상승"이 그대로 맞음. (한때 REL을
            # 툴좌표 기준으로 착각해서 -Z로 보내야 한다고 판단했었는데, 그건 오해였음:
            # 여기서는 DR_BASE 기준이라 +Z가 곧 위쪽임.)
            if mod == DR_MV_MOD_REL:
                # DSR_ROBOT2.get_normal_pos()는 posx/posj/ndarray 또는 list만
                # 받아들이고 일반 tuple은 "Invalid type : pos"로 거부함 - 그래서
                # 여기서도 반드시 posx로 다시 감싸야 함(실측으로 확인된 문제).
                x, y, z, rx, ry, rz = pos
                pos = posx(x, y, z, rx, ry, rz)
            ret = amovel(pos, ref=DR_BASE, mod=mod)
            # amovel()은 컨트롤러가 명령을 실제로 접수했는지(0=성공, 그 외=거부)를
            # 반환하는데 지금까지는 이 값을 그냥 버렸음 - lift 단계가 폴링상으로는
            # "정지 확인 완료"인데 실제 z는 전혀 안 바뀌는 사례가 실측으로 반복
            # 확인됐고(재시도해도 동일), 이게 check_motion() 타이밍 레이스가 아니라
            # 컨트롤러가 애초에 명령을 거부한 것인지 다음 발생 시 바로 알 수 있도록 남김.
            if ret != 0:
                node.get_logger().error(f"[E-STOP] amovel 명령이 거부됐습니다(ret={ret}): pos={pos}, mod={mod}")
            return ret

        def _recovery_movej(pos):
            ret = amovej(pos)
            if ret != 0:
                node.get_logger().error(f"[E-STOP] amovej 명령이 거부됐습니다(ret={ret}): pos={pos}")
            return ret

        return check_and_recover(
            node, status_bus, estop_tracker, emergency_stop,
            movel=_recovery_movel, movej=_recovery_movej, wait=wait,
            check_motion=check_motion, gripper_open=gripper_open,
            home_pos=HOME, camera_pos=CAMERA,
            get_current_pos=lambda: get_current_posx(DR_BASE)[0], posx_factory=posx,
            # comm_node를 넘기는 이유: stop_motion()과 동일하게 DSR 제어용 메인 node를
            # 여기서 또 spin하면 충돌 위험이 있음 (comm_node는 이미 별도 스레드에서 spin 중).
            check_hw_safety_stop=lambda: is_controller_in_hardware_safety_stop(comm_node),
            # 하드웨어 안전정지가 SAFE_STOP/SAFE_OFF/RECOVERY류(SetRobotControl.srv
            # 기준 서비스 호출만으로 STANDBY까지 복구 가능한 상태)일 때, 운영자가
            # 프론트엔드에서 재개(RESUME)를 눌러 현장 확인을 마치면 로봇 재연결/
            # 컨트롤러 재부팅 없이 바로 이어서 복구할 수 있도록 함
            # (estop_handler.check_and_recover의 reset_hw_safety_stop 인자 참고).
            reset_hw_safety_stop=lambda: reset_controller_safety_stop(comm_node),
            # resume_motion(move_resume)은 더 이상 안 씀 - stop_motion() 기본값이
            # DR_HOLD에서 DR_QSTOP으로 바뀌면서(safe_motion.STOP_MODE_DEFAULT 참고)
            # 애초에 "재개"가 필요한 일시정지 상태를 만들지 않음(auto_dump_robot_pkg와
            # 동일한 방식 - move_resume 없이도 정지 직후 movel/movej가 정상 동작함이
            # 그 프로젝트에서 실기로 검증돼 있음).
            describe_robot_state=describe_robot_state,
            wait_for_resume=wait_for_resume_signal,
        )

    # ------------------------------------------------------------------
    # 백엔드 연동: /decision/result 구독, /robot/command 구독
    # ------------------------------------------------------------------
    decision_queue: "queue.Queue[dict]" = queue.Queue()
    started = threading.Event()

    def decision_result_callback(msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            node.get_logger().error(f"{TOPIC_DECISION_RESULT} JSON 파싱 실패: {msg.data}")
            return
        decision_queue.put(data)

    def robot_command_callback(msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            node.get_logger().error(f"{TOPIC_ROBOT_COMMAND} JSON 파싱 실패: {msg.data}")
            return

        command = data.get("command")
        if command == "START":
            node.get_logger().info(f"[{TOPIC_ROBOT_COMMAND}] START 수신")
            started.set()
        elif command == "EMERGENCY_STOP":
            node.get_logger().error(f"[{TOPIC_ROBOT_COMMAND}] EMERGENCY_STOP 수신")
            emergency_stop.set()
        elif command == "RESUME":
            # E-STOP 물리 복구가 끝난 뒤 wait_for_resume_signal()에서 대기 중일 때만
            # 의미가 있음 - 그 외 상황에서 와도 이벤트만 걸어두는데, wait_for_resume_signal()이
            # 대기 시작 직전에 항상 clear부터 하므로 낡은(이전 건에 대한) RESUME이
            # 다음 E-STOP 게이트를 곧바로 통과시키는 일은 없음.
            node.get_logger().info(f"[{TOPIC_ROBOT_COMMAND}] RESUME 수신")
            resume_requested.set()
            # 프론트엔드 PAUSE 버튼이 두 번째 클릭에서 같은 RESUME 명령을 보내
            # 일시정지도 풀게 되어 있음 - MANUAL_PAUSE 중이 아니면 그냥 아무 효과
            # 없는 clear()라 안전함.
            pause_requested.clear()
        elif command == "MANUAL_PAUSE":
            node.get_logger().info(f"[{TOPIC_ROBOT_COMMAND}] MANUAL_PAUSE 수신 - 다음 사과 집기 전에 일시정지합니다.")
            pause_requested.set()
        elif command == "HOLD":
            # 백엔드(hitl_state_machine.py)가 HITL 질문을 시작할 때 표시용으로 보냄.
            # 별도 처리가 필요 없음 - decision_queue는 답변이 해석되기 전까지
            # 그 사과의 작업 자체를 받지 않으므로(hitl_state_machine.py가 답변
            # 전에는 /decision/result를 보내지 않음), 로봇은 이미 구조적으로
            # 다음 decision이 올 때까지 대기 상태다. 여기서 process_state를
            # READY 밖으로 바꾸면 오히려 백엔드의 "READY일 때만 새 Vision 판단"
            # 게이팅(main.py)이 막혀서, 무시(skip) 처리 후 다른 사과를 이어서
            # 인식해야 하는 흐름이 깨지므로 의도적으로 상태를 건드리지 않는다.
            node.get_logger().info(f"[{TOPIC_ROBOT_COMMAND}] HOLD 수신 (표시용, decision_queue 대기로 이미 반영됨)")
        else:
            node.get_logger().warn(f"[{TOPIC_ROBOT_COMMAND}] '{command}' 명령은 아직 처리하지 않습니다.")

    comm_node.create_subscription(String, TOPIC_DECISION_RESULT, decision_result_callback, 10)
    comm_node.create_subscription(String, TOPIC_ROBOT_COMMAND, robot_command_callback, 10)

    # movel/wait이 메인 스레드를 블록하는 동안에도 위 구독 콜백이 처리되도록
    # comm_node를 별도 스레드에서 spin (DSR 제어용 node는 건드리지 않음 - 위 주석 참고).
    #
    # executor를 명시적으로 만들어서 넘기는 이유: rclpy.spin()/rclpy.spin_until_future_complete()이
    # executor를 안 넘기면 둘 다 프로세스 전체에서 공유되는 rclpy 내부의 "글로벌 executor"
    # 싱글턴을 쓴다. DSR_ROBOT2.py의 movel/amovel/movej/amovej/check_motion 등은 내부적으로
    # rclpy.spin_until_future_complete(g_node, future)를 executor 없이 호출하므로 이 글로벌
    # executor를 메인 스레드에서 사용함. comm_node를 여기서도 executor 없이 spin하면 같은
    # 글로벌 executor 객체를 백그라운드 스레드가 동시에 돌리게 되어(서로 다른 node를 등록해도
    # executor 인스턴스 자체는 하나), 두 스레드가 executor 내부의 콜백 대기 제너레이터를 동시에
    # 건드리면서 "ValueError: generator already executing"로 크래시할 수 있다. 그래서 comm_node
    # 전용 executor를 따로 만들어 완전히 분리한다.
    comm_executor = SingleThreadedExecutor()
    comm_executor.add_node(comm_node)
    spin_thread = threading.Thread(target=comm_executor.spin, daemon=True)
    spin_thread.start()

    vision_client = node.create_client(SrvAppleStatus, VISION_SERVICE_NAME)
    while not vision_client.wait_for_service(timeout_sec=3.0):
        node.get_logger().info(f"Waiting for {VISION_SERVICE_NAME} service...")

    # vision_client와 달리 시작을 블록하지 않음: 이 서비스는 backend 프로세스
    # (basket_bridge.py)가 떠 있어야 응답 가능한데, 안 떠 있어도 로봇 사이클은
    # "빈 바스켓" 기본값으로 계속 동작해야 하기 때문 (is_basket_occupied 참고).
    basket_client = node.create_client(SrvBasketStatus, BASKET_SERVICE_NAME)

    # 로그 표시용 - DESTINATION_TO_BOX(위 주석)에 적힌 실제 물리 배치와 동일:
    # b1=가공용, b2=못난이, b3=정상, b4=폐기.
    BOX_DISPLAY_NAMES = {"b1": "가공용", "b2": "못난이", "b3": "정상", "b4": "폐기"}

    def is_basket_occupied(box_name: str) -> bool:
        """box_name(b1..b4)의 최신 점유 상태를 get_basket_status로 조회.

        서비스가 아직 준비 안 됐거나(backend 미기동) 응답이 valid=False(아직
        /basket_status를 한 번도 못 받은 상태)면 '비어있음'으로 간주한다 -
        기본 동작(기존과 동일한 빈 바스켓 로직)으로 안전하게 폴백하기 위함.

        조회에 성공하면(valid=True) box_name 하나만 보고 끝내지 않고, 세컨
        카메라가 이번에 확인한 b1~b4 전체 상태를 로그로 남긴다 - 지금 놓으려는
        박스뿐 아니라 다른 박스 상태도 운영 중에 눈으로 바로 확인할 수 있게 하기
        위함."""
        if not basket_client.service_is_ready():
            return False
        future = basket_client.call_async(SrvBasketStatus.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=1.0)
        response = future.result()
        if response is None or not response.valid:
            return False

        statuses = {
            name: getattr(response, f"{name}_occupied", False) for name in BOX_DISPLAY_NAMES
        }
        node.get_logger().info(
            "세컨 카메라 바스켓 점유 확인: " + ", ".join(
                f"{name}({BOX_DISPLAY_NAMES.get(name, name)})="
                f"{'사과 있음' if occ else '비어있음'}"
                for name, occ in statuses.items()
            )
        )
        return statuses[box_name]

    def _query_apple_status(condition):
        """
        get_apple_status를 한 번 호출해서 (status, pick_pos_or_None)을 반환하는
        공용 헬퍼. refresh_pick_pos(그립 실패 후 재시도 직전)와
        confirm_apple_present_before_pick(집기 시도 자체를 시작하기 전 사전
        재확인)이 공유해서 씀 - 둘 다 "CAMERA에 서 있는 상태에서 get_apple_status를
        다시 불러 최신 상태를 본다"는 동일한 동작이라 분리함.

        status는 response.status 그대로("empty"/"unknown"/실제 사과 분류) 반환.
        pick_pos는 position이 유효(비어있지 않고 전부 0이 아님)할 때만
        camera_to_base로 변환해서 채우고, 그 외(서비스 호출 실패/depth 측정 실패)는
        None - 호출부가 "이번 조회로는 위치를 못 얻었다"로 구분해서 처리해야 함.
        """
        future = vision_client.call_async(SrvAppleStatus.Request())
        rclpy.spin_until_future_complete(node, future)
        response = future.result()
        if response is None:
            node.get_logger().error(f"{VISION_SERVICE_NAME} 서비스 호출 실패")
            return None, None

        position = list(response.position or [])
        if not position or all(v == 0.0 for v in position):
            return response.status, None

        camera_pose_at_detection = get_current_posx()[0]
        depth_offset = _depth_offset_for_condition(condition)
        bx, by, bz = camera_to_base(position, camera_pose_at_detection, depth_offset_mm=depth_offset)
        node.get_logger().info(
            f"Vision 재조회: status={response.status} camera={position} -> "
            f"base=({bx:.2f}, {by:.2f}, {bz:.2f})"
        )
        return response.status, posx(bx, by, bz, *DEFAULT_PICK_ORIENTATION)

    def refresh_pick_pos(condition):
        """
        그립 실패 후 재시도 직전에 호출. CAMERA 위치에 서 있는 상태에서
        get_apple_status를 다시 호출해 최신 위치로 pick_pos를 재계산함
        (사과가 그립 실패 중 살짝 밀렸을 수 있음). 실패(서비스 오류/depth 측정
        실패)하면 None을 반환하고, 호출부는 이전 pick_pos로 그대로 재시도함.
        """
        _status, pick_pos = _query_apple_status(condition)
        if pick_pos is None:
            node.get_logger().warn("위치 재획득 실패 - 서비스 호출 실패 또는 depth 측정 실패(position이 [0,0,0])")
        return pick_pos

    def confirm_apple_present_before_pick(condition):
        """
        decision을 실제로 처리하기 시작하기 직전(아직 pick_pos로 출발하지 않고
        CAMERA에 그대로 서 있는 상태)에 get_apple_status를 한 번 더 호출해서,
        백엔드가 다수결로 판정을 확정한 시점과 로봇이 실제로 도착하는 시점 사이의
        시간차 동안 사과가 이미 없어졌는지를 가볍게 재확인한다.

        실측으로 확인된 문제: 이 재확인 없이 decision.pose를 그대로 믿고 바로
        내려가면, 실제로는 이미 없는 경우(다른 사이클에 먼저 집혔거나 밀려남/
        vision 오탐)에도 그리퍼를 완전히 닫아보고 반발력이 안 걸릴 때까지 가봐야만
        (수 초~십수 초) 실패를 알 수 있어서, 매 헛손질마다 이동+파지+재시도+복귀
        비용이 통째로 낭비됐음. CAMERA에 이미 서 있는 상태라 위치 이동 없이 1초
        내외만 더 쓰면 되므로 비용 대비 이득이 큼.

        Returns:
            (True, pick_pos_or_None): 사과가 여전히 있는 것으로 보이거나(status가
                "empty"가 아님) 서비스 오류/depth 실패로 판단 자체가 안 되는 경우 -
                두 경우 다 "일단 집기를 시도해야 한다"는 뜻이라 True. pick_pos가
                None이 아니면 방금 새로 확인한 위치로 갱신해서 쓰고, None이면
                호출부가 기존 decision.pose 기반 pick_pos를 그대로 써야 함.
            (False, None): status가 명시적으로 "empty"(카메라에 물체 자체가 안
                잡힘) - 사과가 이미 없어진 게 거의 확실하므로, 호출부는 집기
                시도 자체를 건너뛰고 다음 decision으로 넘어가야 함.
        """
        status, pick_pos = _query_apple_status(condition)
        if status == "empty":
            node.get_logger().warn(
                "사전 재확인 결과 트레이에 아무것도 안 잡힘(status=empty) - 이미 "
                "없어진 것으로 판단해 이번 사과는 집기 시도 없이 건너뜁니다."
            )
            return False, None
        return True, pick_pos

    set_velx(45, 30)
    set_accx(45, 30)
    set_velj(30)
    set_accj(30)

    def place_apple_back_on_worktable(pick_pos):
        """
        박스 접촉 감지(force_controlled_place)가 실패해서 사과를 박스에 놓지
        못했을 때의 예외 처리. 그립을 계속 유지한 채로 복귀시키는 대신, 원래
        집었던 자리(pick_pos, 이미 트레이 범위 안임이 검증된 좌표)로 돌아가
        작업대(트레이)에 다시 내려놓는다.

        pick_apple()과 동일하게 pick_pos가 트레이 벽 근처면 대각선 경유점을
        먼저 거치고(compute_wall_aware_approach), box_pos에 놓을 때와 동일하게
        force_controlled_place로 접촉(트레이 바닥/다른 사과)을 감지하며 부드럽게
        내려간다. 접촉 감지 자체가 또 실패하더라도(예: 트레이가 예상과 다르게
        비어있어 MAX_DOWN_DISTANCE 안에 아무것도 안 닿는 경우) 사과를 계속 쥔 채로
        둘 수는 없으므로 그리퍼는 무조건 연다 - 이 함수의 목적 자체가 "무슨 일이
        있어도 그립을 풀어서 다음 사이클을 진행할 수 있게 하는 것"이기 때문.
        """
        node.get_logger().info(f"작업대(원래 집었던 자리)에 사과 재배치: pick_pos={pick_pos}")

        hover_pos, tilted_pick_pos = compute_wall_aware_approach(pick_pos, posx)
        if hover_pos is not None:
            node.get_logger().info(
                f"재배치 위치도 트레이 벽 근처로 판단되어 대각선 경유점을 먼저 거칩니다: {hover_pos}"
            )
            do_safe_movel(hover_pos)

        px, py, pz, prx, pry, prz = tilted_pick_pos
        force_start_pos = posx(px, py, pz + WORKTABLE_FORCE_START_CLEARANCE_MM, prx, pry, prz)
        do_safe_movel(force_start_pos)
        wait(0.3)

        contact_ok = force_controlled_place(
            node, force_start_pos,
            emergency_stop_event=emergency_stop, stop_node=comm_node,
            check_hw_safety_stop=check_hw_safety_stop,
        )
        if contact_ok:
            node.get_logger().info("작업대 접촉 확인 - 그리퍼를 엽니다.")
        else:
            node.get_logger().warn(
                "작업대 재배치 중에도 접촉 감지 실패 - 그래도 사과를 계속 쥐고 있을 "
                "수는 없으므로 그리퍼를 엽니다."
            )

        gripper_opened = gripper_open()
        status_bus.publish_gripper_status(not gripper_opened)
        if not gripper_opened:
            node.get_logger().error("작업대 재배치: 그리퍼 오픈 실패 - 그립을 유지한 채로 복귀합니다.")
            status_bus.publish_safety("ERR_DROP", "작업대 재배치 그리퍼 오픈 실패")
        else:
            estop_tracker.mark_placed()

        wait(1.0)

        retreat_pos = posx(px, py, pz + 50, prx, pry, prz)
        do_safe_movel(retreat_pos)
        wait(0.2)

    def place_apple_in_box(name, box_pos, way_pos, pick_pos):
        """
        이미 집어서 그립까지 확인해둔 사과를 목적지 박스(고정 절대좌표 box_pos)에
        놓는 사이클 (집기 자체는 여기서 하지 않음).
        홈(안전 경유) -> 웨이포인트 -> box_pos로 이동 -> 힘제어 하강 ->
        그리퍼 오픈 -> 퇴피 -> 웨이포인트 -> 카메라 복귀.
        박스 안에 이미 뭔가 있는지는 get_basket_status로 미리 조회해서, 있으면
        힘제어 시작 높이(clearance)를 더 높게 잡는다(PLACE_FORCE_START_CLEARANCE_MM_OCCUPIED) -
        어차피 힘제어(force_controlled_place)가 접촉을 감지하며 내려가므로 이미
        사과가 있어도 접촉 시점에 멈추긴 하지만, 시작 높이가 실제 내용물에 맞게
        조정되어야 접촉 감지 여유가 안정적으로 확보됨. 조회 실패/backend 미기동
        시에는 항상 "비어있음"으로 간주해 기존과 동일하게 동작한다
        (is_basket_occupied 참고).

        접촉 감지 자체가 실패한 경우(박스에 못 놓은 경우)는 그립을 유지한 채로
        복귀하지 않고, place_apple_back_on_worktable(pick_pos)로 원래 집었던
        작업대 자리에 다시 내려놓은 뒤 카메라로 복귀해서 다음 사이클을 재개한다
        (호출부 pick_pos 필요 - box_sequence_test.py 메인 루프의 pick_pos를 그대로 전달).
        """
        node.get_logger().info(f"--- Sorting to {name} ---")

        node.get_logger().info("Move to HOME before heading to box (안전 경유)")
        do_safe_movej(HOME)
        wait(0.3)

        # 경유점(way_pos)으로 출발하기 전에 미리 점유 상태를 확인해둔다 - 이후
        # way_pos -> force_start_pos 접근 속도(occupied면 감속)와 힘제어 시작
        # 높이(clearance)를 정하는 데 필요한 값이라, 도착 후가 아니라 출발 전에
        # 알고 있어야 그 다음 이동부터 바로 반영할 수 있다.
        occupied = is_basket_occupied(name)
        clearance = (
            PLACE_FORCE_START_CLEARANCE_MM_OCCUPIED if occupied else PLACE_FORCE_START_CLEARANCE_MM
        )
        node.get_logger().info(
            f"{name} 바스켓 상태: {'사과 있음' if occupied else '비어있음(기본)'} "
            f"(힘제어 시작 높이: box_pos + {clearance}mm"
            + (f", 접근 속도 저감: vel={OCCUPIED_APPROACH_VEL} acc={OCCUPIED_APPROACH_ACC})" if occupied else ")")
        )

        node.get_logger().info(f"Move to way point before {name}")
        do_safe_movel(way_pos)
        wait(0.3)

        box_x, box_y, box_z, box_rx, box_ry, box_rz = box_pos
        force_start_pos = posx(
            box_x, box_y, box_z + clearance, box_rx, box_ry, box_rz
        )
        # occupied면 clearance(30mm)가 실측이 아닌 추정 오프셋이라, 실제 더미가
        # 더 높을 때 force_controlled_place(컴플라이언스)가 시작되기 전인 이
        # rigid movel 구간에서 충돌할 수 있음 - 접근 속도를 낮춰서 대응 여유를 둠
        # (OCCUPIED_APPROACH_VEL/ACC 주석 참고). 비어있으면 기존과 동일하게
        # 기본 순항 속도(vel/acc=None -> set_velx/set_accx 전역값)로 접근.
        do_safe_movel(
            force_start_pos,
            vel=OCCUPIED_APPROACH_VEL if occupied else None,
            acc=OCCUPIED_APPROACH_ACC if occupied else None,
        )
        wait(0.3)
        contact_ok = force_controlled_place(
            node, force_start_pos,
            emergency_stop_event=emergency_stop, stop_node=comm_node,
            check_hw_safety_stop=check_hw_safety_stop,
            timeout_sec=OCCUPIED_PLACE_TIMEOUT_SEC if occupied else BASE_PLACE_TIMEOUT_SEC,
        )

        status_bus.set_motion("PLACING", name)

        if contact_ok:
            node.get_logger().info(f"Apple contact confirmed at {name}. Opening gripper.")
            gripper_opened = gripper_open()
            status_bus.publish_gripper_status(not gripper_opened)

            if not gripper_opened:
                node.get_logger().error(
                    f"{name}: 그리퍼 오픈 실패 - 그립을 유지한 채로 복귀합니다."
                )
                status_bus.publish_safety("ERR_DROP", f"{name} 그리퍼 오픈 실패")
            else:
                # 실제로 박스에 내려놨으므로 이제 잡고 있지 않은 상태로 리셋
                # (그리퍼 오픈이 실패했으면 여전히 잡고 있을 수 있으므로 리셋하지 않음)
                estop_tracker.mark_placed()

            wait(1.0)

            retreat_x, retreat_y, retreat_z, rx, ry, rz = box_pos
            retreat_pos = posx(retreat_x, retreat_y, retreat_z + 50, rx, ry, rz)
            do_safe_movel(retreat_pos)
            wait(0.2)

            node.get_logger().info(f"Return to way point after {name}")
            do_safe_movel(way_pos)
            wait(0.3)

            node.get_logger().info("Return to CAMERA position")
            do_safe_movel(CAMERA)
            wait(0.3)
        else:
            # 박스 접촉 실패 - 그립 유지한 채로 복귀하지 않고, 원래 집었던
            # 작업대 자리(pick_pos)에 다시 내려놓아 다음 사이클을 진행할 수 있게 함
            # (place_apple_back_on_worktable 참고).
            node.get_logger().error(
                f"{name}에서 접촉 실패 - 박스에 놓지 못해 작업대(원래 집었던 자리)에 "
                "다시 내려놓습니다."
            )
            status_bus.publish_safety("ERR_DROP", f"{name} 접촉 실패 - 작업대에 재배치")

            node.get_logger().info(f"Return to way point after {name}")
            do_safe_movel(way_pos)
            wait(0.3)

            node.get_logger().info("Return to CAMERA position (작업대 재배치 전 경유)")
            do_safe_movel(CAMERA)
            wait(0.3)

            place_apple_back_on_worktable(pick_pos)

            node.get_logger().info("Return to CAMERA position (작업대 재배치 후)")
            do_safe_movel(CAMERA)
            wait(0.3)

        return contact_ok

    node.get_logger().info("Opening gripper before starting")
    if not gripper_open():
        node.get_logger().error("시작 시 그리퍼 오픈 실패 - 하드웨어 연결 상태를 확인하세요.")

    status_bus.set_state("READY")
    node.get_logger().info("=== Box sequence test: /robot/command의 START 대기 중 ===")
    started.wait()

    node.get_logger().info("=== Box sequence test start ===")
    status_bus.set_state("MOVING")

    # HOME은 사이클 전체에서 맨 처음(여기)과 맨 마지막(루프를 빠져나간 뒤)에만 거침.
    # 그 사이에는 CAMERA <-> WAY <-> BOX 사이만 순환함.
    #
    # 이 최초 HOME->CAMERA 구간은 메인 while 루프 진입 전이라 루프 안쪽의
    # try/except EmergencyStopError(아래 참고)로는 보호되지 않음 - 실측으로 확인된
    # 문제: 이 구간에서 EMERGENCY_STOP이 오면 EmergencyStopError가 그대로 전파돼
    # 복구 없이 프로세스 자체가 죽어버림. 그래서 여기도 동일하게 감싸야 함.
    try:
        node.get_logger().info("Move to HOME (최초 1회)")
        do_safe_movej(HOME)
        wait(0.3)

        node.get_logger().info("Move to CAMERA position")
        do_safe_movel(CAMERA)
        wait(0.3)
    except EmergencyStopError:
        recover_from_estop_if_needed()

    # CAMERA 위치에 도착 + 그리퍼가 비어있는 이 시점부터가 백엔드 입장에서
    # "지금 보이는 게 진짜 새 사과"라고 신뢰할 수 있는 유일한 구간임.
    # /robot/process_state="READY"를 여기서 명시적으로 발행해야, 백엔드의
    # _vla_consumer_loop가 이 시점에 들어온 Vision 감지만 판단(자동 실행/사람에게
    # 질문)하고, 그 외 구간(집기/이동/내려놓기 중)에 카메라가 우연히 잡는 잔상은
    # 무시하도록 게이팅할 수 있음.
    status_bus.set_state("READY")

    # PAUSED -> READY로 되돌아왔을 때 딱 한 번만 로그/상태 전환하기 위한 플래그
    # (아래 while 루프에서 pause_requested가 풀렸는지 확인할 때 씀).
    was_paused = False

    while rclpy.ok():
        # 사과 사이사이(대기 중)에는 이동/힘제어 폴링 루프 자체가 없어서, 그
        # 안에 있는 hw_safety_watch도 전혀 안 돌고 있음 - 그래서 소프트웨어
        # /robot/command 없이 펜던트 물리 비상정지 버튼만 눌린 경우를 여기서
        # 따로 확인해야 함. decision_queue.get(timeout=1.0)이 어차피 매
        # 반복마다 최대 1초씩 대기하므로, 이 확인도 자연스럽게 초당 1회
        # 정도로만 나가 과부하가 없음.
        if not emergency_stop.is_set():
            hw_blocked, hw_state_name = is_controller_in_hardware_safety_stop(comm_node)
            if hw_blocked:
                node.get_logger().error(
                    f"대기 중 컨트롤러 하드웨어 안전정지 감지({hw_state_name}) - "
                    "EMERGENCY_STOP 처리로 전환합니다."
                )
                emergency_stop.set()

        # 사과 사이사이(대기 중)에 비상정지가 걸린 경우 - 지금 진행 중인 이동이
        # 없어서 아래 try/except(이동 도중 감지용)로는 못 잡으므로 별도로 확인.
        if recover_from_estop_if_needed():
            continue

        # 프론트엔드 "PAUSE ROBOT MOTION" 버튼(MANUAL_PAUSE) - 이미 진행 중인
        # 집기/이동 사이클(아래 decision 처리 블록)은 안전하게 끝까지 마치고,
        # 다음 사과를 새로 시작하기 전인 이 지점(decision_queue.get() 이전)에서만
        # 대기시킴. RESUME 명령이 오면 robot_command_callback이 pause_requested를
        # clear()하므로 자동으로 풀림.
        if pause_requested.is_set():
            was_paused = True
            status_bus.set_state("PAUSED")
            wait(0.5)
            continue
        elif was_paused:
            was_paused = False
            node.get_logger().info("[PAUSE] 재개(RESUME) 신호를 받아 계속 진행합니다.")
            status_bus.set_state("READY")

        try:
            decision = decision_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        fruit = decision.get("fruit")
        destination = decision.get("destination")
        pose = decision.get("pose")
        condition = decision.get("condition")

        if destination not in DESTINATION_TO_BOX:
            node.get_logger().error(f"알 수 없는 destination: {destination} - 이 사과는 건너뜁니다.")
            continue

        box_name, box_pos, way_pos = DESTINATION_TO_BOX[destination]

        # 작은 사과/미확인 과일은 몸통이 낮거나 크기를 몰라서 일반 오프셋만큼
        # 내려가면 트레이/받침대와 충돌할 수 있으므로 condition별 전용 오프셋을 씀.
        depth_offset = _depth_offset_for_condition(condition)

        if not pose:
            node.get_logger().error(
                f"decision.pose가 없어({fruit}) 위치를 알 수 없으므로 이 사과는 건너뜁니다."
            )
            status_bus.publish_safety("ERR_PICK", f"{fruit} pose 없음")
            continue

        # pose는 카메라 좌표계 -> 로봇이 현재 서 있는(=CAMERA에서 감지된
        # 순간의) pose를 기준으로 로봇 베이스 좌표계로 변환해야 movel에 그대로
        # 쓸 수 있음. 자세한 설명은 vision_transform.py 상단 docstring 참고.
        camera_pose_at_detection = get_current_posx()[0]
        bx, by, bz = camera_to_base(pose, camera_pose_at_detection, depth_offset_mm=depth_offset)
        node.get_logger().info(
            f"Vision 좌표 변환: camera={pose} -> base=({bx:.2f}, {by:.2f}, {bz:.2f})"
        )
        pick_pos = posx(bx, by, bz, *DEFAULT_PICK_ORIENTATION)

        # 실측으로 확인된 문제: vision이 트레이 작업 영역을 한참 벗어난 위치를
        # 줄 때가 있는데(오탐/depth 오차), wall_avoidance.py의 벽-근접 판정이
        # "경계를 벗어남"과 "벽에 딱 붙어있음"을 구분 못 해서 그 위치로 그대로
        # 접근을 시도하다 파지가 반복 실패했음. 그래서 트레이 범위(wall_avoidance.
        # TRAY_X/Y_MIN/MAX_MM) 밖이면 애초에 집기 시도 자체를 하지 않고 건너뜀.
        px, py, _pz, _prx, _pry, _prz = pick_pos
        if not is_within_tray_bounds(px, py):
            node.get_logger().error(
                f"pick 좌표(x={px:.1f}, y={py:.1f})가 트레이 작업 영역을 벗어나서 "
                "이 사과는 건너뜁니다 (vision 오탐/depth 오차 가능성)."
            )
            status_bus.publish_safety("ERR_PICK", f"{fruit} pick 좌표가 트레이 범위 밖")
            continue

        # 백엔드가 다수결로 이 decision을 확정한 시점과 로봇이 실제로 집기를
        # 시작하는 지금 이 시점 사이에 시간차가 있어(다른 decision 처리/이동 등),
        # 그 사이 사과가 이미 없어졌을 수 있음. 아직 CAMERA에 서 있는 지금
        # get_apple_status를 한 번 더 불러 가볍게 재확인함 - confirm_apple_
        # present_before_pick 참고. 여기서 걸러지면 이동+파지+재시도까지 가는
        # 훨씬 비싼 헛손질을 아예 안 하게 됨.
        still_present, confirmed_pos = confirm_apple_present_before_pick(condition)
        if not still_present:
            status_bus.publish_safety("ERR_PICK", f"{fruit} 사전 재확인 결과 이미 없어짐")
            continue
        if confirmed_pos is not None:
            cpx, cpy, _cpz, _cprx, _cpry, _cprz = confirmed_pos
            if is_within_tray_bounds(cpx, cpy):
                pick_pos = confirmed_pos
            else:
                node.get_logger().warn(
                    f"사전 재확인 위치(x={cpx:.1f}, y={cpy:.1f})가 트레이 범위 밖이라 "
                    "무시하고 기존 decision 위치로 진행합니다."
                )

        node.get_logger().info(f"--- 사과: fruit={fruit} destination={destination} -> {box_name} ---")

        try:
            # 1) 사과 집기. 그립 실패(힘 감지로 사과를 놓친 것으로 판단) 시
            # pick_pos 하나로 계속 헛손질하지 않도록, CAMERA로 돌아가 위치를
            # 다시 받아온 뒤 MAX_PICK_ATTEMPTS번까지 재시도함.
            picked_ok = False
            for attempt in range(1, MAX_PICK_ATTEMPTS + 1):
                status_bus.set_state("MOVING")
                status_bus.set_motion("PICKING", fruit)
                # 파지 성공/실패가 판단되기 전(pick_apple 내부의 한 감지 구간)에
                # E-STOP이 걸려도 "잡았다"고 간주하도록, 집기 시도 전 원위치를
                # 미리 기록해둠 (estop_handler.py 참고).
                estop_tracker.begin_pick_attempt(pick_pos)
                picked_ok = pick_apple(
                    node, pick_pos, do_safe_movel,
                    emergency_stop_event=emergency_stop, stop_node=comm_node,
                    check_hw_safety_stop=check_hw_safety_stop,
                )
                status_bus.publish_gripper_status(picked_ok)
                if picked_ok:
                    break

                estop_tracker.mark_pick_failed()
                status_bus.publish_safety(
                    "ERR_PICK", f"{fruit} 파지(힘 감지) 실패 ({attempt}/{MAX_PICK_ATTEMPTS})"
                )
                if attempt == MAX_PICK_ATTEMPTS:
                    break

                node.get_logger().warn(
                    f"그립 실패(사과를 놓친 것으로 판단) - CAMERA로 돌아가 위치를 "
                    f"다시 받아서 재시도합니다. ({attempt}/{MAX_PICK_ATTEMPTS})"
                )
                # 다음 시도 전에 그리퍼를 확실히 열어둠 (실패한 그립이 반쯤 닫힌
                # 채로 남아있으면 다음 파지 힘 감지 기준선이 틀어질 수 있음).
                gripper_open()
                do_safe_movel(CAMERA)
                wait(0.3)
                refreshed_pos = refresh_pick_pos(condition)
                if refreshed_pos is not None:
                    rx, ry, _rz, _rrx, _rry, _rrz = refreshed_pos
                    if is_within_tray_bounds(rx, ry):
                        pick_pos = refreshed_pos
                    else:
                        node.get_logger().error(
                            f"재획득한 pick 좌표(x={rx:.1f}, y={ry:.1f})가 트레이 범위 밖이라 "
                            "무시하고 이전 좌표로 재시도합니다."
                        )
                # 재획득 실패(또는 범위 밖) 시 이전 pick_pos로 그대로 재시도함
            wait(0.3)

            if not picked_ok:
                node.get_logger().error(
                    f"{MAX_PICK_ATTEMPTS}번 시도해도 그립에 실패해 이번 사과는 포기합니다."
                )
                node.get_logger().info("Move to CAMERA position (집기 포기 후 복귀)")
                do_safe_movel(CAMERA)
                wait(0.3)
                status_bus.set_state("READY")
                continue

            # 2) 집은 직후 별도 퇴피 없이 바로 CAMERA 위치로 복귀
            node.get_logger().info("Move to CAMERA position (집은 직후 바로 복귀)")
            do_safe_movel(CAMERA)
            wait(0.3)

            # 3) 목적지 박스로 분류해서 놓음 (경유점에서 박스 점유 확인 포함)
            place_apple_in_box(box_name, box_pos, way_pos, pick_pos)

            # CAMERA 위치 도착 + 그리퍼는 이미 위에서 열림 -> 다음 사과를 볼 준비가
            # 된 상태이므로 READY로 되돌림 (여기서 안 되돌리면 백엔드가 "카메라
            # 위치에서 대기 중"과 "이동 중"을 구분 못하게 됨).
            status_bus.set_state("READY")

        except EmergencyStopError:
            # do_safe_movel/do_safe_movej/force_controlled_place 중 이동 도중 터졌으면
            # 여기 한 곳에서만 잡으면 됨 - 로봇은 이미 safe_motion.py가 물리적으로
            # 멈춰둔 상태이고, 여기서는 "이동으로 복구할지"만 결정/실행함.
            recover_from_estop_if_needed()
            continue

    node.get_logger().info("=== Box sequence test done ===")

    node.get_logger().info("Move to HOME (종료)")
    try:
        do_safe_movej(HOME)
        wait(0.3)
    except EmergencyStopError:
        # 종료 구간이라 recover_from_estop_if_needed()가 굳이 CAMERA까지 다시
        # 보낼 필요는 없지만, 그래도 정지 명령 자체는 이미 나갔으므로(safe_motion.py)
        # 로봇이 안전하게 멈춘 상태 - 프로세스가 죽지 않도록 잡아만 둠.
        node.get_logger().error("종료 HOME 이동 중 EMERGENCY_STOP 감지 - 그대로 종료합니다.")

    comm_executor.shutdown()
    comm_node.destroy_node()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
