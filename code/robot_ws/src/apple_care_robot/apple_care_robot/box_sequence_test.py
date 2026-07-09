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
from apple_care_robot.wall_avoidance import is_within_tray_bounds
from apple_care_robot.status_bus import StatusBus
from apple_care_robot.estop_handler import EstopRecoveryTracker, check_and_recover
from apple_care_robot.safe_motion import (
    safe_movel, safe_movej, EmergencyStopError, is_controller_in_hardware_safety_stop,
    get_controller_robot_state, ROBOT_STATE_NAMES,
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

# 세컨 카메라(basket_camera 노드) -> 백엔드(basket_bridge.py)가 캐싱한 바스켓
# 점유 상태를 배치 직전에 조회하는 서비스. VISION_SERVICE_NAME과 동일한 이유로
# 반드시 절대 이름이어야 함.
BASKET_SERVICE_NAME = "/get_basket_status"

# 세컨 카메라/백엔드가 아직 안 켜져 있거나 서비스 호출이 실패해도 로봇 사이클이
# 멈추지 않도록, 응답을 못 받으면 이 시간(초) 안에 포기하고 기본값(비어있음)으로
# 넘어간다.
BASKET_SERVICE_TIMEOUT_SEC = 1.0

# 바스켓에 이미 사과가 있다고 판단됐을 때, 그 위에 그대로 얹지 않도록 옆으로
# 살짝 옮겨서 놓는 오프셋(mm, box 로컬 x축 기준). 바스켓 실측 크기/배치가 아직
# 확정되지 않았으므로 placeholder 값 - 실제 하드웨어에서 겹침/충돌 없이 들어가는
# 값으로 재조정해야 함.
BOX_OCCUPIED_OFFSET_X_MM = 40.0

# decision.pose가 None으로 온 경우(비전이 좌표를 못 준 경우)에 쓰는 임시 pick 좌표.
FALLBACK_PICK_POS_XYZ = (492.0, 13.48, 23.39)

# 그립 자체가 실패하는 경우(사과를 놓침 - grasp_apple_with_force_feedback이
# picked_ok=False)를 대비한 재시도 횟수. 매번 CAMERA로 돌아가서 위치를 다시
# 받아온 뒤 집기를 재시도함.
MAX_PICK_ATTEMPTS = 3


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
    from apple_care_robot.force_place import force_controlled_place

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
        else:
            # HOLD/MANUAL_PAUSE 등 사이클 도중 정지/재개는 이번 연동 범위 밖.
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

    # get_apple_status와 달리 여기서는 서비스가 뜰 때까지 블로킹 대기하지 않음 -
    # 세컨 카메라/백엔드는 아직 선택 구성 요소라, 안 켜져 있어도 로봇 사이클이
    # "기본값(비어있음)"으로 정상 진행되어야 하기 때문 (check_basket_occupied 참고).
    basket_client = node.create_client(SrvBasketStatus, BASKET_SERVICE_NAME)

    def check_basket_occupied(box_name: str) -> bool:
        """
        배치 직전에 백엔드(basket_bridge.py)가 캐싱해둔 바스켓 점유 상태를 조회.

        서비스가 아직 안 떠 있거나(basket_camera/backend 미기동), 응답이 없거나,
        아직 한 번도 유효한 판정을 못 받았으면(valid=False) 항상 False(비어있음)로
        폴백한다 - 기본은 "없는 걸로 실행"하고, 확실히 "있다"고 확인된 경우에만
        오프셋 배치 분기를 탄다.
        """
        if not basket_client.service_is_ready():
            node.get_logger().warn(
                f"{BASKET_SERVICE_NAME} 서비스 미기동 - {box_name} 점유 여부는 "
                "기본값(비어있음)으로 처리합니다."
            )
            return False

        future = basket_client.call_async(SrvBasketStatus.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=BASKET_SERVICE_TIMEOUT_SEC)
        response = future.result()
        if response is None or not response.valid:
            node.get_logger().warn(
                f"{BASKET_SERVICE_NAME} 응답 없음/무효 - {box_name} 점유 여부는 "
                "기본값(비어있음)으로 처리합니다."
            )
            return False

        occupied = {
            "b1": response.b1_occupied,
            "b2": response.b2_occupied,
            "b3": response.b3_occupied,
            "b4": response.b4_occupied,
        }.get(box_name, False)
        node.get_logger().info(f"{box_name} 점유 상태 조회 결과: {'있음' if occupied else '없음'}")
        return occupied

    def refresh_pick_pos(condition):
        """
        그립 실패 후 재시도 직전에 호출. CAMERA 위치에 서 있는 상태에서
        get_apple_status를 다시 호출해 최신 위치로 pick_pos를 재계산함
        (사과가 그립 실패 중 살짝 밀렸을 수 있음). 실패(서비스 오류/depth 측정
        실패)하면 None을 반환하고, 호출부는 이전 pick_pos로 그대로 재시도함.
        """
        future = vision_client.call_async(SrvAppleStatus.Request())
        rclpy.spin_until_future_complete(node, future)
        response = future.result()
        if response is None:
            node.get_logger().error(f"{VISION_SERVICE_NAME} 서비스 호출 실패 (위치 재획득)")
            return None

        position = list(response.position or [])
        if not position or all(v == 0.0 for v in position):
            node.get_logger().warn("위치 재획득 실패 - depth 측정 실패(position이 [0,0,0])")
            return None

        camera_pose_at_detection = get_current_posx()[0]
        depth_offset = _depth_offset_for_condition(condition)
        bx, by, bz = camera_to_base(position, camera_pose_at_detection, depth_offset_mm=depth_offset)
        node.get_logger().info(
            f"위치 재획득: camera={position} -> base=({bx:.2f}, {by:.2f}, {bz:.2f})"
        )
        return posx(bx, by, bz, *DEFAULT_PICK_ORIENTATION)

    set_velx(45, 30)
    set_accx(45, 30)
    set_velj(30)
    set_accj(30)

    def place_apple_in_box(name, box_pos, way_pos):
        """
        이미 집어서 그립까지 확인해둔 사과를 목적지 박스(고정 절대좌표 box_pos)에
        놓는 사이클 (집기 자체는 여기서 하지 않음).
        홈(안전 경유) -> 웨이포인트 -> (바스켓 점유 조회) -> box_pos(또는 오프셋
        위치)로 이동 -> 힘제어 하강 -> 그리퍼 오픈 -> 퇴피 -> 웨이포인트 -> 카메라 복귀.
        세컨 카메라(basket_camera)/백엔드(basket_bridge.py)가 이 바스켓에 이미
        사과가 있다고 판단하면 옆으로 오프셋을 준 위치(target_pos)로 배치하고,
        기본(없음/서비스 미기동)이면 기존과 동일하게 box_pos 그대로 사용한다.
        어느 쪽이든 force_controlled_place로 접촉을 감지하며 내려가므로, 오프셋이
        정확하지 않아도 접촉 시점에 멈춰 안전함.
        """
        node.get_logger().info(f"--- Sorting to {name} ---")

        node.get_logger().info("Move to HOME before heading to box (안전 경유)")
        do_safe_movej(HOME)
        wait(0.3)

        node.get_logger().info(f"Move to way point before {name}")
        do_safe_movel(way_pos)
        wait(0.3)

        # 세컨 카메라(basket_camera)/백엔드(basket_bridge.py)가 판단한 이 바스켓의
        # 사과 유무를 조회해서, 이미 사과가 있으면 같은 자리에 그대로 얹지 않고
        # 옆으로 오프셋을 준 위치에 놓는다. 기본(서비스 미기동/무효 응답 포함)은
        # "없음"으로 간주해 기존과 동일하게 box_pos 그대로 사용.
        occupied = check_basket_occupied(name)
        if occupied:
            bx, by, bz, brx, bry, brz = box_pos
            target_pos = posx(bx + BOX_OCCUPIED_OFFSET_X_MM, by, bz, brx, bry, brz)
            node.get_logger().info(
                f"{name} 바스켓에 이미 사과가 있는 것으로 판단 - "
                f"오프셋({BOX_OCCUPIED_OFFSET_X_MM}mm) 배치로 전환합니다."
            )
            status_bus.publish_safety("BASKET_OCCUPIED", f"{name} 이미 사과 있음 - 오프셋 배치")
        else:
            target_pos = box_pos

        node.get_logger().info(f"Move to {name}")
        do_safe_movel(target_pos)
        wait(0.3)
        contact_ok = force_controlled_place(
            node, target_pos,
            emergency_stop_event=emergency_stop, stop_node=comm_node,
            check_hw_safety_stop=check_hw_safety_stop,
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

            retreat_x, retreat_y, retreat_z, rx, ry, rz = target_pos
            retreat_pos = posx(retreat_x, retreat_y, retreat_z + 50, rx, ry, rz)
            do_safe_movel(retreat_pos)
            wait(0.2)
        else:
            node.get_logger().error(
                f"{name}에서 접촉 실패 - 그립 유지한 채로 복귀. "
                f"이 사과는 놓지 못한 상태로 다음 단계 진행됨."
            )
            status_bus.publish_safety("ERR_DROP", f"{name} 접촉 실패")

        node.get_logger().info(f"Return to way point after {name}")
        do_safe_movel(way_pos)
        wait(0.3)

        node.get_logger().info("Return to CAMERA position")
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

        if pose:
            # pose는 카메라 좌표계 -> 로봇이 현재 서 있는(=CAMERA에서 감지된
            # 순간의) pose를 기준으로 로봇 베이스 좌표계로 변환해야 movel에 그대로
            # 쓸 수 있음. 자세한 설명은 vision_transform.py 상단 docstring 참고.
            camera_pose_at_detection = get_current_posx()[0]
            bx, by, bz = camera_to_base(pose, camera_pose_at_detection, depth_offset_mm=depth_offset)
            node.get_logger().info(
                f"Vision 좌표 변환: camera={pose} -> base=({bx:.2f}, {by:.2f}, {bz:.2f})"
            )
            pick_pos = posx(bx, by, bz, *DEFAULT_PICK_ORIENTATION)
        else:
            node.get_logger().warn("decision.pose가 없어 임시 pick 좌표를 사용합니다.")
            pick_pos = posx(*FALLBACK_PICK_POS_XYZ, *DEFAULT_PICK_ORIENTATION)

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
            place_apple_in_box(box_name, box_pos, way_pos)

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
