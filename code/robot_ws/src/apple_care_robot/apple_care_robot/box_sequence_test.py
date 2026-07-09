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

박스에 이미 사과가 있는지는 더 이상 가정(하드코딩)하지 않음. 카메라가 그리퍼에
고정된 eye-in-hand 구조라(vision_transform.py 참고) 박스로 가는 길을 어차피
거치는 웨이포인트에서는 그 카메라로 박스 안이 보임 - 이 점을 이용해서 박스별로
따로 조사하러 접근하지 않고, 웨이포인트에 도착한 김에 get_apple_status를 한 번 더
호출해 이미 뭔가(사과) 있는지 확인함(detect_box_occupancy 참고). "empty"가 아니면
(사과든, 확신치 않은 "unknown"이든) 안전하게 "이미 뭔가 있음"으로 보고 그 다음부터
조심 접근(hover 위치까지 진입 + 낮춘 힘제어 임계값)을, "empty"면 기존처럼 바로
접근함.

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
from apple_care_msgs.srv import SrvAppleStatus
from apple_care_robot.openclose import gripper_open
from apple_care_robot.pick_helpers import (
    pick_apple, _depth_offset_for_condition, DEFAULT_PICK_ORIENTATION,
)
from apple_care_robot.vision_transform import camera_to_base
from apple_care_robot.status_bus import StatusBus
from apple_care_robot.estop_handler import EstopRecoveryTracker, check_and_recover
from apple_care_robot.safe_motion import (
    safe_movel, safe_movej, EmergencyStopError, is_controller_in_hardware_safety_stop,
    resume_motion,
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
    )
    from apple_care_robot.force_place import (
        force_controlled_place,
        CAREFUL_APPROACH_VEL, CAREFUL_APPROACH_ACC,
        EXISTING_APPLE_HOVER_CLEARANCE, EXISTING_APPLE_FORCE_THRESHOLD,
    )

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
    # b1=가공용, b2=못난이, b3=정상, b4=폐기. 박스에 이미 사과가 있는지는 더 이상
    # 여기서 가정하지 않고, place_apple_in_box가 매번 detect_box_occupancy()로
    # 카메라를 통해 직접 확인함.
    DESTINATION_TO_BOX = {
        "processing_box": ("b1", B1, WAY1),
        "ugly_box":       ("b2", B2, WAY1),
        "normal_box":     ("b3", B3, WAY1),
        "discard_box":    ("b4", B4, WAY2),
    }

    emergency_stop = threading.Event()
    estop_tracker = EstopRecoveryTracker()

    # 이동/힘제어 폴링 루프 안에서 공용으로 쓸 하드웨어 안전정지 조회 콜백.
    # comm_node를 넘기는 이유: stop_motion()과 동일하게 DSR 제어용 메인 node를
    # 여기서 또 spin하면 충돌 위험이 있음 (comm_node는 이미 별도 스레드에서 spin 중).
    def check_hw_safety_stop():
        return is_controller_in_hardware_safety_stop(comm_node)

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
        def _recovery_movel(pos):
            ret = amovel(pos, ref=DR_BASE)
            # amovel()은 컨트롤러가 명령을 실제로 접수했는지(0=성공, 그 외=거부)를
            # 반환하는데 지금까지는 이 값을 그냥 버렸음 - lift 단계가 폴링상으로는
            # "정지 확인 완료"인데 실제 z는 전혀 안 바뀌는 사례가 실측으로 반복
            # 확인됐고(재시도해도 동일), 이게 check_motion() 타이밍 레이스가 아니라
            # 컨트롤러가 애초에 명령을 거부한 것인지 다음 발생 시 바로 알 수 있도록 남김.
            if ret != 0:
                node.get_logger().error(f"[E-STOP] amovel 명령이 거부됐습니다(ret={ret}): pos={pos}")
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
            # E-STOP 시 stop_motion()이 move_stop(HOLD)로 걸어둔 일시정지 상태를
            # 복구 이동 전에 풀어줌 - 실측으로 확인된 문제(안 풀면 movel이 성공
            # 응답을 받고도 실제로는 안 움직임) 참고: safe_motion.resume_motion.
            resume_motion=lambda: resume_motion(comm_node),
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
        else:
            # HOLD/RESUME/MANUAL_PAUSE 등 사이클 도중 정지/재개는 이번 연동 범위 밖.
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

    def detect_box_occupancy():
        """
        (로봇이 지금 박스로 가는 길의 웨이포인트에 서있어서 그리퍼에 고정된
        카메라가 박스 안을 보고 있다고 가정) get_apple_status를 한 번 호출해서
        박스 안에 이미 뭔가(사과) 있는지를 True/False로 판단한다.
        박스 전용으로 따로 조사하러 접근하지 않고, 어차피 지나가는 웨이포인트에서
        바로 확인함. status가 정확히 뭔지는 신경 쓰지 않고 "empty"인지 아닌지만
        봄 - "unknown"도 박스 안에 뭔가 있다는 뜻이므로 안전하게 "있음"으로
        취급해서 조심 접근을 유도한다 (애매하다고 "없음"으로 취급하면 blind
        접근하다 부딪힐 수 있음). 서비스 호출 자체가 실패하면 마찬가지로
        안전한 쪽(있음)으로 취급한다.
        """
        future = vision_client.call_async(SrvAppleStatus.Request())
        rclpy.spin_until_future_complete(node, future)
        response = future.result()

        if response is None:
            node.get_logger().warn(
                f"{VISION_SERVICE_NAME} 서비스 호출 실패 - 박스 상태를 확인할 수 없어 "
                "안전하게 '이미 사과가 있음'으로 간주합니다."
            )
            return True

        has_apple = response.status != "empty"
        node.get_logger().info(
            f"박스 점검: status={response.status} -> "
            f"{'이미 사과 있음(조심 접근)' if has_apple else '비어 있음(기존 접근)'}"
        )
        return has_apple

    set_velx(45, 30)
    set_accx(45, 30)
    set_velj(30)
    set_accj(30)

    def place_apple_in_box(name, box_pos, way_pos):
        """
        이미 집어서 그립까지 확인해둔 사과를 목적지 박스에 놓는 사이클
        (집기 자체는 여기서 하지 않음).
        홈(안전 경유) -> 웨이포인트(도착한 김에 카메라로 박스 안 확인) ->
        [사과 있으면: box_pos보다 높은 위치로 조심히 접근 후 힘제어 하강 |
        없으면: box_pos까지 바로 내려가서 기존 방식대로 힘제어 하강] ->
        그리퍼 오픈 -> 퇴피 -> 웨이포인트 -> 카메라 복귀
        (박스가 비어있어도 항상 힘제어로 하강함 - 고속/생략 없음)
        """
        node.get_logger().info(f"--- Sorting to {name} ---")

        node.get_logger().info("Move to HOME before heading to box (안전 경유)")
        do_safe_movej(HOME)
        wait(0.3)

        node.get_logger().info(f"Move to way point before {name}")
        do_safe_movel(way_pos)
        wait(0.3)

        node.get_logger().info(f"{name}: 경유점에서 박스 상태를 확인합니다.")
        has_existing_apple = detect_box_occupancy()

        if has_existing_apple:
            hx, hy, hz, hrx, hry, hrz = box_pos
            safe_hover_pos = posx(hx, hy, hz + EXISTING_APPLE_HOVER_CLEARANCE, hrx, hry, hrz)
            node.get_logger().info(
                f"{name}: 박스 안에 사과가 있음을 확인 - 원래 hover보다 "
                f"{EXISTING_APPLE_HOVER_CLEARANCE}mm 위에서부터 조심히 접근합니다."
            )
            do_safe_movel(safe_hover_pos, vel=CAREFUL_APPROACH_VEL, acc=CAREFUL_APPROACH_ACC)
            wait(0.3)
            contact_ok = force_controlled_place(
                node, safe_hover_pos, force_threshold=EXISTING_APPLE_FORCE_THRESHOLD,
                emergency_stop_event=emergency_stop, stop_node=comm_node,
                check_hw_safety_stop=check_hw_safety_stop,
            )
        else:
            node.get_logger().info(f"{name}: 박스가 비어 있음을 확인 - 기존 접근 방법대로 진행합니다.")
            do_safe_movel(box_pos)
            wait(0.3)
            contact_ok = force_controlled_place(
                node, box_pos,
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

            retreat_x, retreat_y, retreat_z, rx, ry, rz = box_pos
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
    node.get_logger().info("Move to HOME (최초 1회)")
    do_safe_movej(HOME)
    wait(0.3)

    node.get_logger().info("Move to CAMERA position")
    do_safe_movel(CAMERA)
    wait(0.3)

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
                    pick_pos = refreshed_pos
                # 재획득 실패 시 이전 pick_pos로 그대로 재시도함
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
    do_safe_movej(HOME)
    wait(0.3)

    comm_executor.shutdown()
    comm_node.destroy_node()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
