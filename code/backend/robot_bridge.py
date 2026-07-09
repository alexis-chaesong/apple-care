# HOLD/RESUME 커맨드 타입만 추가

"""
robot_bridge.py
================
백그라운드 스레드에서 ROS2 노드를 spin하고, 토픽이 들어올 때마다
asyncio.Queue에 데이터를 넣는 역할만 전담

기존 코드와의 차이점:
  - 기존: _status_callback에서 asyncio.run_coroutine_threadsafe로
    WebSocket.send_json()을 직접 호출 (ROS가 WebSocket을 알고 있었음)
  - 변경: _status_callback은 call_soon_threadsafe로 메인 이벤트 루프의
    asyncio.Queue에 데이터를 넣기만 함. WebSocket은 전혀 모름.

  - 기존: self.executor.spin() (블로킹, 종료 신호를 줄 수 없음)
  - 변경: spin_once(timeout_sec=...)를 반복하며 stop_event를 체크
    → shutdown()을 안전하게 구현 가능

active_connections 리스트와 add_connection/remove_connection은
connection_manager.py로 이동했으므로 이 파일에서는 제거했음 

이 코드는 "ROS 2의 스레드 환경"에서 들어오는 실시간 데이터를 "FastAPI의 비동기(AsyncIO) 루프"로 
안전하게 던져주는 스레드 안전(Thread-safe) 브리지 역할을 완벽하게 수행하고 있음
"""
import base64
import json
import logging
import threading
import time
from asyncio import AbstractEventLoop, Queue, QueueFull

import cv2
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool  # 실제 로봇 상태 토픽 메시지 타입에 맞게 조정 가능
from cv_bridge import CvBridge

logger = logging.getLogger(__name__)

ROS_NODE_NAME = "fastapi_robot_bridge"

# 현민님의 StatusBus가 발행하는 3개 토픽
TOPIC_PROCESS_STATE = "/robot/process_state"
TOPIC_MOTION_STATUS = "/robot/motion_status"
TOPIC_SAFETY_EVENT = "/robot/safety_event"
# [백업 대비 추가] motion_controller가 발행하는 긴급정지 복구 체크포인트 토픽
# 기존 공정 상태 문자열만으로는 음식물 배출 완료 전/후를 정확히 판별할 수 없어 별도 토픽으로 분리
TOPIC_RECOVERY_STAGE = "/robot/recovery_stage"
TOPIC_COMMAND = "/robot/command"
TOPIC_GRIPPER_STATUS = "/gripper/status"
# obj_detection 노드가 publish하는 YOLO 인식결과 오버레이 이미지
TOPIC_DEBUG_IMAGE = "/obj_detection/debug_image"

# JPEG 인코딩 품질 (0~100). 웹소켓으로 계속 흘려보내야 하므로 화질보다 전송 크기를 우선함
DEBUG_IMAGE_JPEG_QUALITY = 70

# HMI 화면에 실제로 그려지는 크기(hmi_app/view/dashboard.py의 cam_box_w/h)보다
# 원본 해상도가 훨씬 커서(예: 1280x720), 화면에서 어차피 축소해서 보여줄 걸
# 원본 그대로 보내면 대역폭 낭비임. 인코딩 전에 이 너비 기준으로 먼저
# 축소해서 전송량 자체를 줄인다 (비율은 유지, 세로는 자동 계산).
DEBUG_IMAGE_MAX_WIDTH = 640

# 로봇팀의 motion_planner_node.py가 실제로 구독하는 토픽 (/robot/command와는
# 완전히 다른 스키마). publish_command()와 별개로 publish_decision_result()가
# 전용으로 사용함 (2단계 조사 결과: motion_planner_node.py의 decision_callback()이
# 현재 어떤 키도 검증하지 않는 완전 pass-through라서, fruit/destination/pose
# 키 이름은 그쪽 docstring/주석에 적힌 미래 구현 계획에 맞춰둠)
TOPIC_DECISION_RESULT = "/decision/result"

SPIN_TIMEOUT_SEC = 0.1

# 상단에 상태 매핑 테이블 추가 (msg가 없을 때를 위한 fallback)
STATE_KO_MAP = {
    "IDLE": "대기 중 (사용자 이용 전)",
    "INIT": "시스템 초기화 중",
    "READY": "준비 완료",
    "MOVING": "구동 중 (이동 처리 중)",
    "DUMPING": "음식물 배출 처리 중",
    "WASHING": "세척 처리 중",
    "COMPLETE": "작업 완료",
    "PAUSED": "일시 중단",
    "COLLISION": "충돌 위험 감지",
    "ERROR": "오류 발생",
}

def _split_colon(raw: str) -> tuple[str, str]:
    """
    "state" 또는 "state:message" 형태의 문자열을
    (state, message) 튜플로 안전하게 분리.

    현민님 코드의 publish 규칙:
        text = state.value if not msg else f"{state.value}:{msg}"
    즉 msg가 없으면 콜론이 아예 없을 수 있음 -> split(":", 1)로 1회만 분리하고
    두 번째 값이 없으면 빈 문자열로 채움.
    """
    if ":" in raw:
        head, _, tail = raw.partition(":")
        return head, tail
    return raw, ""


def _resize_for_transport(frame, max_width: int):
    """가로 max_width 기준으로 비율 유지 축소 (이미 그보다 작으면 그대로 반환)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


class RobotBridgeManager:
    """ROS2 Node + Executor + Thread 생명주기를 관리하는 매니저."""

    def __init__(self) -> None:
        self.node: Node | None = None
        self.executor: SingleThreadedExecutor | None = None
        self.ros_thread: threading.Thread | None = None
        self.loop: AbstractEventLoop | None = None
        self.queue: Queue | None = None
        self._stop_event = threading.Event()
        # ★ 종료 제어의 핵심. 스레드 간에 "이제 그만 멈춰!"라는 신호를 안전하게 주고받기 위한 플래그(Flag) 객체
        self.command_pub = None  # publish_command에서 사용할 Publisher (start_bridge에서 생성)
        self.decision_result_pub = None  # publish_decision_result에서 사용할 Publisher (start_bridge에서 생성)

        # 그리퍼가 지금 사과를 쥐고 있는지(True) 여부. main.py의 _vla_consumer_loop가
        # 이 값을 보고, 로봇이 사과를 집어서 박스로 옮기는 중(그리퍼 닫힘)에는 그 사이
        # 카메라에 잡히는 감지 결과(그리퍼/사과 잔상, 손 등)를 실제 새 사과로 오인해
        # ask_human을 띄우지 않도록 걸러냄. apple_sorting_cycle.py의 상태 문자열은
        # 사이클 시작 후 계속 "MOVING"으로만 남아있어(READY로 복귀 안 함) "카메라 위치에서
        # 대기 중"과 "박스로 이동 중"을 구분 못하지만, 그리퍼 개폐 여부는 정확히
        # "집어서 이동중"과 일치하는 신호라 이걸로 게이팅함.
        self.gripper_grasped: bool = False

        # /robot/process_state의 최신 원시(raw) 상태 문자열(예: "READY", "MOVING").
        # apple_sorting_cycle.py가 CAMERA 위치에 도착해 다음 사과를 볼 준비가 됐을
        # 때만 "READY"를 발행하도록 수정해뒀으므로, 이 값이 정확히 "READY"일 때만
        # 카메라가 지금 보고 있는 게 신뢰할 수 있는 새 사과라는 뜻. main.py의
        # _vla_consumer_loop가 이 값과 gripper_grasped를 함께 확인해서, 그 외
        # 구간(집기/이동/내려놓기 중)에 잡히는 unknown bbox 등은 전부 무시함.
        self.latest_process_state: str = "IDLE"

    # ------------------------------------------------------------------
    def start_bridge(self, fastapi_loop: AbstractEventLoop, output_queue: Queue) -> None:
        """
        FastAPI lifespan에서 호출되어 백그라운드 스레드로 ROS2를 구동함

        Args:
            fastapi_loop: uvicorn이 돌고 있는 메인 이벤트 루프.
                          ROS 콜백 스레드에서 이 루프로 작업을 위임하기 위해 필요.
            output_queue: ROS에서 수신한 데이터가 쌓이는 큐.
                          이 큐를 누가 소비하는지(WebSocket인지)는 여기서 알 필요 없음.

        main.py의 lifespan 시점에 호출되어, 메인 비동기 루프(fastapi_loop)와 데이터 바구니(output_queue)를 인자로 받아옴 
        """
        if not rclpy.ok():
            rclpy.init(args=None) # ROS 2 통신 컨텍스트를 초기화

        self.node = Node(ROS_NODE_NAME)
        # Node(ROS_NODE_NAME): 백엔드가 ROS 네트워크에서 인식될 노드를 생성하고, 메인 루프와 큐를 클래스 내부에 저장
        self.loop = fastapi_loop
        self.queue = output_queue

        # 1. 현민님 StatusBus 규격에 맞춘 3개 토픽 구독
        self.process_state_sub = self.node.create_subscription(
            String,
            TOPIC_PROCESS_STATE,
            self._process_state_callback,
            10,
        )
        self.motion_status_sub = self.node.create_subscription(
            String,
            TOPIC_MOTION_STATUS,
            self._motion_status_callback,
            10,
        )
        self.safety_event_sub = self.node.create_subscription(
            String,
            TOPIC_SAFETY_EVENT,
            self._safety_event_callback,
            10,
        )
        # [백업 대비 추가] RecoveryStage 전용 구독자를 등록
        # START 작업 중 기록된 마지막 안전 체크포인트를 FastAPI 큐로 전달해,
        # HMI가 긴급정지 후 HOME 복귀 또는 남은 공정 재개를 정확히 선택할 수 있게 함
        self.recovery_stage_sub = self.node.create_subscription(
            String,
            TOPIC_RECOVERY_STAGE,
            self._recovery_stage_callback,
            10,
        )

        self.gripper_status_sub = self.node.create_subscription(
            Bool,
            TOPIC_GRIPPER_STATUS,
            self._gripper_status_callback,
            10,
        )

        # obj_detection의 디버그 이미지(인식결과 오버레이) 구독 -> HMI 실시간 모니터링용
        self.cv_bridge = CvBridge()
        self.debug_image_sub = self.node.create_subscription(
            Image,
            TOPIC_DEBUG_IMAGE,
            self._debug_image_callback,
            1,  # 이미지는 크므로 큐 깊이를 얕게 둬서 항상 최신 프레임만 유지
        )

        #  명령 하달용 Publisher 생성
        self.command_pub = self.node.create_publisher(String, TOPIC_COMMAND, 10)

        # 로봇팀 motion_planner_node.py용 Publisher (/decision/result)
        self.decision_result_pub = self.node.create_publisher(String, TOPIC_DECISION_RESULT, 10)

        # 2. 기존과 동일하게 타이머도 유지 (헬스체크/디버깅 용도)
        # self.timer = self.node.create_timer(1.0, self.timer_callback)

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        # Executor 등록: 노드의 이벤트를 처리해 줄 실행기(Executor)에 방금 만든 노드를 등록

        print("Nodes:", self.executor.get_nodes())

        self._stop_event.clear()
        self.ros_thread = threading.Thread(
            target=self._spin_loop, name="ros2-spin-thread", daemon=True
        )
        self.ros_thread.start()
        # 독립 스레드 분리: FastAPI 메인 스레드가 ROS 2 통신 때문에 멈추면(Blocking) 안 되므로, 
        # 별도의 백그라운드 스레드(ros2-spin-thread)를 파서 ROS 루프(_spin_loop)를 실행시킴 
        # daemon=True로 지정하여 메인 프로세스가 종료되면 함께 강제 종료되도록 안전장치를 둠 

    def shutdown(self) -> None:
        """spin 루프를 안전하게 멈추고 ROS 리소스를 정리합니다."""
        print("RobotBridgeManager shutdown 시작...", flush=True)

        self._stop_event.set() # 1) 스레드 루프 종료 신호

        if self.ros_thread is not None:
            self.ros_thread.join(timeout=5.0) # 2) 스레드가 죽을 때까지 최대 5초 대기
            if self.ros_thread.is_alive():
                logger.warning("spin 스레드가 timeout 내에 종료되지 않았습니다.")

        if self.executor is not None and self.node is not None:
            self.executor.remove_node(self.node)
        if self.node is not None:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

        # 노드를 파괴하고 ROS 2 컨테이너 시스템을 깨끗하게 정리(rclpy.shutdown())

        print("RobotBridgeManager shutdown 완료.", flush=True)

    # ------------------------------------------------------------------
    def _spin_loop(self) -> None:
        """
        별도 스레드에서 실행. executor.spin() 대신 spin_once를 반복하여
        _stop_event를 주기적으로 체크할 수 있게 함

        기존의 executor.spin()은 무한 대기에 빠져 외부에서 멈출 방법이 없었음
        변경된 구조에서는 spin_once(timeout_sec=0.1)를 사용하여 0.1초 동안만 이벤트를 체크하고, 
        즉시 빠져나와 _stop_event가 켜졌는지 확인함 덕분에 서버 종료 요청에 즉각적으로 반응하는 유연한 서브루프가 완성되었음!
        """
        assert self.executor is not None
        print("ROS2 spin 루프 시작.", flush=True)

        while not self._stop_event.is_set():
            try:
                self.executor.spin_once(timeout_sec=SPIN_TIMEOUT_SEC)
            except rclpy.executors.ExternalShutdownException:
                # Ctrl+C(SIGINT) 등으로 rclpy.shutdown()이 외부에서 호출된 경우.
                # 이건 정상적인 종료 신호이므로 ERROR로 찍지 않고 즉시 루프를 빠져나감 
                print("ExternalShutdownException 감지, spin 루프 종료.", flush=True)
                self._stop_event.set()
                break
            except Exception:  # noqa: BLE001
                logger.error("spin_once 루프에서 예외 발생", exc_info=True)
                time.sleep(0.5)

        print("ROS2 spin 루프 종료.", flush=True)

    # def timer_callback(self) -> None:
        # print("Timer works!", flush=True)

    # ------------------------------------------------------------------
    # 추가상행 파이프라인 (FastAPI -> ROS) 진입점
    # ------------------------------------------------------------------
    def publish_command(
        self,
        command_type: str,
        task_id: str | None = None,
        mode_id: int | None = None,
        payload: dict | None = None,
    ) -> None:
        
        """
        FastAPI 요청 스레드(uvicorn worker)에서 직접 호출됨.

        주의: rclpy의 Publisher.publish()는 ROS 2 내부적으로 스레드 세이프하게
        설계되어 있어(rmw 레이어가 락을 관리함), call_soon_threadsafe 없이
        바로 호출해도 안전함. 하행(Queue) 파이프라인과는 방향이 반대이므로
        별도의 동기화 장치가 필요 없음 — 단, command_pub이 아직 초기화되지
        않은 경우(start_bridge 호출 전)를 방어적으로 체크함.
        """
        if self.command_pub is None or self.node is None:
            logger.error(
                "publish_command 호출 실패: ROS Bridge가 아직 시작되지 않음. (command_type=%s)",
                command_type,
            )
            return

        body = {
            "command": command_type,
            "task_id": task_id,
            "mode_id": mode_id,
            "timestamp": time.time(),
        }
        if payload is not None:
            body["payload"] = payload
            
        msg = String(data=json.dumps(body, ensure_ascii=False))

        self.command_pub.publish(msg)
        print(f"[ROS2 /robot/command 발행]: {msg.data}", flush=True)

    def publish_decision_result(
        self,
        fruit: str,
        destination: str,
        pose: list[float] | None,
        **extra_fields,
    ) -> None:
        """
        motion_planner_node.py가 구독하는 /decision/result 토픽으로 발행.
        publish_command()와는 완전히 다른 스키마(command/task_id/mode_id/payload 구조가
        아니라 fruit/destination/pose 평면 구조)이므로 별도 메서드로 분리함.

        motion_planner_node.decision_callback()은 현재 어떤 키도 검증하지 않는
        완전 pass-through라서(2단계 재확인 결과), pose=None이어도 그대로 통과되며
        크래시하지 않음. 다만 그쪽 plan_motion_order()가 나중에 실제로 구현되면
        pose를 실제로 써야 하므로, Vision이 좌표를 못 준 경우에도 굳이 값을
        지어내지 않고 None을 그대로 흘려보냄.
        """
        if self.decision_result_pub is None or self.node is None:
            logger.error(
                "publish_decision_result 호출 실패: ROS Bridge가 아직 시작되지 않음. (fruit=%s)",
                fruit,
            )
            return

        body = {"fruit": fruit, "destination": destination, "pose": pose}
        body.update(extra_fields)

        msg = String(data=json.dumps(body, ensure_ascii=False))

        self.decision_result_pub.publish(msg)
        print(f"[ROS2 /decision/result 발행]: {msg.data}", flush=True)

    # ------------------------------------------------------------------
    # ROS 콜백 3종 (모두 ROS 스레드에서 실행됨)
    #
    # 절대 이 안에서 WebSocket이나 connection_manager를 직접 호출하지 않고,
    # call_soon_threadsafe로 메인 이벤트 루프의 Queue에만 데이터를 넘김.
    #
    # 문제의 원인이었던 부분 완벽 해결: 이 함수는 FastAPI가 아닌 ROS 2 스레드
    # 위에서 실행됨. 여기서 AsyncIO 영역인 웹소켓을 직접 건드리면
    # 스레드가 꼬여서 터졌던 것.
    # ------------------------------------------------------------------
    
    def _process_state_callback(self, msg: String) -> None:
        """
        /robot/process_state 수신.
        StatusBus.set_state()가 publish하는 토픽 (motion_status와 동일 포맷).
        "state" 또는 "state:message" 형태.
        """
        print(f"[ROS2 /robot/process_state 수신]: {msg.data}", flush=True)

        state, message = _split_colon(msg.data)
        previous_state = self.latest_process_state
        self.latest_process_state = state

        if previous_state == "ERROR" and state == "READY":
            # E-STOP 복구가 막 끝나고 처음 READY로 돌아온 시점. estop_handler.py의
            # RETURN_TO_ORIGIN 복구는 사과를 "원래 집었던 바로 그 위치"에 그대로
            # 다시 내려놓는데, vision_bridge.py의 디바운스(_is_duplicate)는
            # status+position이 직전과 같으면 "이미 처리한 물체"로 보고 재감지
            # 자체를 건너뜀 - 그래서 로봇이 READY로 복귀해도 새 decision이 영원히
            # 안 와서 카메라 위치에 멈춰있는 것처럼 보이는 문제가 있었음(실제로
            # 겪은 문제). 복구 직후에는 반드시 디바운스를 강제로 초기화해서
            # 그 사과가 다시 평가되게 함.
            from vision_bridge import vision_bridge_manager
            vision_bridge_manager.force_redetect()
            logger.info("E-STOP 복구 감지(ERROR->READY) - Vision 디바운스 상태 초기화")

        status_text = message if message else STATE_KO_MAP.get(state, state)

        payload = {
            "type": "PROCESS_STATE",
            "payload": status_text,   # GUI가 기대하는 순수 문자열
            "timestamp": time.time(),
        }
        self._dispatch(payload)

    def _motion_status_callback(self, msg: String) -> None:
        """
        /robot/motion_status 수신.
        StatusBus.set_state()가 동시에 publish하는 토픽.
        "state" 또는 "state:message" 형태.
        """
        print(f"[ROS2 /robot/motion_status 수신]: {msg.data}", flush=True)

        motion, message = _split_colon(msg.data)
        payload = {
            "type": "MOTION_STATUS",
            "motion": motion,
            "message": message,
            "timestamp": time.time(),
        }
        self._dispatch(payload)

    def _safety_event_callback(self, msg: String) -> None:
        """
        /robot/safety_event 수신.
        StatusBus.publish_safety()가 publish하는 토픽.
        "code:msg" 형태 (항상 콜론 포함).
        """
        print(f"[ROS2 /robot/safety_event 수신]: {msg.data}", flush=True)

        error_code, error_msg = _split_colon(msg.data)
        payload = {
            "type": "SAFETY_EVENT",
            "error_code": error_code,
            "error_msg": error_msg,
            "timestamp": time.time(),
        }
        self._dispatch(payload)

    def _gripper_status_callback(self, msg) -> None:
        print(f"[ROS2 /gripper/status 수신]: {msg.data}", flush=True)
        self.gripper_grasped = bool(msg.data)
        payload = {
            "type": "GRIPPER_STATUS",
            "grasped": msg.data,
            "timestamp": time.time(),
        }
        self._dispatch(payload)

    def _debug_image_callback(self, msg: Image) -> None:
        """
        obj_detection의 인식결과 오버레이 이미지를 받아 JPEG로 압축 후
        base64 문자열로 인코딩해서 HMI로 흘려보냄.
        raw Image(bgr8)를 그대로 JSON에 실으면 프레임당 수백 KB~1MB라
        웹소켓/큐가 감당 못하므로 반드시 JPEG로 압축한 뒤 전달함.
        """
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = _resize_for_transport(frame, DEBUG_IMAGE_MAX_WIDTH)
            ok, jpeg = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, DEBUG_IMAGE_JPEG_QUALITY]
            )
            if not ok:
                logger.warning("디버그 이미지 JPEG 인코딩 실패")
                return
            b64_image = base64.b64encode(jpeg.tobytes()).decode('ascii')
        except Exception:  # noqa: BLE001
            logger.error("디버그 이미지 처리 중 예외 발생", exc_info=True)
            return

        self._dispatch({
            "type": "CAMERA_FRAME",
            "image": b64_image,
            "timestamp": time.time(),
        })

    # [백업 대비 추가] ROS 문자열 체크포인트를 백엔드 공통 이벤트 형식으로 변환
    # ROS 콜백에서 웹소켓을 직접 호출하지 않고 _dispatch를 사용하므로,
    # ROS 스레드와 FastAPI AsyncIO 이벤트 루프 사이의 스레드 안전성을 유지
    def _recovery_stage_callback(self, msg: String) -> None:
        print(f"[ROS2 /robot/recovery_stage 수신]: {msg.data}", flush=True)
        self._dispatch({
            "type": "RECOVERY_STAGE",
            "stage": msg.data,
            "timestamp": time.time(),
        })

    # ------------------------------------------------------------------

    def _dispatch(self, payload: dict) -> None:
        """ROS 스레드 -> 메인 이벤트 루프로 payload 전달 예약 (공통 헬퍼)."""
        if self.loop is not None and self.queue is not None:
            self.loop.call_soon_threadsafe(self._enqueue, payload)

    def _enqueue(self, payload: dict) -> None:
        """메인 이벤트 루프 안에서 실행됨 (call_soon_threadsafe에 의해 스케줄됨)."""
        try:
            self.queue.put_nowait(payload)
        except QueueFull:
            # 큐가 가득 찼으면 가장 오래된 데이터를 버리고 최신 데이터를 우선함
            try:
                self.queue.get_nowait() # 큐가 꽉 찼으면 가장 오래된 데이터 하나를 빼서 버림
            except Exception:  # noqa: BLE001
                pass
            try:
                self.queue.put_nowait(payload) # 그리고 최신 데이터를 욱여넣음, 비동기 큐에 데이터를 대기 없이 즉시 집어넣음 
            except QueueFull:
                logger.warning("Queue가 여전히 가득 차 있어 메시지를 버립니다.")

            # 링 버퍼(Ring Buffer) 모방 안전장치: 
            # 만약 웹소켓 전송이 너무 느려져서 큐가 가득 차면(QueueFull), 
            # 가장 오래된 밀린 데이터 하나를 과감히 버리고(get_nowait) 그 자리에 최신 로봇 상태 좌표를 넣음
            # 로봇 제어 및 모니터링에서는 지나간 과거 데이터보다 현재 시점의 최신 상태가 훨씬 중요하기 때문에, 시스템 먹통을 막는 매우 훌륭하고 실전적인 예외 처리 기법

# 어디서나 싱글톤처럼 접근 가능하도록 전역 인스턴스 생성 (기존과 동일한 패턴 유지)
bridge_manager = RobotBridgeManager()