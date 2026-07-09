"""
basket_bridge.py
=================
obj_detection의 basket_camera 노드가 발행하는 `/basket_status` 토픽(B1~B4
사과 유무)을 구독해서 최신 상태를 캐싱하고, 로봇(box_sequence_test.py)이
배치 직전에 조회할 수 있도록 같은 노드에서 `get_basket_status` ROS 서비스도
호스팅한다.

robot_bridge.py / vision_bridge.py와 동일한 원칙:
  - ROS2 콜백 안에서는 무거운 작업을 하지 않고 상태 캐싱 + Queue 적재 예약만 함.
  - ROS2 스레드(SingleThreadedExecutor)와 FastAPI asyncio 이벤트 루프 사이를
    call_soon_threadsafe()로 안전하게 넘나듦.
  - 전역 싱글톤 패턴(basket_bridge_manager)으로 관리, main.py의 lifespan에서 start/shutdown.

vision_bridge.py와의 차이: vision은 요청-응답 서비스(get_apple_status)만 있어서
폴링 클라이언트로 구현했지만, basket_camera는 토픽을 직접 publish하므로 여기는
"구독 콜백 + 캐시"로 구현하고, 로봇이 부르는 get_basket_status는 그 캐시를
그대로 돌려주기만 하는 서비스 서버다 (새로 계산하지 않음).
"""

import base64
import json
import logging
import threading
import time
from asyncio import AbstractEventLoop, Queue, QueueFull
from typing import Optional

import cv2
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

logger = logging.getLogger(__name__)

try:
    from apple_care_msgs.srv import SrvBasketStatus
except ModuleNotFoundError:
    # vision_ws가 이 프로세스 환경에 build/source 되어 있지 않은 경우.
    # vision_bridge.py와 동일하게, 이 브릿지만 비활성화하고 나머지 백엔드는
    # 정상 동작해야 하므로 여기서 서버 전체를 죽이지 않고 경고만 남김.
    SrvBasketStatus = None
    logger.warning(
        "apple_care_msgs(SrvBasketStatus)를 찾을 수 없어 Basket Bridge를 "
        "비활성화합니다. vision_ws에서 'colcon build --packages-select "
        "apple_care_msgs' 후 'source install/setup.bash'를 실행한 뒤 서버를 "
        "재시작하세요."
    )

ROS_NODE_NAME = "fastapi_basket_bridge"
TOPIC_BASKET_STATUS = "/basket_status"
# basket_camera.py가 publish하는 바스켓 bbox + 사과 유무 오버레이 디버그 이미지
# (obj_detection/debug_image를 robot_bridge.py가 CAMERA_FRAME으로 중계하는 것과
# 동일한 패턴 - 여기서는 HMI의 "Cam 2" 자리로 보낼 BASKET_CAMERA_FRAME으로 중계).
TOPIC_DEBUG_IMAGE = "/basket_camera/debug_image"
SERVICE_NAME = "get_basket_status"
SPIN_TIMEOUT_SEC = 0.1

# 웹소켓으로 계속 흘려보내야 하므로 화질보다 전송 크기를 우선함 (robot_bridge.py와 동일)
DEBUG_IMAGE_JPEG_QUALITY = 70

# HMI 화면의 Cam 2(바스켓) 표시 영역보다 원본 해상도가 훨씬 커서, 인코딩 전에
# 이 너비 기준으로 먼저 축소해 전송량을 줄인다 (robot_bridge.py와 동일 패턴).
DEBUG_IMAGE_MAX_WIDTH = 640

BASKET_NAMES = ["b1", "b2", "b3", "b4"]


def _resize_for_transport(frame, max_width: int):
    """가로 max_width 기준으로 비율 유지 축소 (이미 그보다 작으면 그대로 반환)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


class BasketBridgeManager:
    """ROS2 Node(토픽 구독 + 서비스 서버) + Executor + Thread 생명주기를 관리."""

    def __init__(self) -> None:
        self.node: Optional[Node] = None
        self.executor: Optional[SingleThreadedExecutor] = None
        self.ros_thread: Optional[threading.Thread] = None
        self.loop: Optional[AbstractEventLoop] = None
        self.queue: Optional[Queue] = None
        self._stop_event = threading.Event()

        # /basket_status에서 마지막으로 받은 상태. 아직 한 번도 못 받았으면
        # valid=False로 응답해서, 로봇이 "아직 판단 전"과 "확실히 비어있음"을
        # 구분할 수 있게 한다.
        self._lock = threading.Lock()
        self._state: dict = {name: False for name in BASKET_NAMES}
        self._valid = False

    # ------------------------------------------------------------------
    def start_bridge(self, fastapi_loop: AbstractEventLoop, output_queue: Queue) -> None:
        """main.py의 lifespan에서 호출되어 백그라운드 스레드로 ROS2를 구동함."""
        if SrvBasketStatus is None:
            logger.warning("Basket Bridge 비활성화 상태로 시작 건너뜀 (apple_care_msgs 없음).")
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = Node(ROS_NODE_NAME)
        self.loop = fastapi_loop
        self.queue = output_queue

        self.node.create_subscription(
            String, TOPIC_BASKET_STATUS, self._basket_status_callback, 10
        )
        self.node.create_service(
            SrvBasketStatus, SERVICE_NAME, self._handle_get_basket_status
        )

        self.cv_bridge = CvBridge()
        self.node.create_subscription(
            Image, TOPIC_DEBUG_IMAGE, self._debug_image_callback,
            1,  # 이미지는 크므로 큐 깊이를 얕게 둬서 항상 최신 프레임만 유지
        )

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self._stop_event.clear()
        self.ros_thread = threading.Thread(
            target=self._spin_loop, name="ros2-basket-spin-thread", daemon=True
        )
        self.ros_thread.start()
        logger.info("Basket Bridge 시작: topic=%s service=%s", TOPIC_BASKET_STATUS, SERVICE_NAME)

    def shutdown(self) -> None:
        logger.info("BasketBridgeManager shutdown 시작...")
        self._stop_event.set()

        if self.ros_thread is not None:
            self.ros_thread.join(timeout=5.0)
            if self.ros_thread.is_alive():
                logger.warning("basket spin 스레드가 timeout 내에 종료되지 않았습니다.")

        if self.executor is not None and self.node is not None:
            self.executor.remove_node(self.node)
        if self.node is not None:
            self.node.destroy_node()

        logger.info("BasketBridgeManager shutdown 완료.")

    # ------------------------------------------------------------------
    def _spin_loop(self) -> None:
        assert self.executor is not None
        logger.info("Basket ROS2 spin 루프 시작.")

        while not self._stop_event.is_set():
            try:
                self.executor.spin_once(timeout_sec=SPIN_TIMEOUT_SEC)
            except rclpy.executors.ExternalShutdownException:
                logger.info("ExternalShutdownException 감지, basket spin 루프 종료.")
                self._stop_event.set()
                break
            except Exception:  # noqa: BLE001
                logger.error("basket spin_once 루프에서 예외 발생", exc_info=True)
                time.sleep(0.5)

        logger.info("Basket ROS2 spin 루프 종료.")

    # ------------------------------------------------------------------
    # ROS 콜백 (ROS2 스레드에서 실행)
    # ------------------------------------------------------------------
    def _basket_status_callback(self, msg: String) -> None:
        """/basket_status 수신. basket_camera.py가 publish하는 JSON을 그대로 캐싱."""
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            logger.error("/basket_status JSON 파싱 실패: %s", msg.data)
            return

        with self._lock:
            for name in BASKET_NAMES:
                if name in payload:
                    self._state[name] = bool(payload[name])
            self._valid = bool(payload.get("valid", False))

        self._dispatch({
            "type": "BASKET_STATUS",
            "payload": dict(self._state),
            "valid": self._valid,
            "timestamp": payload.get("timestamp", time.time()),
        })

    def _debug_image_callback(self, msg: Image) -> None:
        """/basket_camera/debug_image 수신 -> JPEG+base64로 압축해 HMI로 중계.

        robot_bridge.py의 _debug_image_callback(픽 카메라용 CAMERA_FRAME)과
        동일한 이유(raw Image를 그대로 JSON에 실으면 프레임당 수백KB~1MB라
        웹소켓/큐가 감당 못함)로 반드시 JPEG 압축 후 전달한다."""
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = _resize_for_transport(frame, DEBUG_IMAGE_MAX_WIDTH)
            ok, jpeg = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, DEBUG_IMAGE_JPEG_QUALITY]
            )
            if not ok:
                logger.warning("basket 디버그 이미지 JPEG 인코딩 실패")
                return
            b64_image = base64.b64encode(jpeg.tobytes()).decode('ascii')
        except Exception:  # noqa: BLE001
            logger.error("basket 디버그 이미지 처리 중 예외 발생", exc_info=True)
            return

        self._dispatch({
            "type": "BASKET_CAMERA_FRAME",
            "image": b64_image,
            "timestamp": time.time(),
        })

    def _handle_get_basket_status(self, request, response):
        """로봇이 배치 직전에 호출하는 서비스. 새로 계산하지 않고 캐시를 그대로 반환."""
        with self._lock:
            response.valid = self._valid
            response.b1_occupied = self._state["b1"]
            response.b2_occupied = self._state["b2"]
            response.b3_occupied = self._state["b3"]
            response.b4_occupied = self._state["b4"]
        return response

    # ------------------------------------------------------------------
    def _dispatch(self, payload: dict) -> None:
        if self.loop is not None and self.queue is not None:
            self.loop.call_soon_threadsafe(self._enqueue, payload)

    def _enqueue(self, payload: dict) -> None:
        try:
            self.queue.put_nowait(payload)
        except QueueFull:
            try:
                self.queue.get_nowait()
            except Exception:  # noqa: BLE001
                pass
            try:
                self.queue.put_nowait(payload)
            except QueueFull:
                logger.warning("Queue가 여전히 가득 차 있어 basket_status 메시지를 버립니다.")


# 어디서나 싱글톤처럼 접근 가능하도록 전역 인스턴스 생성 (robot_bridge.py/vision_bridge.py와 동일 패턴)
basket_bridge_manager = BasketBridgeManager()
