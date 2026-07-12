"""
hmi_app/api/client.py
=======================
HMI(Tkinter)가 backend(FastAPI, http://localhost:8000)와 통신하는 유일한 창구.

- REST 호출 함수들: requests 라이브러리를 쓰는 평범한 동기 함수.
  Tkinter 메인 스레드에서 직접 부르면 화면이 멈추므로(LLM이 걸리는 요청은 최대
  8초 가까이 걸릴 수 있음), 반드시 호출부(dashboard.py)가 threading.Thread로
  감싸서 호출해야 함 - 이 파일 자체는 스레딩을 전혀 모른다.
- WebSocketClient: 백그라운드 daemon thread에서 연결을 유지하며 메시지를 수신하고,
  전달받은 queue.Queue에 적재만 함. Tkinter 위젯은 이 클래스 안에서 절대 건드리지
  않음 (메인 스레드가 root.after()로 큐를 비우면서 위젯을 갱신하는 건 호출부 책임).
"""

import json
import logging
import threading
import time
from typing import Optional

import requests
import websocket

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/robot/status"

# REST 요청 타임아웃. LLM 호출(openai_timeout_sec=8초)이 걸리는 엔드포인트가 있어
# 여유를 두되, HMI가 무한 대기하지 않도록 상한은 반드시 둠
DEFAULT_TIMEOUT_SEC = 15


def get_decision_explain(audit_id: int) -> dict:
    """
    Track4(설명가능성): 특정 판정(tb_decision_audit 행)의 근거를 자연어로 설명받음.
    GET /api/decision/{audit_id}/explain -> {"result": "SUCCESS", "explanation": str, "raw": dict}

    audit_id가 없는 경우(404) 또는 LLM 호출 실패(502)면 requests.HTTPError를 던짐 -
    호출부(dashboard.py)가 messagebox로 에러를 보여줘야 함.
    """
    resp = requests.get(
        f"{BASE_URL}/api/decision/{audit_id}/explain",
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.json()


def post_vla_feedback(
    fruit_type: str, condition: str, raw_answer: str, session_id: Optional[str] = None
) -> dict:
    """
    현재 진행 중인 HITL 세션에 사람의 답변을 제출.
    POST /api/feedback -> 성공 시 {"result": "SUBMITTED", "session_id": ...}

    주의: 이 엔드포인트는 "답변을 접수했다"는 것만 알려줌. 실제 LLM 해석 성공 여부,
    destination 확정, Bayesian 학습 반영 결과는 backend가 비동기로 처리하며
    현재 WebSocket으로 별도 브로드캐스트하지 않음 (hitl_state_machine.py가
    _run_session() 완료 시점에 broadcast를 안 하고 있음 - 알려진 backend 공백).
    세션이 없거나(409) 타이밍이 안 맞으면 requests.HTTPError를 던짐.
    """
    resp = requests.post(
        f"{BASE_URL}/api/feedback",
        json={
            "fruit_type": fruit_type,
            "condition": condition,
            "raw_answer": raw_answer,
            "session_id": session_id,
        },
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.json()


def post_policy_command(raw_text: str) -> dict:
    """
    작업자 자연어 정책 명령 제출.
    POST /api/policy/command -> {"result": "SUCCESS", "raw_text": ..., "applied_policies": [...]}
    """
    resp = requests.post(
        f"{BASE_URL}/api/policy/command",
        json={"raw_text": raw_text},
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.json()


def post_robot_start() -> dict:
    """POST /api/robot/start -> {"result": "SUCCESS", "command": "START"}"""
    resp = requests.post(f"{BASE_URL}/api/robot/start", timeout=DEFAULT_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()


def post_robot_pause() -> dict:
    """POST /api/robot/pause -> {"result": "SUCCESS", "command": "MANUAL_PAUSE"}"""
    resp = requests.post(f"{BASE_URL}/api/robot/pause", timeout=DEFAULT_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()


def post_robot_estop(task_id: Optional[str] = None) -> dict:
    """POST /api/robot/emergency_stop -> {"result": "SUCCESS", "command": "EMERGENCY_STOP"}"""
    resp = requests.post(
        f"{BASE_URL}/api/robot/emergency_stop",
        json={"task_id": task_id},
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.json()


def post_robot_resume() -> dict:
    """POST /api/robot/resume -> {"result": "SUCCESS", "command": "RESUME"}"""
    resp = requests.post(f"{BASE_URL}/api/robot/resume", timeout=DEFAULT_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------
# 그리퍼 / MoveJ 제어용 함수 (TODO: backend 엔드포인트 미구현)
# ------------------------------------------------------------------
# routers/robot_router.py에는 아직 그리퍼(open/close), MoveJ(관절각 일괄 전송)에
# 대응하는 엔드포인트가 없음. 아래 두 함수는 backend 담당자가 엔드포인트를 추가할
# 것을 전제로 미리 만들어둔 것이며, 엔드포인트가 없는 동안은 항상 404로 실패한다.
# 호출부(dashboard.py)는 이 실패를 그대로 사용자에게 보여주고, 절대 로컬에서
# 임의로 "성공" 처리하지 않는다.
#
# 엔드포인트가 추가되면 필요 시 URL/바디만 맞춰 조정하면 됨:
#   POST /api/robot/gripper       {"action": "OPEN" | "CLOSE"}
#   POST /api/robot/move_joint    {"J1": .., "J2": .., ..., "J6": ..}  (models.MoveJointRequest)
def post_gripper_command(action: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/api/robot/gripper",
        json={"action": action},
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.json()


def post_move_joint(joint_angles: dict) -> dict:
    resp = requests.post(
        f"{BASE_URL}/api/robot/move_joint",
        json=joint_angles,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.json()


class WebSocketClient:
    """
    ws://localhost:8000/ws/robot/status 연결을 백그라운드 daemon thread에서 유지.
    수신 메시지(JSON)를 message_queue에 적재만 하고, 연결이 끊기면
    (backend 재시작 등) reconnect_interval_sec 간격으로 재연결을 계속 시도함.
    """

    def __init__(
        self,
        message_queue,
        url: str = WS_URL,
        reconnect_interval_sec: float = 2.0,
    ) -> None:
        self._queue = message_queue
        self._url = url
        self._reconnect_interval_sec = reconnect_interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws_app: Optional[websocket.WebSocketApp] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_forever, name="ws-client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws_app is not None:
            try:
                self._ws_app.close()
            except Exception:  # noqa: BLE001
                pass

    def _run_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._ws_app = websocket.WebSocketApp(
                    self._url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:  # noqa: BLE001
                logger.error("WebSocket 연결 중 예외 발생", exc_info=True)

            if self._stop_event.is_set():
                break
            logger.info("WebSocket 연결 끊김/실패, %.1f초 후 재연결 시도", self._reconnect_interval_sec)
            time.sleep(self._reconnect_interval_sec)

    def _on_open(self, ws) -> None:
        logger.info("WebSocket 연결 수립: %s", self._url)

    def _on_message(self, ws, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("WebSocket 메시지 JSON 파싱 실패: %s", message)
            return
        self._queue.put(data)

    def _on_error(self, ws, error) -> None:
        logger.error("WebSocket 에러: %s", error)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        logger.info("WebSocket 연결 종료 (code=%s, msg=%s)", close_status_code, close_msg)