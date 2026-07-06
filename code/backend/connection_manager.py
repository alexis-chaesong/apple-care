"""
connection_manager.py
=======================
WebSocket 연결 생명주기(연결/해제)와 브로드캐스트만 전담하는 파일.
robot_bridge.py(ROS 스레드)나 판단 로직(decision_planner, hitl_state_machine)의
내부 구현은 전혀 모른다 - 오직 "지금 붙어있는 클라이언트들에게 메시지를 뿌린다"는
책임만 짐 (docs/backend.txt 1번 원칙: 역할이 다르면 파일이 다르다).
"""

import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        logger.info("WebSocket 연결 수립 (현재 %d개)", len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.info("WebSocket 연결 해제 (현재 %d개)", len(self._connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """연결된 모든 클라이언트에 JSON 메시지 전송. 끊어진 연결은 조용히 정리."""
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            if ws in self._connections:
                self._connections.remove(ws)


# 전역 싱글톤 인스턴스. robot_bridge.py의 bridge_manager와 동일한 패턴 -
# main.py가 자체적으로 인스턴스를 만들지 않고 여기서 만든 걸 import해서 씀.
# 이렇게 해야 HTTP 요청 핸들러가 아닌 백그라운드 asyncio task(예: hitl_state_machine.py)도
# main.py를 거치지 않고(순환참조 없이) `from connection_manager import connection_manager`로
# 바로 접근할 수 있음.
connection_manager = ConnectionManager()
