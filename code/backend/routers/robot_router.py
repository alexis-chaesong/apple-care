"""
routers/robot_router.py
=========================
로봇 직접 제어용 엔드포인트(EmergencyStopRequest, ResetRequest, MoveJointRequest 등,
models.py에 이미 정의된 요청 스키마)를 등록할 라우터.

실제 엔드포인트 구현은 로봇 팀과 커맨드 스펙을 확정한 뒤 별도로 채울 예정이라,
지금은 main.py의 app.include_router(robot_router.router) 호출이 깨지지 않도록
빈 라우터만 선언해둠.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/robot")
