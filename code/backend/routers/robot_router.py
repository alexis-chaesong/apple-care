"""
routers/robot_router.py
=========================
로봇 직접 제어용 엔드포인트(EmergencyStopRequest, ResetRequest, MoveJointRequest 등,
models.py에 이미 정의된 요청 스키마)를 등록할 라우터.

START/PAUSE/EMERGENCY_STOP은 robot_bridge.py의 publish_command()가 커맨드 타입을
화이트리스트로 제한하지 않는 제네릭 구조라는 것을 그대로 활용함 (HOLD/RESUME/
SORT_DECISION과 동일한 패턴). 실제 ROS 노드가 이 커맨드들을 받아 무엇을 하는지는
로봇 팀 담당이라, 여기서는 "ROS 쪽으로 발행까지는 확실히 한다"만 보장함.
"""

from fastapi import APIRouter

from models import EmergencyStopRequest
from robot_bridge import bridge_manager

router = APIRouter(prefix="/api/robot")


@router.post("/start", summary="로봇 시스템 시작")
async def start_robot():
    bridge_manager.publish_command(command_type="START")
    return {"result": "SUCCESS", "command": "START"}


@router.post("/pause", summary="로봇 동작 수동 일시정지")
async def pause_robot():
    # HITL 흐름에서 decision_planner.decide()가 자동으로 거는 "HOLD"와는
    # 별개의 커맨드 타입(MANUAL_PAUSE)임. 관리자가 직접 누른 수동 정지와
    # 시스템이 판단 불확실성 때문에 자동으로 거는 정지를 ROS/로그에서
    # 구분할 수 있어야 하므로 절대 "HOLD"를 재사용하지 않음
    bridge_manager.publish_command(command_type="MANUAL_PAUSE")
    return {"result": "SUCCESS", "command": "MANUAL_PAUSE"}


@router.post("/emergency_stop", summary="긴급 정지")
async def emergency_stop_robot(payload: EmergencyStopRequest):
    bridge_manager.publish_command(command_type="EMERGENCY_STOP", task_id=payload.task_id)
    return {"result": "SUCCESS", "command": "EMERGENCY_STOP"}


@router.post("/resume", summary="긴급 정지 이후 재개")
async def resume_robot():
    # estop_handler.check_and_recover()가 물리 복구(상승->홈->카메라)까지 끝낸 뒤
    # wait_for_resume으로 이 명령을 기다림 - 운영자가 현장을 확인하고 프론트엔드
    # 재개 버튼을 눌러야만 다음 사이클(비전 재탐지)이 진행되도록 하기 위함.
    bridge_manager.publish_command(command_type="RESUME")
    return {"result": "SUCCESS", "command": "RESUME"}
