# POST /api/vision/result — Vision→Backend 판정 수신
# Vision 프로세스가 판정 결과(JSON)를 POST로 보내면 받아서 vision_queue에 적재

"""
vla_router.py
=============
Vision 프로세스(별도 FastAPI 서버)가 판정 결과를 POST로 보내는 창구.

이 라우터는 판정 결과를 받아 Pydantic 검증만 하고, 실제 판단(decision_planner)이나
LLM 호출은 절대 하지 않는다. vision_queue에 적재만 하고 즉시 응답을 돌려줌.

왜 여기서 바로 decide()를 안 부르는가:
  Vision이 프레임마다 빠르게 연달아 POST를 보낼 수 있는데, 요청 핸들러 안에서
  직접 decide() -> (필요시) LLM 호출까지 다 하면, 느린 LLM 응답이 이 HTTP 커넥션을
  붙잡고 있게 되어 Vision 쪽 다음 프레임 전송이 밀릴 위험이 있음.
  main.py의 _vla_consumer_loop()가 vision_queue를 백그라운드에서 순서대로 소비하며
  decide()를 호출하는 구조로 분리해, Vision 프로세스는 절대 대기 없이
  "보내고 바로 다음 프레임으로" 넘어갈 수 있게 함.
  robot_bridge.py가 ROS 콜백에서 절대 블로킹 작업 없이 큐에만 적재하는 것과 동일한 원칙.

큐 접근 방식:
  main.py의 lifespan에서 app.state.vision_queue에 큐 인스턴스를 저장해두고,
  라우터는 request.app.state를 통해서만 접근함 (모듈 직접 import로 인한
  main.py <-> router 순환 참조를 피하기 위한 FastAPI 표준 패턴).
"""

import logging
from asyncio import QueueFull
from fastapi import APIRouter, Request, HTTPException

from models import VisionFeatureIn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.post("/vision/result", summary="Vision 프로세스의 판정 결과 수신")
async def receive_vision_result(payload: VisionFeatureIn, request: Request):
    """
    Vision 프로세스가 사과 1개를 판정할 때마다 호출하는 엔드포인트.
    """
    vision_queue = request.app.state.vision_queue

    try:
        vision_queue.put_nowait(payload)
    except QueueFull:
        # 큐가 가득 찼으면 가장 오래된 항목을 버리고 최신 데이터를 우선함
        # (robot_bridge.py의 _enqueue 패턴과 동일 - 로봇 제어에서는 과거보다 현재가 중요)
        logger.warning("vision_queue가 가득 참, 가장 오래된 항목을 버림")
        try:
            vision_queue.get_nowait()
            vision_queue.put_nowait(payload)
        except Exception as exc:
            logger.error("vision_queue 적재 재시도 실패, 데이터 유실됨", exc_info=True)
            raise HTTPException(
                status_code=503, detail="vision_queue가 가득 찼습니다."
            ) from exc

    return {"result": "ACCEPTED", "queue_size": vision_queue.qsize()}