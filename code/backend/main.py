# 백엔드 핵심 서버 코드
# [Architecture] ROS 2(Thread)와 FastAPI(AsyncIO)의 실행 모델 분리를 위한 Queue 기반 Producer-Consumer 구조.
# RobotBridge(ROS)는 오직 Queue 적재만 담당하며, 웹소켓(ConnectionManager) 및 HTTP API와 완전 분리되어 결합도를 낮춤.
# 이를 통해 스레드 간 경계를 명확히 하고 비동기 I/O 병목을 해결하여 실시간 데이터 브리지의 안정성을 확보함.
#
# 코드는 FastAPI의 비동기 이벤트 루프와 ROS 2의 스레드라는 서로 다른 실행 모델을 asyncio.Queue 기반의 Producer-Consumer 패턴으로 완벽하게 중재하고 있음.
#
# VLA(Vision-Language-Action) 파이프라인도 동일한 Producer-Consumer 패턴으로 이어붙임:
#   Vision 프로세스 -> vla_router.py(POST /api/vision/result) -> vision_queue -> _vla_consumer_loop
#   -> decision_planner.decide() -> execute(ROS 발행 + HMI 브로드캐스트) 또는 ask_human(hitl_state_machine)

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import init_db
from models import VisionFeatureIn
from robot_bridge import bridge_manager
from vision_bridge import vision_bridge_manager
from connection_manager import connection_manager
from services.decision_planner import decide
from state.hitl_state_machine import hitl_state_machine
from routers import robot_router, vla_router, hitl_router


logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

## 초기 선언 및 글로벌 객체 생성

# ROS Callback → Queue → Background Task → ConnectionManager.broadcast() 구조의
# 중심이 되는 두 객체. main.py 모듈 레벨에 두어 lifespan/엔드포인트에서 공유
broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
# ROS 2가 던져준 데이터를 임시로 담아두는 비동기 큐. 최대 크기(maxsize=1000)를 지정해 메모리 폭주를 막음
# connection_manager 싱글톤은 connection_manager.py 자체에서 생성됨(bridge_manager와 동일 패턴).
# main.py는 그걸 import해서 쓸 뿐 - 이렇게 해야 hitl_state_machine.py 같은 백그라운드
# asyncio task도 main.py를 거치지 않고(순환참조 없이) 직접 import해서 broadcast를 호출할 수 있음

_consumer_task: asyncio.Task | None = None
# 백그라운드에서 평생 돌며 큐를 감시할 타스크 객체를 저장할 변수

vision_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.vision_queue_maxsize)
# Vision 프로세스가 vla_router.py를 통해 판정 결과를 밀어 넣는 큐.
# broadcast_queue(ROS 상태용)와는 별개의 소비 로직이므로 큐 자체도 분리함

_vla_consumer_task: asyncio.Task | None = None
# _consumer_task(기존 broadcast_queue용)와 별개로 관리되는 두 번째 백그라운드 태스크.
# 두 큐(broadcast_queue, vision_queue)는 서로 다른 소비 로직이므로 태스크도 분리함
# (하나의 루프에서 두 큐를 같이 처리하면, 한쪽이 느려질 때 다른 쪽까지 밀리는 문제가 생김)


async def _queue_consumer_loop() -> None:
    """
    Queue에서 ROS 상태 데이터를 꺼내 프론트엔드 HMI 포맷으로 정제한 뒤
    ConnectionManager.broadcast()로 넘기는 유일한 Background Task.
    (ROS Callback → Queue → [데이터 정제] → 여기 → WebSocket)
    """
    print("Queue consumer 루프 시작.", flush=True)

    # 최신 상태를 누적 유지하는 캐시 (함수 호출 1회 동안 while 루프에서 계속 유지됨)
    latest_status = {
        "process_state": "대기 중 (사용자 이용 전)",
        "task_id": None,
        "gripper_state": "UNKNOWN",

        # [백업 대비 추가] 로봇이 긴급정지된 정확한 공정 체크포인트를 보관
        # 기존에는 화면용 process_state만 있어 배출 전/후를 정확히 구분할 수 없었음
        # 이 값으로 HMI가 HOME 복귀와 중단 공정 재개를 안전하게 분기
        "recovery_stage": "IDLE",

        "joints": {"J1": 0.0, "J2": 0.0, "J3": 0.0, "J4": 0.0, "J5": 0.0, "J6": 0.0},
    }

    try:
        while True:
            # 1) ROS 2 브리지 스레드가 적재한 원본 raw 데이터 추출 (dict 형태 가정)
            raw_ros_data = await broadcast_queue.get()
            try:
                msg_type = raw_ros_data.get("type")

                # CAMERA_FRAME은 로봇 상태 캐시(latest_status)와 무관한 별도 스트림이라,
                # 캐시 갱신/ROBOT_STATUS 누적 브로드캐스트 로직을 타지 않고 그대로 즉시 전달만 함
                if msg_type == "CAMERA_FRAME":
                    await connection_manager.broadcast(raw_ros_data)
                    continue

                # 2) 들어온 메시지 타입에 따라 최신 상태 캐시만 갱신 (나머지는 유지)
                if msg_type == "PROCESS_STATE":
                    latest_status["process_state"] = raw_ros_data.get("payload", latest_status["process_state"])
                elif msg_type == "GRIPPER_STATUS":
                    latest_status["gripper_state"] = "GRASPED" if raw_ros_data.get("grasped") else "OPEN"
                elif msg_type == "SAFETY_EVENT":
                    latest_status["last_safety_event"] = {
                        "error_code": raw_ros_data.get("error_code"),
                        "error_msg": raw_ros_data.get("error_msg"),
                        "timestamp": raw_ros_data.get("timestamp", time.time()),
                    }

                # [백업 대비 추가] robot_bridge가 전달한 RecoveryStage를 최신 상태 캐시에 반영
                # 긴급정지 직후에도 마지막 안전 체크포인트가 유지되므로 RESET 복구 기준이 사라지지 않음
                elif msg_type == "RECOVERY_STAGE":
                    latest_status["recovery_stage"] = raw_ros_data.get("stage", latest_status["recovery_stage"])

                # MOTION_STATUS, joints 갱신용 토픽이 추가되면 여기에 elif로 계속 확장

                # 3) 현재 GUI(process_queue)가 인식하는 형태로 PROCESS_STATE 브로드캐스트
                #    -> data.get("type") == "PROCESS_STATE", data.get("payload")가 문자열이어야 함
                if msg_type == "PROCESS_STATE":
                    await connection_manager.broadcast({
                        "type": "PROCESS_STATE",
                        "payload": latest_status["process_state"],
                        "timestamp": time.time(),
                    })
                elif msg_type == "SAFETY_EVENT":
                    await connection_manager.broadcast({
                        "type": "SAFETY_EVENT",
                        "error_code": raw_ros_data.get("error_code"),
                        "error_msg": raw_ros_data.get("error_msg"),
                        "timestamp": raw_ros_data.get("timestamp", time.time()),
                    })

                # [백업 대비 추가] 복구 단계 변경을 HMI에 즉시 전송
                # HMI는 이 이벤트로 진행 바를 갱신하고, WASTE_DUMPED 전에는 HOME 복귀,
                # 이후에는 남은 세척·배수·복귀 동작 재개로 버튼과 화면을 전환
                elif msg_type == "RECOVERY_STAGE":
                    await connection_manager.broadcast({
                        "type": "RECOVERY_STAGE",
                        "stage": latest_status["recovery_stage"],
                        "timestamp": raw_ros_data.get("timestamp", time.time()),
                    })

                # 4) 누적 캐시 전체("ROBOT_STATUS")도 함께 브로드캐스트
                #    -> 지금 GUI는 무시하지만, 추후 관절각/그리퍼 표시 추가 시 바로 쓸 수 있음
                formatted_payload = {
                    "type": "ROBOT_STATUS",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "payload": dict(latest_status),
                }
                await connection_manager.broadcast(formatted_payload)

            except Exception:  # noqa: BLE001
                logger.error("데이터 정제 및 웹소켓 broadcast 중 에러 발생", exc_info=True)
    except asyncio.CancelledError:
        # 서버가 종료될 때 이 루프를 안전하게 취소(cancel)하여 자원을 깔끔하게 해제
        print("Queue consumer 루프 취소됨, 정상 종료.", flush=True)
        raise
# 데이터 파이프라인의 생명주기(Lifespan)와 결합도로 robot_router가 아닌 main에서 GET /ws/robot/status
# 웹소켓 엔드포인트 함수(websocket_endpoint)는 클라이언트가 들어오면 connection_manager.connect(websocket)를 호출해야 하므로, 이 인스턴스에 반드시 접근할 수 있어야함


async def _vla_consumer_loop() -> None:
    """
    vision_queue에서 Vision 판정 결과를 하나씩 꺼내 decision_planner.decide()를 호출하고,
    결과에 따라 즉시 실행(SORT_DECISION 발행)하거나 hitl_state_machine으로 넘기는
    백그라운드 Background Task.

    (Vision Feature -> Queue -> [여기] -> decide() -> execute or ask_human)

    주의: decide() 자체는 동기 함수이고 DB 조회만 하므로 빠르다.
    느릴 수 있는 LLM 호출은 ask_human 경로를 탈 때만 hitl_state_machine.handle_ask_human()이
    별도 백그라운드 task로 실행하므로, 이 루프 자체는 절대 블로킹되지 않고
    다음 vision_queue 항목을 계속 빠르게 소비할 수 있음.
    """
    print("VLA Queue consumer 루프 시작.", flush=True)

    try:
        while True:
            raw_feature: VisionFeatureIn = await vision_queue.get()

            try:
                result = decide(raw_feature)

                if result.action == "execute":
                    # 즉시 실행 가능한 경우 - 로봇팀 motion_planner_node.py가 구독하는
                    # /decision/result로 발행 (2단계 결정: /robot/command로 로봇팀을
                    # 맞추게 하지 않고, 우리가 그쪽 스펙에 맞춰 발행하는 쪽으로 감)
                    bridge_manager.publish_decision_result(
                        fruit=result.fruit_type,
                        destination=result.destination,
                        pose=result.position,
                        reason=result.reason,
                    )
                    # 실시간 모니터링을 위해 HMI에도 브로드캐스트
                    await connection_manager.broadcast({
                        "type": "VLA_DECISION",
                        "payload": result.model_dump(),
                        "timestamp": time.time(),
                    })

                else:  # action == "ask_human"
                    # HITL 세션은 백그라운드 task로 실행 - 이 루프를 블로킹하지 않음
                    # result.condition은 decision_planner.decide()가 이미 계산해 넘겨준 값을
                    # 그대로 재사용 (여기서 조건을 다시 계산하면 우선순위 로직이 두 군데로
                    # 흩어져 서로 어긋날 위험이 있으므로 반드시 decide()의 결과만 신뢰함)
                    #
                    # VLA_ASK_HUMAN 브로드캐스트는 여기서 하지 않는다 - 세션이 이미
                    # 진행 중이면 handle_ask_human()이 대기열에 쌓기만 하는데, 예전엔
                    # 그래도 매번 여기서 브로드캐스트해서 같은 물체의 flicker/재시도마다
                    # HITL 팝업이 반복해서 새로 뜨는 문제가 있었음. 이제
                    # hitl_state_machine._run_session()이 실제 세션 시작 시점에만
                    # 한 번 브로드캐스트한다.
                    await hitl_state_machine.handle_ask_human(
                        fruit_type=result.fruit_type,
                        condition=result.condition,
                        vision_confidence=result.confidence,
                    )

            except Exception:  # noqa: BLE001
                logger.error("VLA 판단 처리 중 에러 발생", exc_info=True)

    except asyncio.CancelledError:
        print("VLA Queue consumer 루프 취소됨, 정상 종료.", flush=True)
        raise


## 애플리케이션 생명주기 관리 (lifespan)
# FastAPI 서버가 켜질 때(Startup)와 꺼질 때(Shutdown) 작동하는 컨트롤러

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _consumer_task, _vla_consumer_task
    try:
        print("lifespan start")

        init_db() # 1) DB 초기화

        current_loop = asyncio.get_running_loop()
        # robot_bridge는 이제 connection_manager를 모르고, broadcast_queue만 받음
        # 2) ROS 2 브리지 스레드 가동
        bridge_manager.start_bridge(current_loop, broadcast_queue)
        # 현재 FastAPI가 돌고 있는 비동기 이벤트 루프(current_loop)와 데이터 바구니(broadcast_queue)를 ROS 2 브리지에 넘겨주며 스레드를 실행함.
        # 이제 robot_bridge는 웹소켓이 뭔지 몰라도 이 큐에 데이터를 밀어 넣을 수 있게 됨

        # 2-1) Vision Bridge 가동 (get_apple_status 서비스를 폴링해서 vision_queue에 적재)
        vision_bridge_manager.start_bridge(current_loop, vision_queue)

        # 3) 백그라운드 큐 소비자 가동
        _consumer_task = asyncio.create_task(_queue_consumer_loop())
        # : 앞서 만든 소비자 루프(_queue_consumer_loop)를 백그라운드에서 비동기로 독립시켜 상시 구동

        # 3-1. VLA 큐 소비자 가동
        _vla_consumer_task = asyncio.create_task(_vla_consumer_loop())

        # 3-2. 큐 인스턴스를 app.state에 등록
        # 라우터(vla_router.py)가 main.py를 직접 import하지 않고도
        # request.app.state를 통해 큐에 접근할 수 있게 하기 위함 (순환 참조 방지)
        app.state.vision_queue = vision_queue

        print("lifespan startup finished")

        yield
        # 이 지점에서 서버가 정상 가동되며 사용자의 요청을 받기 시작

    except Exception as e:
        print("LIFESPAN EXCEPTION:", repr(e), flush=True)
        raise

    finally:
        # # 서버가 꺼질 때 안전하게 청소(Graceful Shutdown)
        print("lifespan shutdown", flush=True)

        # 1) consumer task를 먼저 취소
        if _consumer_task is not None:
            _consumer_task.cancel() # 소비자 정지
            try:
                await _consumer_task
            except asyncio.CancelledError:
                pass

        # 1-1) VLA consumer task도 함께 안전하게 취소
        if _vla_consumer_task is not None:
            _vla_consumer_task.cancel()
            try:
                await _vla_consumer_task
            except asyncio.CancelledError:
                pass

        # 2) ROS spin 스레드 정리 + rclpy.shutdown
        # vision_bridge_manager를 먼저 정리하고(rclpy.shutdown()을 호출하지 않으므로 순서 무관하지만
        # 관례상 마지막에 전체 rclpy 컨텍스트를 내리는 bridge_manager보다 먼저 정리함), 그다음 bridge_manager.shutdown()
        vision_bridge_manager.shutdown()
        bridge_manager.shutdown() # ROS 2 스레드 및 rclpy 안전 종료
        # 서버가 꺼질 때 백그라운드 태스크를 먼저 취소하고, ROS 2 스레드까지 안전하게 셧다운(bridge_manager.shutdown())하여 좀비 프로세스가 남는 것을 원천 차단

app = FastAPI(
    title="Apple-care",
    description="apple-care API",
    version="1.0.0",
    lifespan=lifespan,
)
# CORS 미들웨어 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계에서는 전체 허용, 추후 프론트 주소만 지정 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(robot_router.router)
app.include_router(vla_router.router)
app.include_router(hitl_router.router)

@app.get("/")
def read_root():
    return {"message": "AutoDumpBot Backend Server is Running!"}

## 웹소켓 엔드포인트 (/ws/robot/status)
# 기존: bridge_manager.add_connection / remove_connection을 직접 호출
# 변경: connection_manager에 등록만 함. robot_bridge는 이 함수의 존재를 모름.
# 프론트엔드가 웹소켓을 연결하면 connection_manager 주머니에 넣고 관리

# 이 엔드포인트는 오직 클라이언트의 접속과 해제만 관리
# 로봇이 데이터를 보내든 말든 상관하지 않으며, 데이터 전송은 오직 백그라운드 소비자(_queue_consumer_loop)가 담당하므로 코드가 매우 단단해짐
@app.websocket("/ws/robot/status")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.disconnect(websocket)