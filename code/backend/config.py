# LLM/threshold 설정 필드 추가

"""
config.py
=========
프로젝트 전역 설정을 한곳에서 관리

다른 모듈(robot_bridge, connection_manager, main, database)은
설정값이 필요할 때 절대 os.environ을 직접 읽지 않고,
이 파일의 Settings 인스턴스(settings)를 통해서만 접근

이렇게 하면:
  - 설정값의 출처가 한 곳으로 고정되어 추적이 쉬움
  - 테스트 시 Settings만 교체하면 전체 동작을 바꿀 수 있음
  - .env / 환경변수 / 기본값의 우선순위가 한곳에서 결정됨

프로젝트의 "중앙 제어실(Single Source of Truth)"
환경 변수(os.getenv)나 하드코딩된 설정값들이 프로젝트 여기저기 흩어져 있으면 
나중에 로봇 IP가 바뀌거나 DB 주소가 바뀔 때 모든 파일을 다 뒤져야 함. 
config.py는 이를 방지하기 위해 모든 설정(ROS 2, FastAPI, DB, 큐 크기 등)을 단 하나의 객체(settings)로 모아서 관리하는 역할을 함.
"""

import os
# 시스템 환경 변수(.env 파일이나 도커 컴포즈에서 넘겨준 값)를 읽어오기 위해 가져옴
from dataclasses import dataclass, field
from typing import Optional
# 파이썬에서 데이터를 저장하는 목적의 클래스를 보일러플레이트(반복되는 코드) 없이 깔끔하게 만들 수 있도록 도와주는 내장 라이브러리

from dotenv import load_dotenv
load_dotenv()
# .env 파일의 키=값을 os.environ에 미리 채워 넣음. 아래 Settings 클래스의 각 필드가
# os.getenv(...)로 값을 읽기 전에 반드시 실행되어야 하므로 import 직후, 클래스 정의보다 위에 둠
# (find_dotenv()가 현재 작업 디렉토리부터 상위로 올라가며 .env를 찾으므로,
#  .env가 backend/ 바로 아래가 아니라 상위 디렉토리에 있어도 정상적으로 로드됨)

# 클래스 데코레이터 설정
@dataclass(frozen=True)
# 자동으로 __init__ 메소드 등을 만들어주어 객체 생성을 편하게함
# (frozen=True): 아주 중요한 설정. 이 객체는 한 번 생성되면 내부 값을 절대 수정할 수 없도록(Immutable) 동결시킴. 
# 시스템 구동 중에 실수로 런타임에 설정값이 오염되는 것을 원천 차단
class Settings:
    # -----------------------------
    # ROS2 관련 설정
    # -----------------------------
    # 구독할 토픽 이름과 메시지 타입은 실제 Doosan 토픽에 맞게 수정해야함!
    ros_node_name: str = "fastapi_bridge_node"
    ros_topic_name: str = field(
        # ros_topic_name: 로봇의 상태를 받아올 토픽명
        default_factory=lambda: os.getenv("ROS_TOPIC_NAME", "/dsr01/state")
        # field(default_factory=...): 동적으로 초기 기본값을 할당할 때 사용함.
        # 여기서는 시스템 환경 변수에서 ROS_TOPIC_NAME을 먼저 찾고, 없으면 두산 로봇의 기본 상태 토픽 예시인 "/dsr01/state"를 사용
    )
    # spin_once를 몇 초 타임아웃으로 반복할지.
    # 0.1초마다 루프를 돌며 로봇의 새 메시지가 왔는지 체크하고 없으면 통과하므로, 
    # CPU 과점유를 막으면서도 안전한 종료 처리를 가능하게 함.
    ros_spin_timeout_sec: float = 0.1

    # -----------------------------
    # Queue 관련 설정
    # -----------------------------
    # ROS 콜백이 WebSocket Broadcast보다 빠르게 쌓일 경우를 대비한 스레드 간 브리지 역할을 하는 Queue의 최대 버퍼 크기
    # 0이면 무제한(권장하지 않음 - 메모리 누수 위험).
    # 로봇이 초당 수백 개의 데이터를 쏟아내는데 웹소켓이 느려지면 메모리가 터질 수(Memory Leak) 있음.
    # 이를 막기 위해 최대 대기열을 1000개로 제한하는 안전장치
    broadcast_queue_maxsize: int = 1000

    # -----------------------------
    # FastAPI / Uvicorn 관련 설정
    # -----------------------------
    app_host: str = field(default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0"))
    # app_host: 웹 서버(FastAPI)가 요청을 받을 IP 주소
    # 기본값인 "0.0.0.0"은 외부(도커 컨테이너 외부, 혹은 다른 PC)의 접근을 허용하겠다는 의미
    app_port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    # app_port: 서버가 열릴 포트 번호입니다. 환경 변수가 없으면 기본값인 8000번 포트를 정수로 변환(int())하여 사용

    # -----------------------------
    # Database 관련 설정
    # -----------------------------
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./app.db")
    )
    # database_url: 데이터베이스 연결 문자열
    # 기본값으로 로컬 테스트용인 SQLite 파일 DB(sqlite:///./app.db) 경로를 잡아둠 

    # -----------------------------
    # 로깅 레벨
    # -----------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    # log_level: 시스템 로그의 출력 상세 수준을 결정함.
    # 배포 환경에서는 INFO나 WARNING을 쓰고, 개발 단계나 디버깅 단계에서는 환경 변수를 DEBUG로 바꾸어 더 상세한 로그를 볼 수 있게 유연성을 열어둔 것

    # ------------------------------------------------------------------
# 아래 필드들을 기존 Settings 클래스(@dataclass(frozen=True)) 본문 안에
# 그대로 이어 붙이면 됨. 클래스를 새로 만들지 않고 기존 settings 인스턴스에
# 필드만 추가하는 형태 (기존 "중앙 제어실" 패턴 그대로 유지)
# ------------------------------------------------------------------

    # -----------------------------
    # OpenAI API 관련 설정
    # -----------------------------
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    # openai_api_key: LLM 호출용 인증키. 반드시 환경변수(.env)로만 주입하고
    # 코드에 하드코딩하지 않음. 비어있으면 llm_service.py에서 시작 시점에
    # 명시적으로 에러를 내도록 처리할 예정 (조용히 실패하는 것보다 나음)

    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )
    # openai_model: 정책 번역/질문 생성/피드백 해석에 쓸 모델명.
    # 지연시간이 곧 로봇 Hold 시간으로 직결되므로 기본값은 저지연 경량 모델로 잡음.
    # 데모 리허설 때 응답 품질이 부족하면 환경변수로만 상위 모델로 교체 (코드 수정 불필요)

    openai_timeout_sec: float = field(
        default_factory=lambda: float(os.getenv("OPENAI_TIMEOUT_SEC", "8.0"))
    )
    # openai_timeout_sec: LLM 응답을 몇 초까지 기다릴지.
    # 이 시간을 넘기면 llm_service.py가 예외를 던지고, decision_planner.py는
    # "일단 사람에게 물어봐(ask_human)"로 안전하게 fallback 해야 함
    # (LLM이 죽었다고 로봇이 무한 대기하면 안 되므로 반드시 필요한 안전장치)

    # -----------------------------
    # Decision / Bayesian 관련 설정
    # -----------------------------
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    )
    # confidence_threshold: Vision이 준 confidence가 이 값보다 낮으면
    # 규칙과 무관하게 무조건 사람에게 확인받음 (decision_planner.py에서 사용)

    bayesian_prior_alpha: float = field(
        default_factory=lambda: float(os.getenv("BAYESIAN_PRIOR_ALPHA", "1.0"))
    )
    bayesian_prior_beta: float = field(
        default_factory=lambda: float(os.getenv("BAYESIAN_PRIOR_BETA", "1.0"))
    )
    # bayesian_prior_alpha/beta: 처음 보는 fruit_type+condition 조합에 대한
    # 초기 Prior. 1.0/1.0은 "아무 정보도 없다"는 균등 prior(Uniform)를 의미함

    bayesian_auto_threshold: float = field(
        default_factory=lambda: float(os.getenv("BAYESIAN_AUTO_THRESHOLD", "0.65"))
    )
    # bayesian_auto_threshold: 이 값을 confidence가 넘어야 다음부터
    # 사람에게 안 물어보고 자동 처리함.
    # prior 1.0/1.0에서 사람 피드백 1회 후 confidence = 2/3 ≈ 0.667이 되므로,
    # "두 번째 피드백부터 자동 처리"가 되려면 threshold가 0.667보다 낮아야 함 -> 0.65로 고정
    # (0.7이나 0.8로 설정되면 두 번째 피드백에서도 자동 처리가 안 되므로 반드시 0.65 미만 유지)

    # -----------------------------
    # VLA Queue 관련 설정
    # -----------------------------
    vision_queue_maxsize: int = field(
        default_factory=lambda: int(os.getenv("VISION_QUEUE_MAXSIZE", "500"))
    )
    llm_queue_maxsize: int = field(
        default_factory=lambda: int(os.getenv("LLM_QUEUE_MAXSIZE", "200"))
    )
    # vision_queue_maxsize / llm_queue_maxsize: 기존 broadcast_queue_maxsize와
    # 동일한 목적 (메모리 폭주 방지용 상한선). LLM 큐는 Vision 큐보다 처리 속도가
    # 느릴 것이 예상되므로(외부 API 왕복) 상한을 더 낮게 잡아, 밀린 요청이 쌓이면
    # 오래된 것부터 버려지도록 함 (robot_bridge.py의 QueueFull 처리 패턴과 동일)

    # -----------------------------
    # HITL 대화 관련 설정
    # -----------------------------
    hitl_response_timeout_sec: float = field(
        default_factory=lambda: float(os.getenv("HITL_RESPONSE_TIMEOUT_SEC", "15.0"))
    )
    # hitl_response_timeout_sec: 로봇이 질문한 뒤 사람의 STT 응답을 몇 초까지
    # 기다릴지. hitl_state_machine.py의 LISTENING 상태 타임아웃 기준값

    hitl_max_reask_attempts: int = field(
    default_factory=lambda: int(os.getenv("HITL_MAX_REASK_ATTEMPTS", "2"))
    )

    stt_sample_rate: int = field(default_factory=lambda: int(os.getenv("STT_SAMPLE_RATE", "16000")))
    stt_recording_duration_sec: float = field(
        default_factory=lambda: float(os.getenv("STT_RECORDING_DURATION_SEC", "8.0"))
    )
    # stt_recording_duration_sec: 마이크 녹음 자체의 길이(초).
    # hitl_response_timeout_sec(답변 전체 대기 시간)과 값이 같으면, 녹음 종료 시점과
    # hitl_state_machine.py의 asyncio.wait_for 타임아웃이 거의 동시에 발생해
    # 스케줄링 오차로 STT 결과가 도착하기 전에 Task가 cancel되는 경쟁 상태가 생김.
    # 이를 피하려고 녹음 시간을 답변 대기 시간보다 충분히 짧게 분리해서 관리함
    # (listen_once()가 실제 녹음 길이를 정할 때 이 값과 timeout_sec을 함께 고려함)
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "alloy"))
    tts_audio_player: str = field(default_factory=lambda: os.getenv("TTS_AUDIO_PLAYER", "afplay"))
    # 로컬(Mac) 개발 중엔 기본값 "afplay" 그대로 두고,
    # 실제 로봇 배포 환경(Ubuntu)에서는 .env에 TTS_AUDIO_PLAYER=ffplay 로 오버라이드

    # -----------------------------
    # Vision(ROS2 서비스 폴링) 관련 설정
    # -----------------------------
    vision_poll_interval_sec: float = field(
        default_factory=lambda: float(os.getenv("VISION_POLL_INTERVAL_SEC", "0.5"))
    )
    # vision_poll_interval_sec: obj_detection 패키지의 get_apple_status 서비스를
    # 얼마나 자주 호출할지. Vision이 토픽을 publish하는 게 아니라 요청-응답 서비스라서
    # backend가 능동적으로 폴링해야 함 (voice_processing의 UnknownObjectWatcher가
    # 쓰던 2초보다 짧게 잡아 반응성을 높임)

    vision_position_dedup_threshold_mm: float = field(
        default_factory=lambda: float(os.getenv("VISION_POSITION_DEDUP_THRESHOLD_MM", "5.0"))
    )
    # vision_position_dedup_threshold_mm: 직전 폴링 응답과 이번 응답의 position(x,y,z)
    # 각 축 차이가 전부 이 값(mm) 이내면 "같은 사과가 아직 안 치워졌다"고 보고 큐에 넣지 않음.
    # 폴링 방식 특유의 중복 처리 문제(같은 물체를 매 polling마다 새로 감지된 것처럼 착각)를
    # 막기 위한 디바운스 임계값

    # -----------------------------
    # 음성 정책 명령(웨이크워드) 관련 설정
    # -----------------------------
    voice_wakeword_enabled: bool = field(
        default_factory=lambda: os.getenv("VOICE_WAKEWORD_ENABLED", "false").lower() == "true"
    )
    # 개발 PC에는 마이크 장치/openwakeword/tflite 모델이 없을 수 있으므로 기본값 false.
    # 실제 로봇 배포 환경(.env)에서만 true로 켜서 stt_tts/wakeup_listener.py가 상시
    # 마이크를 열고 웨이크워드("헬로 로키")를 대기하도록 함

    wakeword_mic_device_index: int = field(
        default_factory=lambda: int(os.getenv("WAKEWORD_MIC_DEVICE_INDEX", "4"))
    )
    # cobot2_ws/voice_processing/MicController.py가 실제 로봇 환경에서 확인해둔 값(4)을
    # 기본값으로 가져옴. 마이크 장치가 바뀌면 pyaudio.PyAudio().get_device_info_by_index(i)로
    # 재확인 후 .env의 WAKEWORD_MIC_DEVICE_INDEX만 수정하면 됨

    stt_mic_device_index: Optional[int] = field(
        default_factory=lambda: (
            int(os.environ["STT_MIC_DEVICE_INDEX"])
            if os.getenv("STT_MIC_DEVICE_INDEX") is not None
            else None
        )
    )
    # stt_service.py(sounddevice)가 쓰는 마이크 장치. wakeword_mic_device_index는
    # pyaudio(ALSA hw 인덱스) 기준이고 이건 sounddevice(PortAudio) 기준이라 번호 체계가
    # 다를 수 있음 - scripts/check_mic_level.py로 확인 후 .env의 STT_MIC_DEVICE_INDEX로
    # 지정. 값이 없으면(None) sounddevice의 시스템 기본 입력 장치를 그대로 사용함
    # (기존 동작과 동일 - 기본 장치가 맞다면 아무것도 설정할 필요 없음)

# 모듈 전역에서 공유하는 단일 설정 인스턴스.
# 다른 파일에서는 `from config import settings` 로만 가져다 사용
# settings = Settings(): 클래스를 정의하는 것에 그치지 않고, 실제로 사용할 단 하나의 설정 객체(Instance)를 미리 찍어냄 
settings = Settings()

if not settings.openai_api_key:
    import logging
    logging.getLogger(__name__).critical(
        "OPENAI_API_KEY가 비어 있습니다. backend/.env 파일을 확인하세요."
    )

# 이 코드 덕분에, 도커(docker-compose.yml)에서 환경 변수를 어떻게 바꾸든 백엔드 내부 코드는 단 한 줄도 수정할 필요가 없어짐. 
# 파이프라인의 중심을 잡아주는 든든한 뼈대임