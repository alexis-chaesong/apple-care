# VLA 관련 Pydantic 모델 추가

# 웹 백엔드(FastAPI) 개발에서 models.py는 쉽게 말해 "우리 서버에 들어오고 나가는 데이터의 규격(신분증 검사대)"을 정의하는 곳
# "너 나한테 데이터 보낼 거면 내가 정한 이 규칙(타입, 이름)에 정확히 맞춰서 보내야 해"라고 선언해 두는 역할
# FastAPI는 이 모델을 보고 데이터가 올바른 형식인지 자동으로 검증하고, 
# 만약 숫자가 와야 하는데 글자가 오면 알아서 에러(422 Unprocessable Entity)를 튕겨내줌
# Swagger UI(스웨거 문서)가 이 규격을 바탕으로 자동 완성

from pydantic import BaseModel, Field
# 파이썬에서 가장 유명한 데이터 검증 라이브러리인 Pydantic에서 BaseModel이라는 핵심 클래스를 가져옴
# 우리가 만들 데이터 규격 클래스들이 이 BaseModel을 상속받아(기능을 물려받아) 강력한 데이터 검증 기능을 가질 수 있게함

from typing import Optional
# 파이썬 표준 라이브러리에서 Optional이라는 타입을 가져옴

class TaskStartRequest(BaseModel):
    # TaskStartRequest라는 이름의 새로운 데이터 규격 클래스를 만듦
    # (BaseModel)을 붙여서 Pydantic의 데이터 검증 기능을 입혔음 
    # "Tkinter가 서버에 배출 작업을 시작해 달라고 요청(Request)할 때 보낼 JSON 데이터 포맷이야"라는 뜻
    
    mode_id: int  # 1 (일반 배출) 또는 2 (강하게 털기)
    # 이 요청 안에는 반드시 mode_id라는 이름의 필드가 있어야 하며, 그 값의 타입은 int(정수)여야 한다고 선언한 것
    # 주석에 적힌 대로 사용자가 화면에서 1번 버튼을 눌렀는지, 2번 버튼을 눌렀는지 그 번호(정수)를 받겠다는 뜻
    # 만약 Tkinter가 실수로 mode_id: "일반배출" 이렇게 문자열로 보내면 FastAPI가 단칼에 거절함

class ErrorLogRequest(BaseModel):
    # ErrorLogRequest라는 이름의 새로운 데이터 규격 클래스를 만듦
    # (BaseModel)을 붙여서 Pydantic의 데이터 검증 기능을 입혔음
    # "Tkinter가 서버에 에러 로그를 보고할 때 보낼 JSON 데이터 포맷이야"라는 뜻
    
    task_id: str
    # 에러가 발생한 배출 작업의 ID를 받으며, 타입은 str(문자열)이어야함
    # "아까 발급받은 TASK-20260622-xxxx 중에서 어떤 작업 도중에 터진 에러인지" 매핑하기 위한 고유 키값

    error_code: str  # ERR_PICK, ERR_DROP, ERR_COLLISION 등
    # 에러의 종류를 나타내는 짧은 코드를 받으며, 타입은 str(문자열)
    # 나중에 아파트 관리자가 에러 통계를 내거나 필터링하기 쉽도록 ERR_PICK(통 못 잡음), ERR_COLLISION(충돌) 같은 표준화된 코드를 문자열로 받겠다는 설계
    
    error_msg: str
    # 에러의 상세한 내용을 담는 필드이며, 타입은 str(문자열)
    # "정격 토크 초과 충돌 감지"나 "그리퍼 파지력 부족"처럼 엔지니어가 나중에 보고 수리할 수 있도록 사람이 읽을 수 있는 구체적인 실패 원인 문장을 받음

# ------------------------------------------------------------------
# 아래부터 신규 추가: 로봇 직접 제어용 요청 모델 3종
# ------------------------------------------------------------------

class EmergencyStopRequest(BaseModel):
    # "관리자 GUI에서 비상 정지 버튼을 눌렀을 때 보낼 JSON 데이터 포맷이야"라는 뜻
    task_id: Optional[str] = None
    # 특정 작업을 정지시키는 거면 그 작업의 task_id를 보내고,
    # 전체 비상정지(어떤 작업인지 특정 안 함)인 경우엔 안 보내도 되도록 Optional + 기본값 None 처리


class ResetRequest(BaseModel):
    """
    시스템 리셋 요청 바디.
    현재는 별도 입력 파라미터가 없지만, 추후 확장(예: 리셋 사유 코드 등)을
    고려해 빈 모델로 선언해둠. 요청 바디 없이 호출해도 FastAPI가 정상 처리함.
    """
    pass


class MoveJointRequest(BaseModel):
    # "관리자 GUI 슬라이더로 6축 관절각을 조정한 뒤 MoveJ 전송을 누르면 보낼 JSON 데이터 포맷이야"라는 뜻
    J1: float = Field(..., description="Joint 1 각도 (deg)")
    J2: float = Field(..., description="Joint 2 각도 (deg)")
    J3: float = Field(..., description="Joint 3 각도 (deg)")
    J4: float = Field(..., description="Joint 4 각도 (deg)")
    J5: float = Field(..., description="Joint 5 각도 (deg)")
    J6: float = Field(..., description="Joint 6 각도 (deg)")
    # 6개 필드 모두 필수(...)이며 float(실수) 타입.
    # 두산 협동로봇의 관절 가동범위 제한이 정해지면 Field(..., ge=최소값, le=최대값)으로
    # 범위 검증을 추가하는 게 안전함 (현재는 타입 검증만 수행)

# ------------------------------------------------------------------
# 아래부터 신규 추가: Apple-care VLA 프로젝트용 요청/응답 모델
# 기존 models.py 하단에 그대로 이어 붙이면 됨
# ------------------------------------------------------------------

from typing import Optional, Literal
from pydantic import BaseModel, Field


class VisionFeatureIn(BaseModel):
    # Vision 프로세스(별도 FastAPI 서버)가 판정 결과를 백엔드로 POST할 때 보낼 데이터 규격
    # "이 사과를 봤는데, 결함은 이거고, 확신도는 이 정도야"라는 뜻

    fruit_type: str
    # 검출된 과일 종류. 예: "apple", "kiwi"
    # 학습되지 않은 객체라도 YOLO가 최대한 유사한 클래스명을 주거나, unknown 처리용으로 "unknown"이 올 수 있음

    size: Optional[str] = None
    # 크기 등급. 예: "small", "normal", "large". 정책 판단에 쓰이므로 Optional이지만 있으면 우선 반영

    defect_type: Optional[str] = None
    # 결함 유형. 예: "bruise"(멍), "scratch"(흠집), "mold"(곰팡이 의심), None(결함 없음)

    confidence: float = Field(..., ge=0.0, le=1.0)
    # Vision 모델이 이 판정에 대해 갖는 확신도. 0.0~1.0 범위로 강제
    # decision_planner.py가 이 값을 threshold와 비교해서 자동 처리할지 사람에게 물을지 결정함

    unknown_flag: bool = False
    # 학습되지 않은 새로운 객체(예: 처음 보는 키위)인 경우 True
    # True면 confidence 값과 무관하게 Unknown Object 처리 경로로 감

    bbox: Optional[list[float]] = None
    # Bounding Box 좌표 [x_min, y_min, x_max, y_max]. Motion Planner의 Pick Pose 계산에 사용
    # 지금 단계에서 필수는 아니라 Optional로 둠 (Vision 팀과 좌표계 합의 후 필수로 바꿔도 됨)

    center: Optional[list[float]] = None
    # 객체 중심 3D 좌표 [x, y, z]. RealSense Depth로 계산됨

    frame_id: Optional[str] = None
    # 어느 카메라 프레임에서 나온 데이터인지 추적용 (디버깅/로그 매칭 목적)


class PolicyUpdateRequest(BaseModel):
    # 작업자가 자연어/음성으로 정책을 바꿨을 때, LLM이 해석한 결과를 backend가 저장할 때 쓰는 규격
    # "오늘은 작은 사과도 판매해" 같은 원문과, LLM이 변환한 JSON을 함께 보관

    raw_text: str
    # 작업자가 실제로 말하거나 입력한 원문. 나중에 로그 추적/재현할 때 원인 파악용으로 반드시 저장

    fruit_type: str
    # 이 정책이 적용되는 과일 종류

    condition: str
    # 정책이 걸리는 조건 키. 예: "small", "scratch", "mold", "unknown"

    destination: Literal["normal_box", "processing_box", "discard_box", "ask_human"]
    # 조건에 대응하는 목적지. 임의 문자열이 아니라 4가지로 강제해서 오타로 인한 사고를 방지

    source: Literal["llm_policy", "human_feedback"] = "llm_policy"
    # 이 정책이 "작업자 자연어 명령"으로 생긴 건지, "Unknown Object에 대한 개별 피드백"으로 생긴 건지 구분
    # Bayesian Update 대상은 human_feedback만 해당 (llm_policy는 명시적 규칙이라 확률 갱신 대상 아님)


class HumanFeedbackRequest(BaseModel):
    # STT로 받은 사람의 답변을 LLM이 해석한 뒤, backend가 Memory에 저장할 때 쓰는 규격
    # "Unknown 물어봤더니 '가공용으로 보내'라고 답했다"는 상황을 표현

    fruit_type: str
    # 질문 대상이었던 과일 종류. 예: "kiwi"

    destination: Literal["normal_box", "processing_box", "discard_box"]
    # 사람이 지정한 목적지. ask_human은 여기 올 수 없음 (사람이 이미 답했으므로)

    raw_answer: str
    # 사람이 실제로 말한 원문 답변 (예: "가공용으로 보내"). 해석이 잘못됐을 때 원인 추적용

    session_id: Optional[str] = None
    # 어떤 HITL 대화 세션에 대한 응답인지 추적 (hitl_state_machine.py가 발급)


class DecisionResult(BaseModel):
    # decision_planner.py가 판단을 끝낸 뒤 내놓는 최종 결과 규격
    # Motion Planner와 HITL 상태 머신이 각각 이 결과를 보고 다음 행동을 정함

    action: Literal["execute", "ask_human"]
    # execute면 바로 Motion Queue로, ask_human이면 hitl_state_machine으로

    destination: Optional[str] = None
    # action이 execute일 때만 값이 채워짐. 어느 박스로 보낼지

    reason: Literal["rule_match", "memory_match", "low_confidence", "unknown_object"]
    # 왜 이런 판단이 나왔는지 근거. 로그/디버깅 및 데모 시연 설명용으로 중요

    fruit_type: str
    condition: str
    # 판단의 근거가 된 조건 키 (예: "normal", "mold", "scratch", "unknown")
    # tb_policy_memory의 복합키(fruit_type, condition)와 매칭하기 위해 반드시 필요

    confidence: float


class HoldCommand(BaseModel):
    # 백엔드가 robot_bridge.publish_command()를 통해 ROS로 내려보낼 HOLD/RESUME 명령 규격
    # 기존 EmergencyStopRequest와 패턴을 맞춤 (task_id Optional 처리 등)

    command_type: Literal["HOLD", "RESUME"]
    reason: Optional[str] = None
    # 왜 멈추는지 (예: "unknown_object_detected", "low_confidence"). HMI 화면 표시용

class RawFeedbackIn(BaseModel):
    # STT로 받은 "해석 전" 원본 답변을 hitl_router가 받을 때 쓰는 입력 규격
    # HumanFeedbackRequest와 다른 점: destination을 아직 모름 (LLM이 이제 해석할 차례)
    fruit_type: str
    condition: str
    # 어떤 조건에 대한 질문이었는지 (Unknown Object면 "unknown", 그 외엔 defect_type 등)
    raw_answer: str
    session_id: Optional[str] = None