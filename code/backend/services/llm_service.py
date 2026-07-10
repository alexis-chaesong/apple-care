# open ai 호출
# 자연어→Policy JSON / Unknown 질문 문장 생성 / 사람 응답 해석

"""
llm_service.py
===============
OpenAI API 호출만 전담하는 파일.

이 파일은 ROS도, Queue도, DB도 모른다.
"자연어 문자열을 넣으면 구조화된 JSON이 나온다"는 순수 변환 기능만 제공.

호출부(decision_planner.py, hitl_router.py 등)는 이 파일의 함수 3개만 알면 됨:
  1. translate_policy_command()   : 작업자 자연어 -> Policy JSON 리스트
  2. generate_unknown_question()  : Unknown Object 발견 시 TTS로 읽을 질문 문장 생성
  3. interpret_human_answer()     : 사람의 STT 답변 -> destination 분류

공통 원칙:
  - 모든 호출은 response_format={"type": "json_object"}로 강제해서
    자유 텍스트가 섞여 나오는 것을 원천 차단
  - config.settings.openai_timeout_sec을 넘기면 예외를 던짐.
    호출부(특히 decision_planner)는 이 예외를 반드시 잡아서
    "일단 사람에게 물어봐(ask_human)"로 안전하게 fallback해야 함
    -> LLM이 느려지거나 죽어도 로봇이 무한 대기하지 않게 하기 위한 안전장치
"""

import json
import logging
from typing import Optional

from openai import AsyncOpenAI, APITimeoutError, APIError, AuthenticationError

from config import settings

logger = logging.getLogger(__name__)

if not settings.openai_api_key:
    # 조용히 넘어가지 않고 시작 시점에 바로 티가 나게 함.
    # (배포 후 첫 LLM 호출에서야 에러 나면 원인 찾는 데 시간 낭비됨)
    logger.warning("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

_client = AsyncOpenAI(api_key=settings.openai_api_key)
# 모듈 레벨 싱글톤. main.py의 bridge_manager, connection_manager와 동일한 패턴.
# 매 호출마다 클라이언트를 새로 만들지 않고 커넥션을 재사용함


class LLMTimeoutError(Exception):
    """OpenAI 응답이 settings.openai_timeout_sec 내에 오지 않았을 때 발생.
    decision_planner.py는 반드시 이 예외를 잡아서 ask_human으로 fallback해야 함"""
    pass


class LLMParseError(Exception):
    """OpenAI가 유효한 JSON을 반환하지 않았을 때 발생 (1회 재시도 후에도 실패한 경우)"""
    pass


class LLMAuthError(Exception):
    """OPENAI_API_KEY가 비어있거나 잘못되어 OpenAI가 인증을 거부했을 때 발생.
    재시도해도 해결되지 않으므로 호출부는 즉시 ask_human/STUCK 등으로 안전하게 처리해야 함"""
    pass


async def _call_json(system_prompt: str, user_prompt: str) -> dict:
    """
    공통 호출 헬퍼. JSON 모드로 OpenAI를 호출하고 파싱까지 마친 dict를 반환.
    실패 시 1회만 재시도 (LLM이 가끔 깨진 JSON을 뱉는 경우가 있어 최소한의 방어)
    """
    for attempt in range(2):
        try:
            response = await _client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                # temperature=0.0: 정책 해석/분류는 창의성이 필요 없고 일관성이 훨씬 중요함
                timeout=settings.openai_timeout_sec,
            )
            raw = response.choices[0].message.content
            return json.loads(raw)

        except AuthenticationError as exc:
            # AuthenticationError는 APIError의 하위 클래스이므로 아래 일반 APIError
            # 핸들러보다 반드시 먼저 잡아야 함. 재시도해도 해결되지 않는 문제이므로
            # 루프를 돌지 않고 즉시 LLMAuthError로 변환해 위로 전파함
            logger.critical("OpenAI 인증 실패 - OPENAI_API_KEY를 확인하세요.")
            raise LLMAuthError(str(exc)) from exc

        except APITimeoutError as exc:
            logger.error("OpenAI 호출 타임아웃 (%.1fs)", settings.openai_timeout_sec)
            raise LLMTimeoutError(str(exc)) from exc

        except json.JSONDecodeError:
            logger.warning("OpenAI 응답 JSON 파싱 실패, 재시도 %d/2", attempt + 1)
            if attempt == 1:
                raise LLMParseError("OpenAI가 유효한 JSON을 반환하지 않았습니다.")
            continue

        except APIError as exc:
            logger.error("OpenAI API 에러", exc_info=True)
            raise


async def translate_policy_command(raw_text: str) -> list[dict]:
    """
    작업자의 자연어/음성 명령을 Policy JSON 리스트로 변환.

    예) "오늘은 작은 사과도 판매해"
        -> [{"fruit_type": "apple", "condition": "small", "destination": "normal_box"}]

    한 문장에 여러 정책이 섞여 있을 수 있으므로(예: "작은 건 팔고 멍든 건 폐기해")
    항상 리스트로 반환. models.PolicyUpdateRequest 각각으로 그대로 매핑 가능한 형태.
    """
    system_prompt = (
        "너는 농산물 분류 로봇의 정책 번역기다. "
        "작업자의 한국어 명령을 아래 JSON 스키마의 배열로만 변환해라. "
        '스키마: {"policies": [{"fruit_type": string, "condition": string, '
        '"destination": "normal_box"|"processing_box"|"discard_box"|"ugly_box"|"ask_human"}]} '
        "ugly_box=등급외/못난이 상품(외관은 떨어지지만 폐기·가공 대상은 아닌 경우). "
        "fruit_type은 명시 안 되면 'apple'로 간주한다. "
        "condition은 'small', 'bruise', 'scratch', 'mold', 'unknown' 중 문맥에 맞는 값을 골라라. "
        "'멍'은 반드시 'bruise'로, '흠집'/'스크래치'는 반드시 'scratch'로 구분해라 (동의어로 취급하지 마라). "
        "설명이나 다른 텍스트 없이 이 JSON 객체만 출력해라."
    )
    result = await _call_json(system_prompt, raw_text)
    policies = result.get("policies", [])
    if not isinstance(policies, list):
        raise LLMParseError(f"policies 필드가 리스트가 아님: {result}")
    return policies


async def generate_unknown_question(fruit_guess: Optional[str] = None) -> str:
    """
    Unknown Object가 검출됐을 때 TTS로 읽어줄 질문 문장을 생성.

    fruit_guess: Vision이 완전히 확신은 못 해도 어느 정도 추정한 이름이 있으면 전달
                 (예: YOLO가 "kiwi일 가능성이 높음" 같은 low-confidence 예측을 준 경우)
                 없으면 None -> 일반적인 미확인 객체 질문으로 생성
    """
    system_prompt = (
        "너는 협동로봇의 음성 안내 문구 생성기다. "
        "작업자에게 새로운 과일을 어디로 보낼지 자연스럽고 짧게 물어보는 "
        "한국어 질문 문장을 아래 JSON 스키마로만 출력해라: "
        '{"question": string} '
        "문장은 1문장, 존댓말, TTS로 읽기 자연스러운 구어체로 작성해라."
    )
    user_prompt = (
        f"학습되지 않은 새로운 과일로 추정되는 객체가 감지됨 (추정: {fruit_guess})."
        if fruit_guess
        else "학습되지 않은 새로운 과일 객체가 감지됨 (종류 추정 불가)."
    )
    result = await _call_json(system_prompt, user_prompt)
    question = result.get("question")
    if not question:
        raise LLMParseError(f"question 필드가 없음: {result}")
    return question


async def interpret_human_answer(raw_answer: str) -> dict:
    """
    STT로 받은 사람의 답변을 destination으로 분류.

    예) "가공용으로 보내" -> {"destination": "processing_box"}
        "그냥 버려"       -> {"destination": "discard_box"}
        "그건 정상 판매"   -> {"destination": "normal_box"}
        "못난이로 보내"    -> {"destination": "ugly_box"}
        "무시해, 나중에 할게" -> {"destination": "skip"}

    애매하거나 다섯 가지 중 어디에도 안 맞으면 destination을 null로 반환하게 하고,
    호출부(hitl_state_machine.py)가 null이면 재질문하도록 처리해야 함
    (여기서 억지로 추측해서 잘못된 destination을 반환하면 안 됨 - 안전 최우선).

    "skip"은 재질문이 아니라 확정된 답으로 취급됨 - 이 사과의 분류를 지금 정하지
    않고 뒤로 미루겠다는 명시적 의사표현이므로, null(애매함)과 구분해야 함
    (hitl_state_machine.py가 skip을 받으면 정책 학습/박스 이동 없이 세션을 종료하고
    vision_bridge에 이 사과를 잠시 제외 대상으로 등록해 다른 사과부터 처리하게 함).
    """
    system_prompt = (
        "너는 작업자의 한국어 답변을 분류하는 역할이다. "
        "답변이 다음 다섯 가지 중 어디에 해당하는지 판단해서 "
        '{"destination": "normal_box"|"processing_box"|"discard_box"|"ugly_box"|"skip"|null} '
        "형태의 JSON으로만 출력해라. "
        "normal_box=정상 판매, processing_box=가공용/2차 상품, discard_box=폐기, "
        "ugly_box=못난이/등급외 상품(폐기까지는 아니지만 정상 판매도 아닌 경우), "
        "skip=지금 정하지 말고 나중에 하거나 넘어가자는 뜻(예: '무시해', '나중에 할게', "
        "'일단 넘어가', '다른 것부터 해'). "
        "다섯 중 무엇인지 확실하지 않으면 반드시 null을 출력해라. 추측하지 마라."
    )
    result = await _call_json(system_prompt, raw_answer)
    if "destination" not in result:
        raise LLMParseError(f"destination 필드가 없음: {result}")
    return result