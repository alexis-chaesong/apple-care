# Beta(α,β) prior/posterior 업데이트
# fruit_type × condition 조합별 α,β 관리, posterior 업데이트, 자동전환 임계값 판단

"""
bayesian_policy.py
===================
tb_policy_memory의 alpha, beta를 읽고 갱신하는 유일한 창구.

이 파일은 HTTP도, LLM도, ROS도 전혀 모른다.
오직 "이 fruit_type+condition 조합에 대한 정책을 얼마나 신뢰할 수 있는가"라는
숫자 계산(Beta 분포 posterior update)만 책임진다.

핵심 아이디어 (Bayesian Trust Calibration):
  - 사람의 피드백이 기존 저장된 destination과 "일치"하면 -> alpha += 1 (신뢰 상승)
  - 기존 저장된 destination과 "불일치"하면        -> beta  += 1 (신뢰 하락)
  - confidence = alpha / (alpha + beta)
  - confidence가 설정된 임계값(settings.bayesian_auto_threshold)을 넘어야만
    다음부터 사람에게 안 물어보고 자동 처리(execute)로 넘어감

즉 단순히 "메모리에 있으면 무조건 자동"이 아니라,
사람의 답변이 몇 번 이상 "일관되게" 반복되어야 신뢰가 쌓여 자동화되는 구조.
답이 오락가락하면 confidence가 threshold를 못 넘어서 계속 확인 질문을 함.

주의: 이 파일이 정상 동작하려면 config.py에 아래 필드가 추가되어 있어야 함
  - settings.bayesian_prior_alpha   (기본값 1.0)
  - settings.bayesian_prior_beta    (기본값 1.0)
  - settings.bayesian_auto_threshold (기본값 0.8 등, 자동 전환 임계값)
"""

import logging
from typing import Optional, TypedDict

from database import get_db_connection
from config import settings

logger = logging.getLogger(__name__)


class PolicyRecord(TypedDict):
    # DB 행 하나를 다루기 쉬운 dict 형태로 옮긴 것.
    # decision_planner.py, llm_service.py 등 다른 모듈에서 이 타입을 그대로 참조해서 사용
    fruit_type: str
    condition: str
    destination: str
    alpha: float
    beta: float
    confidence: float
    source: str
    updated_at: str


def _row_to_record(row) -> PolicyRecord:
    """sqlite3.Row 객체 -> PolicyRecord dict 변환 (공통 헬퍼)"""
    return {
        "fruit_type": row["fruit_type"],
        "condition": row["condition"],
        "destination": row["destination"],
        "alpha": row["alpha"],
        "beta": row["beta"],
        "confidence": row["confidence"],
        "source": row["source"],
        "updated_at": row["updated_at"],
    }


def get_policy(fruit_type: str, condition: str) -> Optional[PolicyRecord]:
    """
    특정 fruit_type + condition 조합의 현재 정책을 조회.
    decision_planner.py가 매 프레임마다 호출하는 hot path이므로,
    나중에 트래픽이 많아지면 이 함수 위에 인메모리 캐시를 씌우는 것을 고려
    (지금 MVP 단계에서는 SQLite 직접 조회로 충분함)
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT fruit_type, condition, destination, alpha, beta, confidence, source, updated_at
            FROM tb_policy_memory
            WHERE fruit_type = ? AND condition = ?
            """,
            (fruit_type, condition),
        )
        row = cursor.fetchone()
        return _row_to_record(row) if row else None
    finally:
        conn.close()


def upsert_llm_policy(fruit_type: str, condition: str, destination: str) -> PolicyRecord:
    """작업자가 자연어로 직접 지정한 명시적 규칙을 저장. confidence=1.0으로
    즉시 최고 신뢰도 부여해서 다음 판정부터 바로 자동 적용되게 함."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO tb_policy_memory
                (fruit_type, condition, destination, alpha, beta, confidence, source, updated_at)
            VALUES (?, ?, ?, 1.0, 1.0, 1.0, 'llm_policy', datetime('now', 'localtime'))
            ON CONFLICT(fruit_type, condition) DO UPDATE SET
                destination = excluded.destination,
                confidence = 1.0,
                source = 'llm_policy',
                updated_at = datetime('now', 'localtime')
            """,
            (fruit_type, condition, destination),
        )
        conn.commit()
    finally:
        conn.close()
    updated = get_policy(fruit_type, condition)
    assert updated is not None
    return updated


def should_ask_human(fruit_type: str, condition: str) -> bool:
    """
    이 조합에 대해 사람에게 물어봐야 하는지 여부.

    True를 반환하는 경우:
      1) 정책이 아예 존재하지 않음 (처음 보는 조합)
      2) 정책은 있지만 confidence가 아직 auto_threshold를 못 넘음 (신뢰 부족)

    False(= 자동 처리 가능)를 반환하는 경우:
      - 정책이 존재하고 confidence가 auto_threshold 이상
    """
    policy = get_policy(fruit_type, condition)
    if policy is None:
        return True
    if policy["destination"] == "ask_human":
        # llm_policy로 명시적으로 "애매하면 물어봐"라고 지정된 경우도 포함
        return True
    return policy["confidence"] < settings.bayesian_auto_threshold


def record_human_feedback(
    fruit_type: str,
    condition: str,
    destination: str,
    raw_answer: str,
    session_id: Optional[str] = None,
) -> PolicyRecord:
    """
    사람의 피드백을 받아 tb_human_feedback에 이력을 남기고,
    tb_policy_memory의 alpha/beta/confidence를 갱신한 뒤 최종 정책을 반환.

    이 함수가 프로젝트의 핵심 "학습" 지점.
    hitl_router.py가 사람 응답을 받으면 이 함수 하나만 호출하면 됨.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # 1) 원본 이력은 무조건 남김 (일치하든 불일치하든 추적 목적)
        cursor.execute(
            """
            INSERT INTO tb_human_feedback (session_id, fruit_type, destination, raw_answer)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, fruit_type, destination, raw_answer),
        )

        # 2) 기존 정책 조회
        cursor.execute(
            """
            SELECT alpha, beta, destination FROM tb_policy_memory
            WHERE fruit_type = ? AND condition = ?
            """,
            (fruit_type, condition),
        )
        existing = cursor.fetchone()

        if existing is None:
            # 처음 보는 조합 -> Prior로 새 행 생성.
            # 이번 피드백을 첫 번째 "성공"으로 간주해 prior에 +1 반영
            alpha = settings.bayesian_prior_alpha + 1.0
            beta = settings.bayesian_prior_beta
            confidence = alpha / (alpha + beta)

            cursor.execute(
                """
                INSERT INTO tb_policy_memory
                    (fruit_type, condition, destination, alpha, beta, confidence, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'human_feedback', datetime('now', 'localtime'))
                """,
                (fruit_type, condition, destination, alpha, beta, confidence),
            )
        else:
            prev_alpha, prev_beta, prev_destination = (
                existing["alpha"],
                existing["beta"],
                existing["destination"],
            )

            if prev_destination == destination or prev_destination == "ask_human":
                # 기존 답과 일치 (또는 이전에 명시적 정책이 없었던 ask_human 상태였음)
                # -> 신뢰 상승
                new_alpha, new_beta = prev_alpha + 1.0, prev_beta
                new_destination = destination
            else:
                # 기존 답과 불일치 -> 신뢰 하락
                new_alpha, new_beta = prev_alpha, prev_beta + 1.0
                new_destination = prev_destination

                # beta가 alpha를 확실히 추월하면(= 기존 답이 더 이상 신뢰할 수 없으면)
                # 새 답으로 destination을 교체하고 신뢰도를 prior로 리셋.
                # 그렇지 않으면 destination은 유지한 채 신뢰도만 하락시켜 재확인을 유도.
                if new_beta > new_alpha * 1.5:
                    new_destination = destination
                    new_alpha = settings.bayesian_prior_alpha + 1.0
                    new_beta = settings.bayesian_prior_beta
                    logger.info(
                        "정책 destination 교체됨: %s/%s : %s -> %s (기존 답 신뢰 상실)",
                        fruit_type, condition, prev_destination, new_destination,
                    )

            new_confidence = new_alpha / (new_alpha + new_beta)

            cursor.execute(
                """
                UPDATE tb_policy_memory
                SET destination = ?, alpha = ?, beta = ?, confidence = ?,
                    source = 'human_feedback', updated_at = datetime('now', 'localtime')
                WHERE fruit_type = ? AND condition = ?
                """,
                (new_destination, new_alpha, new_beta, new_confidence, fruit_type, condition),
            )

        conn.commit()

    except Exception:
        conn.rollback()
        logger.error("record_human_feedback 처리 중 에러 발생", exc_info=True)
        raise
    finally:
        conn.close()

    # 최종 반영된 정책을 다시 조회해서 반환 (호출부가 바로 다음 행동을 결정할 수 있게)
    updated = get_policy(fruit_type, condition)
    assert updated is not None
    return updated