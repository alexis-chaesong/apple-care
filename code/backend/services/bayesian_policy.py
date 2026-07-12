# Track 3: 계층적 Prior 도입 - Beta(α,β) 업데이트 로직

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
  - confidence = alpha / (alpha + beta)  (리서치 문서 §4.3.3의 p = alpha_f,c/(alpha_f,c+beta_f,c)와 동일)

즉 단순히 "메모리에 있으면 무조건 자동"이 아니라,
사람의 답변이 몇 번 이상 "일관되게" 반복되어야 신뢰가 쌓여 자동화되는 구조.

"물어볼지 말지"의 최종 판단(과거엔 confidence와 고정 임계값을 직접 비교했으나,
지금은 services/human_query_gate.py의 EVPI_human vs Cost_human(t) 게이트,
§4.3.5가 담당함)은 이 파일의 책임이 아니다 - 이 파일은 오직 alpha/beta/confidence
숫자 계산과 저장만 담당한다.

주의: 이 파일이 정상 동작하려면 config.py에 아래 필드가 추가되어 있어야 함
  - settings.bayesian_prior_alpha   (기본값 1.0)
  - settings.bayesian_prior_beta    (기본값 1.0)
"""

import logging
from typing import Optional, TypedDict

from database import get_db_connection
from config import settings
from services.decision_audit import log_prior_initialized

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


def delete_policy(fruit_type: str, condition: str) -> bool:
    """
    특정 fruit_type+condition 정책을 완전히 삭제해 "아직 한 번도 안 배운 상태"로
    되돌린다. 다음에 이 조합이 나오면 get_policy()가 None을 반환하므로
    decision_planner.decide()가 다시 ask_human으로 처리하고, 그 답변부터 alpha/beta가
    새로 시작된다.

    record_human_feedback()의 학습 구조상, 처음 답변 한 번만으로 confidence가
    충분히 올라 EVPI_human <= Cost_human(t)이 되어 위험 감수 실행으로 넘어가버리면
    그 뒤로는 다시 안 물어보므로(잘못된 답이었어도) 자연스럽게 정정될 기회가 없다 -
    이 함수가 그런 경우 정책을 원점으로 되돌리는 유일한 방법이다 (POST /api/policy/command로
    명시적 규칙을 덮어쓰는 것과는 달리, "확정 규칙을 새로 심는다"가 아니라
    "아예 모른다"로 되돌림).

    Returns:
        bool: 실제로 삭제된 행이 있었으면 True (애초에 없던 조합이면 False)
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM tb_policy_memory WHERE fruit_type = ? AND condition = ?",
            (fruit_type, condition),
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


def reset_all_policies() -> int:
    """
    tb_policy_memory 전체를 비워 모든 조합을 "아직 한 번도 안 배운 상태"로
    되돌린다. 되돌릴 수 없는 작업이므로 호출부(라우터)가 명시적 확인을 받은
    뒤에만 호출해야 한다.

    Returns:
        int: 삭제된 행 수
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tb_policy_memory")
        count = cursor.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


class ThetaInit(TypedDict):
    alpha: float
    beta: float
    method: str
    pooled_mean: Optional[float]
    pooled_sample_size: Optional[int]


def get_hierarchical_prior(fruit_type: str, condition: str) -> tuple[float, float, Optional[float], int]:
    """§5.4 계층적 Prior (partial pooling).

    문서에 구체적 수식은 없다 - §5.4의 의도("신규 과일이 여러 세트에 걸쳐 등장할 때
    더 빠르게 안정적 신뢰도에 도달")를 살리기 위해 이 프로젝트가 자체 설계한 방식:

        pooled_mean = 같은 condition을 이미 사람 피드백으로 학습한(source=
                      'human_feedback') "다른" fruit_type들의 confidence
                      (=alpha/(alpha+beta)) 평균
        prior_alpha = K * pooled_mean
        prior_beta  = K * (1 - pooled_mean)

    source='human_feedback' 행만 pooling 대상에 포함하는 이유: llm_policy 행은
    confidence가 실제 학습된 신뢰도가 아니라 작업자가 강제로 고정한 값(1.0)이라,
    풀링에 넣으면 "이 조건은 다른 과일에서도 항상 확실했다"는 왜곡된 신호를 만들어낸다.

    K(pseudo-count, settings.hierarchical_prior_pseudo_count)를 작게 유지해야
    "약한 prior"라는 의도가 지켜진다 - 사람 피드백 한두 번만으로도 이 prior가 실제
    관측으로 빠르게 압도되어야 한다.

    한계(§5.4/§6): 본 연구는 계층적 prior 검증이 소규모 사례 시연이며 일반화
    근거로는 부족하다. 이 함수의 pooled_mean 계산도 마찬가지로 데모 규모 표본에
    기반한 설계 결정이며, 대규모 다종 과일 데이터로 검증된 것은 아니다.

    Returns:
        (prior_alpha, prior_beta, pooled_mean, pooled_sample_size).
        유효 표본이 없으면(pooled_sample_size=0) Beta(1,1) 균등 prior로 폴백
        (pooled_mean=None으로 반환해 폴백이 일어났음을 명시).
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT confidence FROM tb_policy_memory
            WHERE condition = ? AND fruit_type != ? AND source = 'human_feedback'
            """,
            (condition, fruit_type),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return (settings.bayesian_prior_alpha, settings.bayesian_prior_beta, None, 0)

    confidences = [row["confidence"] for row in rows]
    pooled_mean = sum(confidences) / len(confidences)
    k = settings.hierarchical_prior_pseudo_count
    prior_alpha = k * pooled_mean
    prior_beta = k * (1 - pooled_mean)
    return (prior_alpha, prior_beta, pooled_mean, len(confidences))


def get_or_init_theta(fruit_type: str, condition: str) -> ThetaInit:
    """θ_f,c(alpha, beta)의 단일 진실 소스.

    이 함수 하나만 있어야 한다 - record_human_feedback()과
    hitl_state_machine.py의 감사 스냅샷(alpha_before/beta_before) 양쪽 모두
    이 함수를 호출해야 값이 갈라지지 않는다 (Track4 "재발 방지" 항목과 동일 이유:
    hitl_state_machine.py가 하드코딩된 flat prior를 직접 넣으면 계층적 prior
    활성화 시 감사 로그가 실제 초기화값과 어긋난다).

    이미 정책이 존재하면 그 alpha/beta를 그대로 반환한다(method="existing").
    존재하지 않으면 settings.use_hierarchical_prior 플래그에 따라
    get_hierarchical_prior() 또는 균등 Beta(1,1)로 초기화한다.

    범위 제외(§5.4): "신뢰 붕괴 후 재초기화"(apply_feedback_update의
    new_beta > new_alpha*1.5 리셋 로직)는 계층적 prior 대상이 아니다 - 이건
    기존 조합의 리셋이지 §5.4가 말하는 "신규 조합 첫 등장"이 아니므로, 이
    함수는 오직 tb_policy_memory에 해당 (fruit_type, condition) 행 자체가
    없는 경우에만 신규 초기화 경로를 탄다.
    """
    existing = get_policy(fruit_type, condition)
    if existing is not None:
        return ThetaInit(
            alpha=existing["alpha"],
            beta=existing["beta"],
            method="existing",
            pooled_mean=None,
            pooled_sample_size=None,
        )

    if not settings.use_hierarchical_prior:
        return ThetaInit(
            alpha=settings.bayesian_prior_alpha,
            beta=settings.bayesian_prior_beta,
            method="uniform",
            pooled_mean=None,
            pooled_sample_size=None,
        )

    prior_alpha, prior_beta, pooled_mean, pooled_n = get_hierarchical_prior(fruit_type, condition)
    method = "hierarchical" if pooled_n > 0 else "hierarchical_fallback_uniform"
    return ThetaInit(
        alpha=prior_alpha,
        beta=prior_beta,
        method=method,
        pooled_mean=pooled_mean,
        pooled_sample_size=pooled_n,
    )


def apply_feedback_update(
    prev_alpha: float, prev_beta: float, prev_destination: str, destination: str
) -> tuple[float, float, str]:
    """사람 피드백 1건을 기존 (alpha, beta, destination)에 반영하는 순수함수.

    §5.2 리플레이(replay_session.py)가 실제 posterior 갱신과 동일한 결과를
    재현해야 하므로(고정 threshold/동적 EVPI 두 정책 모두 이 함수를 공유),
    DB I/O 없이 숫자 계산만 담당하도록 record_human_feedback()에서 분리했다.

    Returns:
        (new_alpha, new_beta, new_destination)
    """
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

    return new_alpha, new_beta, new_destination


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
    new_combo_theta_init: Optional[ThetaInit] = None

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
            # 처음 보는 조합 -> get_or_init_theta()로 prior 결정 (Track3, 균등 또는
            # 계층적). 이번 피드백을 첫 번째 "성공"으로 간주해 prior에 +1 반영
            theta_init = get_or_init_theta(fruit_type, condition)
            new_combo_theta_init = theta_init
            alpha = theta_init["alpha"] + 1.0
            beta = theta_init["beta"]
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

            new_alpha, new_beta, new_destination = apply_feedback_update(
                prev_alpha, prev_beta, prev_destination, destination
            )
            if new_destination != prev_destination and prev_destination != "ask_human":
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

    # 신규 조합이었으면, 이번 트랜잭션이 끝난 뒤(별도 커넥션의 쓰기 잠금과
    # 다투지 않도록) 초기화 시점의 순수 prior 값을 감사 로그에 남긴다.
    if new_combo_theta_init is not None:
        log_prior_initialized(
            fruit_type=fruit_type,
            condition=condition,
            alpha=new_combo_theta_init["alpha"],
            beta=new_combo_theta_init["beta"],
            init_method=new_combo_theta_init["method"],
            destination=destination,
            pooled_mean=new_combo_theta_init["pooled_mean"],
            pooled_sample_size=new_combo_theta_init["pooled_sample_size"],
        )

    # 최종 반영된 정책을 다시 조회해서 반환 (호출부가 바로 다음 행동을 결정할 수 있게)
    updated = get_policy(fruit_type, condition)
    assert updated is not None
    return updated
