# Track 4: tb_decision_audit append-only 감사 로그 인프라

"""
decision_audit.py
===================
tb_decision_audit에 append-only로 판정 근거를 기록하는 유일한 창구.

리서치 문서 §8 Implementation Status (Track 3 EVPI 게이트 구현 시점 기준 갱신):
    "Track 4 — 감사 로그 (tb_decision_audit): 설계/수식 확정 ✅, 코드 구현 완료"
    "Track 2 — EVPI 동적 threshold(§4.3): 설계/수식 확정 ✅, 코드 구현 완료
     (services/human_query_gate.py, decision_planner.py에 배선됨)"

지금 실제로 호출되는 지점(코드에 이미 존재하는 분기):
    - log_stage3_query_human(): decision_planner.decide()가 action="ask_human"을
      반환하는 지점. Vision 자체가 확신 못 하는 조기 반환(Stage1 실패, Stage2 VLM
      부재로 곧장 Stage3의 무조건 질문 케이스로 처리)과, Stage3 게이트(§4.3.5)가
      query_human=True를 결정한 경우 둘 다 여기로 옴.
    - log_stage3_risk_accept_execute(): decision_planner.decide()가 Stage3
      게이트(§4.3.5)에서 query_human=False(EVPI_human <= Cost_human(t))로 판단해
      action="execute"를 반환하는 지점 - "위험 감수 실행"
    - log_stage3_human_resolved(): hitl_state_machine.py가
      bayesian_policy.record_human_feedback()으로 posterior를 갱신한 직후
    - log_prior_initialized(): bayesian_policy.record_human_feedback()이 신규
      (fruit_type, condition) 조합의 theta를 처음 초기화하는 지점 (Track 3)

아직 코드 상 분기 자체가 존재하지 않아 호출부가 없는 함수(Track 1/2 구현 후 배선 예정):
    - log_stage2_vlm_call(): GPT-4o Vision 호출 분기 자체가 아직 없음. §4.4가
      정의하는 EVSI_VLM(x_t)/Cost_VLM 계산식이 문서에 없고(다이어그램 레이블만
      존재), Track 1(이미지 파이프라인)도 §8에 "Vision팀의 이미지 제공 방식 확정
      및 데이터 정책 승인 대기"로 명시되어 있어 지금은 만들 근거가 없다.
    - log_stage1_auto_execute(): "Stage1 fast-path가 EVPI 계산 자체를 생략하고
      곧장 정책을 실행"하는 별도 코드 경로가 이 프로젝트엔 없다 (Stage1의
      vision.confidence 게이트를 통과한 뒤에도 항상 Stage3 EVPI 게이트를 거침).
      실제로 그런 fast-path 분기가 추가되면 이 함수를 그 경로에서 호출하면 된다.

append-only 보장: 이 파일에는 UPDATE/DELETE 구문이 전혀 없다. 모든 함수는
_insert_audit_row() 하나만 거쳐 INSERT만 수행하며, tb_decision_audit에는
갱신 가능한 자연키(예: tb_policy_memory의 (fruit_type, condition) 복합 PK 같은)가
없다 - audit_id는 AUTOINCREMENT라 항상 새 행만 추가된다.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from database import get_db_connection
from models import VisionFeatureIn

logger = logging.getLogger(__name__)

# §4.2가 "Calibration 결과 의존적 재도출 필요"라 명시한 잠정치(placeholder).
# 실제 Stage1 게이트가 쓰는 settings.confidence_threshold와는 다른 값일 수
# 있음 - 이 값은 오직 감사 로그의 stage1_gate_passed 계산/기록용이다.
TAU_YOLO_STAGE1_AUDIT = 0.9

BRANCH_STAGE1_AUTO_EXECUTE = "stage1_auto_execute"
BRANCH_STAGE2_VLM_CALL = "stage2_vlm_call"
BRANCH_STAGE3_QUERY_HUMAN = "stage3_query_human"
BRANCH_STAGE3_RISK_ACCEPT_EXECUTE = "stage3_risk_accept_execute"
BRANCH_STAGE3_HUMAN_RESOLVED = "stage3_human_resolved"
BRANCH_PRIOR_INITIALIZED = "prior_initialized"

VALID_BRANCHES = {
    BRANCH_STAGE1_AUTO_EXECUTE,
    BRANCH_STAGE2_VLM_CALL,
    BRANCH_STAGE3_QUERY_HUMAN,
    BRANCH_STAGE3_RISK_ACCEPT_EXECUTE,
    BRANCH_STAGE3_HUMAN_RESOLVED,
    BRANCH_PRIOR_INITIALIZED,
}

_FINAL_ACTIONS = {"execute", "hold"}


def _bool_to_int(value: Optional[bool]) -> Optional[int]:
    """SQLite에는 bool 타입이 없어 0/1/NULL로 저장. None은 '아직 평가 안 됨'을 의미."""
    if value is None:
        return None
    return 1 if value else 0


def _int_to_bool(value: Optional[int]) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _insert_audit_row(
    *,
    branch: str,
    fruit_type: str,
    condition: str,
    final_action: str,
    c_yolo: Optional[float] = None,
    defect_type: Optional[str] = None,
    position: Optional[list] = None,
    stage1_gate_passed: Optional[bool] = None,
    stage2_evsi_gate_passed: Optional[bool] = None,
    vlm_called: Optional[bool] = None,
    vlm_response: Optional[dict] = None,
    theta_exists: Optional[bool] = None,
    dstored: Optional[str] = None,
    evpi_human: Optional[float] = None,
    cost_human: Optional[float] = None,
    n_pending: Optional[int] = None,
    tau_hold_sec: Optional[float] = None,
    query_human: Optional[bool] = None,
    final_destination: Optional[str] = None,
    alpha_before: Optional[float] = None,
    beta_before: Optional[float] = None,
    alpha_after: Optional[float] = None,
    beta_after: Optional[float] = None,
    actual_label: Optional[str] = None,
    session_id: Optional[str] = None,
    prior_init_method: Optional[str] = None,
    prior_pooled_mean: Optional[float] = None,
    prior_pooled_sample_size: Optional[int] = None,
) -> int:
    """tb_decision_audit에 행 1건을 append하는 유일한 저수준 함수.

    이 파일 안의 모든 log_*() 함수는 반드시 이 함수를 거쳐야 하며, 이 함수는
    INSERT 외의 SQL을 실행하지 않는다 (append-only 보장의 근거).
    """
    if branch not in VALID_BRANCHES:
        raise ValueError(f"알 수 없는 branch: {branch!r} (VALID_BRANCHES={VALID_BRANCHES})")
    if final_action not in _FINAL_ACTIONS:
        raise ValueError(f"알 수 없는 final_action: {final_action!r} ({_FINAL_ACTIONS} 중 하나여야 함)")

    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO tb_decision_audit (
                branch, timestamp, fruit_type, condition,
                c_yolo, defect_type, position, tau_yolo, stage1_gate_passed,
                stage2_evsi_gate_passed, vlm_called, vlm_response,
                theta_exists, dstored, evpi_human, cost_human, n_pending, tau_hold_sec,
                query_human, final_action, final_destination,
                alpha_before, beta_before, alpha_after, beta_after,
                actual_label, session_id,
                prior_init_method, prior_pooled_mean, prior_pooled_sample_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                branch, timestamp, fruit_type, condition,
                c_yolo, defect_type, json.dumps(position) if position is not None else None,
                TAU_YOLO_STAGE1_AUDIT, _bool_to_int(stage1_gate_passed),
                _bool_to_int(stage2_evsi_gate_passed), _bool_to_int(vlm_called),
                # ensure_ascii=False: vlm_response는 Track6부터 GPT-4o가 식별한 한국어
                # 물체명을 담을 수 있음 - 기본값(True)이면 \uXXXX로 이스케이프되어
                # DB를 직접 조회할 때 가독성이 크게 떨어짐.
                json.dumps(vlm_response, ensure_ascii=False) if vlm_response is not None else None,
                _bool_to_int(theta_exists), dstored, evpi_human, cost_human, n_pending, tau_hold_sec,
                _bool_to_int(query_human), final_action, final_destination,
                alpha_before, beta_before, alpha_after, beta_after,
                actual_label, session_id,
                prior_init_method, prior_pooled_mean, prior_pooled_sample_size,
            ),
        )
        conn.commit()
        audit_id = cursor.lastrowid
    finally:
        conn.close()

    logger.info(
        "감사 로그 기록 (audit_id=%d, branch=%s): fruit=%s condition=%s final_action=%s",
        audit_id, branch, fruit_type, condition, final_action,
    )
    return audit_id


def get_audit_log_by_id(audit_id: int) -> Optional[dict]:
    """단건 조회 (테스트/디버깅용). position, vlm_response는 JSON 문자열을 다시
    파이썬 객체로, stage1_gate_passed 등 0/1 컬럼은 bool로 역변환해서 돌려준다."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tb_decision_audit WHERE audit_id = ?", (audit_id,))
        row = cursor.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    record = dict(row)
    record["position"] = json.loads(record["position"]) if record["position"] is not None else None
    record["vlm_response"] = json.loads(record["vlm_response"]) if record["vlm_response"] is not None else None
    for bool_field in ("stage1_gate_passed", "stage2_evsi_gate_passed", "vlm_called", "theta_exists", "query_human"):
        record[bool_field] = _int_to_bool(record[bool_field])
    return record


def log_stage1_auto_execute(vision: VisionFeatureIn, condition: str, dstored: str) -> int:
    """Stage1 자동처리 분기.

    문서 아키텍처 다이어그램(§1.2): "1. YOLO Fast-Path (Stage 1, tau_yolo)
    c_yolo(x) > tau_yolo ? -- Yes -> execute(policy_lookup(x))".
    decision_planner.decide()가 action="execute"를 반환하는 지점에서 호출한다.
    (현재 코드에는 이 fast-path 분기 자체가 없음 - 모듈 docstring 참고. 향후
    추가되면 그 경로에서 호출한다.)
    """
    return _insert_audit_row(
        branch=BRANCH_STAGE1_AUTO_EXECUTE,
        fruit_type=vision.fruit_type,
        condition=condition,
        c_yolo=vision.confidence,
        defect_type=vision.defect_type,
        position=vision.center,
        stage1_gate_passed=vision.confidence > TAU_YOLO_STAGE1_AUDIT,
        theta_exists=True,
        dstored=dstored,
        query_human=False,
        final_action="execute",
        final_destination=dstored,
    )


def log_stage2_vlm_call(
    vision: VisionFeatureIn,
    condition: str,
    stage2_evsi_gate_passed: bool,
    vlm_response: Optional[dict],
    final_action: str,
    final_destination: Optional[str] = None,
) -> int:
    """Stage2 VLM 호출 분기 (§4.4).

    "EVSI_VLM(x_t) > Cost_VLM일 때만 GPT-4o Vision 호출. YOLO와 상관된 신호이므로,
     VLM의 답변은 theta_f,c를 직접 갱신하지 않고 tb_decision_audit에 기록 후 일정
     비율 감사 샘플링으로만 사후 검증한다."

    Track 1/2(Vision-LLM 멀티모달, EVSI 동적 threshold)가 아직 "대기" 상태(§8)라
    현재 코드에는 이 분기의 실제 호출부가 없다. Track 2 구현 시 이 함수를
    그 게이트 로직 안에서 호출하면 된다.
    """
    return _insert_audit_row(
        branch=BRANCH_STAGE2_VLM_CALL,
        fruit_type=vision.fruit_type,
        condition=condition,
        c_yolo=vision.confidence,
        defect_type=vision.defect_type,
        position=vision.center,
        stage1_gate_passed=vision.confidence > TAU_YOLO_STAGE1_AUDIT,
        stage2_evsi_gate_passed=stage2_evsi_gate_passed,
        vlm_called=True,
        vlm_response=vlm_response,
        query_human=final_action == "hold",
        final_action=final_action,
        final_destination=final_destination,
    )


def log_stage3_query_human(
    vision: VisionFeatureIn,
    condition: str,
    theta_exists: bool,
    dstored: Optional[str],
    session_id: Optional[str] = None,
    evpi_human: Optional[float] = None,
    cost_human: Optional[float] = None,
    n_pending: Optional[int] = None,
    tau_hold_sec: Optional[float] = None,
) -> int:
    """Stage3 사람 호출 분기 (§4.3.5 "query_human <=> ... (EVPI > Cost)").

    decision_planner.decide()가 action="ask_human"을 반환하는 지점에서 호출한다.
    evpi_human/cost_human/n_pending/tau_hold_sec은 services/human_query_gate.py
    (§4.3.3 EVPI_human, §4.3.4 Cost_human(t))가 계산한 실제 값이 채워진다.
    단, Vision 자체가 확신 못 해 정책 조회 전에 조기 반환하는 Stage1 실패 경로는
    EVPI/Cost 계산 없이 무조건 질문하므로 이 값들이 None으로 남는다 (감사 목적의
    theta_exists/dstored만 채워짐).
    """
    return _insert_audit_row(
        branch=BRANCH_STAGE3_QUERY_HUMAN,
        fruit_type=vision.fruit_type,
        condition=condition,
        c_yolo=vision.confidence,
        defect_type=vision.defect_type,
        position=vision.center,
        stage1_gate_passed=vision.confidence > TAU_YOLO_STAGE1_AUDIT,
        theta_exists=theta_exists,
        dstored=dstored,
        evpi_human=evpi_human,
        cost_human=cost_human,
        n_pending=n_pending,
        tau_hold_sec=tau_hold_sec,
        query_human=True,
        final_action="hold",
        final_destination=None,
        session_id=session_id,
    )


def log_stage3_risk_accept_execute(
    vision: VisionFeatureIn,
    condition: str,
    dstored: str,
    evpi_human: Optional[float] = None,
    cost_human: Optional[float] = None,
    n_pending: Optional[int] = None,
    tau_hold_sec: Optional[float] = None,
) -> int:
    """Stage3 위험 감수 실행 분기 (§1.2 "EVPI_human(theta_f,c) > Cost_human(t)? --
    No -> execute (위험 감수, 명시적 로깅)").

    EVPI_human <= Cost_human(t)이면, 사람에게 묻는 비용(대기열 혼잡도/HOLD
    지속시간에 따라 커짐, §4.3.4)이 오분류 기대 손실보다 커서 위험을 감수하고
    바로 실행하는 분기. decision_planner.decide()가 services.human_query_gate.
    should_query_human()의 결과로 query_human=False를 받았을 때 호출한다.
    """
    return _insert_audit_row(
        branch=BRANCH_STAGE3_RISK_ACCEPT_EXECUTE,
        fruit_type=vision.fruit_type,
        condition=condition,
        c_yolo=vision.confidence,
        defect_type=vision.defect_type,
        position=vision.center,
        stage1_gate_passed=vision.confidence > TAU_YOLO_STAGE1_AUDIT,
        theta_exists=True,
        dstored=dstored,
        evpi_human=evpi_human,
        cost_human=cost_human,
        n_pending=n_pending,
        tau_hold_sec=tau_hold_sec,
        query_human=False,
        final_action="execute",
        final_destination=dstored,
    )


def log_stage3_human_resolved(
    fruit_type: str,
    condition: str,
    destination: str,
    alpha_before: float,
    beta_before: float,
    alpha_after: float,
    beta_after: float,
    session_id: Optional[str] = None,
    c_yolo: Optional[float] = None,
    position: Optional[list] = None,
) -> int:
    """Stage3 사람 호출이 실제로 답변을 받아 posterior가 갱신된 시점의 후속 기록.

    §1.2 "Beta(alpha,beta) posterior 갱신 -> RESUME" 단계.
    hitl_state_machine.py가 bayesian_policy.record_human_feedback() 호출 직후
    (갱신 전/후 alpha, beta를 모두 확보한 시점)에 호출한다.
    posterior 갱신 전/후 필드는 "사람이 답변한 경우만 값이 채워지고, 그 외엔
    null"이라는 요청대로 이 분기에서만 채워진다 (다른 log_* 함수는 이 인자들을
    받지 않아 항상 NULL로 저장됨).

    actual_label(Track5 Calibration 준비): 사람의 답(destination)을 "잠정 정답"으로
    간주해 그대로 기록한다. append-only 원칙상 이 판정을 유발한 원본
    stage3_query_human 행을 소급 수정할 수는 없으므로(그 행에는 영원히
    actual_label=NULL로 남음), 결과가 확정되는 이 시점의 행에 기록하는 것으로
    "사람이 검토한 사례에는 최소 하나의 정답 라벨이 어딘가에 남는다"를 보장한다.
    한계: risk_accept_execute(사람에게 안 묻고 자동 실행)는 애초에 사람 확인이
    없으므로 이 함수가 호출되지 않고, 그런 행들의 actual_label은 계속 NULL로
    남는다 - "자동 실행된 판정은 사후 검증 수단이 없다"는 사실 자체가 정확히
    반영되는 것이지 버그가 아니다.
    """
    return _insert_audit_row(
        branch=BRANCH_STAGE3_HUMAN_RESOLVED,
        fruit_type=fruit_type,
        condition=condition,
        c_yolo=c_yolo,
        position=position,
        theta_exists=True,
        dstored=destination,
        query_human=True,
        final_action="execute",
        final_destination=destination,
        alpha_before=alpha_before,
        beta_before=beta_before,
        alpha_after=alpha_after,
        beta_after=beta_after,
        actual_label=destination,
        session_id=session_id,
    )


def log_prior_initialized(
    fruit_type: str,
    condition: str,
    alpha: float,
    beta: float,
    init_method: str,
    destination: str,
    pooled_mean: Optional[float] = None,
    pooled_sample_size: Optional[int] = None,
) -> int:
    """§5.4 계층적 Prior Ablation 검증용 기록.

    "신규 과일 첫 판단 시점의 confidence 값"(§5.4 실험 계획)을 재구성할 수 있도록,
    신규 (fruit_type, condition) 조합의 theta가 처음 초기화되는 순간의
    (alpha, beta, 초기화 방식)을 append-only로 남긴다. 여기 기록되는 alpha/beta는
    이번 피드백 자체를 반영하기 *전*의 순수 prior 값이다 (그 이후 +1 반영된 최종
    값은 log_stage3_human_resolved()의 alpha_after/beta_after에 별도로 남음).

    bayesian_policy.record_human_feedback()의 existing is None 분기가, 자신의
    SQLite 트랜잭션을 commit/close한 *뒤에* 호출한다 - 이 함수가 여는 별도의
    커넥션이 아직 끝나지 않은 그 트랜잭션과 쓰기 잠금을 다투지 않도록 하기 위함.

    init_method: "uniform"(§4.3.3 균등 prior Beta(1,1)) 또는 "hierarchical"
    (§5.4 계층적 prior, pooled_mean/pooled_sample_size로 근거 확인 가능) 또는
    "hierarchical_fallback_uniform"(계층적 prior를 시도했으나 pooled 표본이 0개라
    균등 prior로 폴백한 경우) 중 하나.

    한계(§5.4/§6): "본 연구는 계층적 prior 검증이 소규모 사례 시연이며 일반화
    근거로는 부족하다."
    """
    return _insert_audit_row(
        branch=BRANCH_PRIOR_INITIALIZED,
        fruit_type=fruit_type,
        condition=condition,
        theta_exists=False,
        dstored=destination,
        final_action="execute",
        final_destination=destination,
        alpha_after=alpha,
        beta_after=beta,
        prior_init_method=init_method,
        prior_pooled_mean=pooled_mean,
        prior_pooled_sample_size=pooled_sample_size,
    )
