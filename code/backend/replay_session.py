# Track 5-prep: 고정 threshold vs 동적 EVPI 베이스라인 비교 리플레이

"""
replay_session.py
====================
리서치 문서 §5.2 "베이스라인 비교 — 고정 threshold 0.65 vs 동적 EVPI"를 실제로
돌리는 리플레이 스크립트.

절차 (문서 §5.2 원문):
    "각 세트를 동적 EVPI 방식으로 1회 실행 -> 모든 판단 근거를 tb_decision_audit에
     기록 -> YOLO 통과로 자동 처리된 사과는 실험 종료 직후 육안으로 사후 라벨링 ->
     저장된 로그를 고정 threshold(0.65)와 동적 EVPI 두 방식으로 각각 재생(replay)해
     동일 물리적 사건에 대한 paired 비교 수행."

이 스크립트는 절차의 마지막 단계("저장된 로그를 두 방식으로 재생")만 담당한다.
실제 판정 게이트(services/decision_planner.py, services/human_query_gate.py)는
전혀 건드리지 않는다 - tb_decision_audit에 남은 관측치(x_t)와 사후 라벨
(actual_label)만 갖고 두 게이팅 규칙을 사후에 시뮬레이션한다.

핵심 모델링 가정 (문서에 명시되지 않아 이 스크립트가 자체적으로 내린 결정,
근거와 한계를 아래에 명시함):

1. 두 정책(고정 threshold, 동적 EVPI)은 각자 독립적인 alpha/beta 상태를
   유지하며 실제 tb_policy_memory를 전혀 건드리지 않는다(순수 시뮬레이션).
   상태는 세트 경계와 무관하게 전체 리플레이 동안 계속 이어진다(현실에서도
   정책 신뢰도는 세트가 바뀐다고 리셋되지 않으므로).

2. §5.2 리플레이는 "고정 threshold vs 동적 EVPI" 게이팅 규칙만 격리해서
   비교하기 위해, 두 정책 모두 신규 (fruit_type, condition) 조합에 항상 균등
   prior Beta(1,1)로 시작한다. §5.4 계층적 prior와의 상호작용은 여기서 다루지
   않는다(별도 ablation 대상 - 섞으면 두 변수의 효과가 뒤섞여 버린다).

3. "사람에게 물어보면 항상 진실(actual_label)에 맞는 정답을 준다"는 L2D의
   표준 오라클 전제를 그대로 채택한다. 즉 query_human=True로 판정된 사건의
   L(y,d) 손실은 항상 0이고, 그 대신 alpha/beta는 "사후 라벨이 가리키는 정답
   destination"으로 갱신된다. query_human=False(정책을 신뢰하고 바로 실행)로
   판정된 사건만 실제 오분류 손실이 발생할 수 있다.

4. Congestion 신호(n(t), tau_hold(t))는 실제 기록된 값을 "물리적 사실"로 그대로
   재사용한다 - 어느 게이팅 정책을 적용하든 그 순간 컨베이어가 얼마나 붐볐는지는
   바뀌지 않는다고 가정한다(반사실적으로 재시뮬레이션하지 않는다).

5. "세트 처리 총 소요 시간"은 로그의 실제 timestamp로부터 근사한다:
   - 사람을 부르는 경우의 소요 시간은, 로그에 실제로 기록된
     stage3_query_human -> stage3_human_resolved 시간차의 평균(세션이 FIFO로
     한 번에 하나씩만 처리된다는 hitl_state_machine.py의 설계를 전제로 순서
     기반 근사 매칭). 로그에 그런 사례가 전혀 없으면 config의
     hitl_response_timeout_sec(설정된 최대 응답 대기 시간)를 보수적 추정치로 사용.
   - 실행(사람 개입 없음)의 소요 시간은, 로그에서 "사람 호출로 지연되지 않은
     두 연속 사건" 사이의 실제 시간차 평균. 그런 사례가 없으면
     config의 vision_poll_interval_sec을 기본값으로 사용.
   이는 근사치이며, 두 정책이 실제로 다른 타이밍으로 물리 로봇을 움직였을
   때의 정확한 재현이 아니다.

사용법:
    cd code/backend && python3 replay_session.py [--items-per-set 6]
        [--fixed-threshold 0.65] [--session-ids ID1 ID2 ...]
"""

import argparse
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import database
from config import settings
from services.bayesian_policy import apply_feedback_update
from services.human_query_gate import should_query_human
from services.loss_matrix import L

logger = logging.getLogger(__name__)

# §5.2 리플레이 대상 branch(stage1/2 실 호출부가 아직 없으므로 - Track4 모듈
# docstring 참고 - 실질적으로 stage3_query_human/stage3_risk_accept_execute만
# 관측되지만, 향후 Stage1 fast-path가 추가될 경우를 대비해 포함해둔다)
DECISION_BRANCHES = ("stage1_auto_execute", "stage3_query_human", "stage3_risk_accept_execute")

FIXED_THRESHOLD_DEFAULT = 0.65
_INITIAL_ALPHA = settings.bayesian_prior_alpha
_INITIAL_BETA = settings.bayesian_prior_beta

CORRECT_DESTINATION_FOR_LABEL = {
    "normal": "normal_box",
    "ugly": "ugly_box",
    "processing": "processing_box",
    "rotten": "discard_box",
    "moldy": "discard_box",
}


@dataclass
class DecisionEvent:
    audit_id: int
    branch: str
    timestamp: datetime
    fruit_type: str
    condition: str
    n_pending: Optional[int]
    tau_hold_sec: Optional[float]
    actual_label: Optional[str]
    session_id: Optional[str]


@dataclass
class ReplayTally:
    """세트(또는 전체) 구간에서 한 정책이 누적한 §5.2 비교 지표 3종
    (+ 진단용 부가 정보)."""
    total_loss: float = 0
    total_time_sec: float = 0
    query_count: int = 0
    n_events: int = 0
    n_unlabeled: int = 0


@dataclass
class SetComparisonRow:
    set_index: int
    fixed: ReplayTally = field(default_factory=ReplayTally)
    dynamic: ReplayTally = field(default_factory=ReplayTally)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_decision_events(session_ids: Optional[list[str]] = None) -> list[DecisionEvent]:
    """tb_decision_audit에서 재생 대상 이벤트(DECISION_BRANCHES)를 timestamp
    오름차순으로 조회. session_ids가 주어지면 그 세션들로만 한정."""
    conn = database.get_db_connection()
    try:
        placeholders = ",".join("?" * len(DECISION_BRANCHES))
        query = f"""
            SELECT audit_id, branch, timestamp, fruit_type, condition,
                   n_pending, tau_hold_sec, actual_label, session_id
            FROM tb_decision_audit
            WHERE branch IN ({placeholders})
        """
        params = list(DECISION_BRANCHES)
        if session_ids:
            session_placeholders = ",".join("?" * len(session_ids))
            query += f" AND session_id IN ({session_placeholders})"
            params.extend(session_ids)
        query += " ORDER BY timestamp ASC, audit_id ASC"
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    return [
        DecisionEvent(
            audit_id=row["audit_id"],
            branch=row["branch"],
            timestamp=_parse_timestamp(row["timestamp"]),
            fruit_type=row["fruit_type"],
            condition=row["condition"],
            n_pending=row["n_pending"],
            tau_hold_sec=row["tau_hold_sec"],
            actual_label=row["actual_label"],
            session_id=row["session_id"],
        )
        for row in rows
    ]


def _load_resolved_timestamps() -> list[datetime]:
    conn = database.get_db_connection()
    try:
        rows = conn.execute(
            "SELECT timestamp FROM tb_decision_audit WHERE branch = 'stage3_human_resolved' ORDER BY timestamp ASC"
        ).fetchall()
    finally:
        conn.close()
    return [_parse_timestamp(row["timestamp"]) for row in rows]


def estimate_ask_duration_sec(query_timestamps: list[datetime], resolved_timestamps: list[datetime]) -> float:
    """stage3_query_human -> stage3_human_resolved 시간차 평균 근사.

    session_id로 직접 매칭할 수 없다(decide()가 세션 생성 전에 로그를 남기므로
    stage3_query_human 행엔 session_id가 없음). hitl_state_machine.py가 세션을
    한 번에 하나씩만(FIFO) 처리한다는 설계를 전제로, 정렬된 두 타임스탬프
    리스트를 순서대로 짝지어 "이 query 다음, 다음 query가 오기 전에 발생한 첫
    resolved"를 그 query의 처리 시간으로 근사한다.
    """
    if not query_timestamps or not resolved_timestamps:
        return settings.hitl_response_timeout_sec

    sorted_queries = sorted(query_timestamps)
    sorted_resolved = sorted(resolved_timestamps)
    durations = []
    ri = 0
    for qi, q_ts in enumerate(sorted_queries):
        next_q_ts = sorted_queries[qi + 1] if qi + 1 < len(sorted_queries) else None
        while ri < len(sorted_resolved) and sorted_resolved[ri] <= q_ts:
            ri += 1
        if ri < len(sorted_resolved) and (next_q_ts is None or sorted_resolved[ri] < next_q_ts):
            durations.append((sorted_resolved[ri] - q_ts).total_seconds())
            ri += 1

    if not durations:
        return settings.hitl_response_timeout_sec
    return statistics.mean(durations)


def estimate_execute_interval_sec(events: list[DecisionEvent]) -> float:
    """사람 호출로 지연되지 않은 두 연속 사건 사이의 실제 시간차 평균 근사
    (순수 컨베이어 처리 간격). events는 timestamp 오름차순으로 정렬되어 있어야 함."""
    intervals = []
    for prev, curr in zip(events, events[1:]):
        if prev.branch != "stage3_query_human" and curr.branch != "stage3_query_human":
            dt = (curr.timestamp - prev.timestamp).total_seconds()
            if dt >= 0:
                intervals.append(dt)
    if not intervals:
        return settings.vision_poll_interval_sec
    return statistics.mean(intervals)


def _fixed_threshold_query_human(entry: Optional[dict], threshold: float) -> bool:
    """과거 하드코딩 게이트 재현: policy 없음 또는 destination="ask_human" 또는
    confidence < threshold -> ask. (Track3 이전 decision_planner.py 로직 그대로,
    §5.2 베이스라인 비교 전용으로 이 스크립트 안에만 존재함)"""
    if entry is None:
        return True
    if entry["destination"] == "ask_human":
        return True
    confidence = entry["alpha"] / (entry["alpha"] + entry["beta"])
    return confidence < threshold


def _dynamic_evpi_query_human(
    entry: Optional[dict], fruit_type: str, condition: str, n_t: int, tau_hold_t: float
) -> bool:
    """Track2/3 실제 게이트(services.human_query_gate.should_query_human)를
    그대로 재사용 - 리서치 문서 §4.3.5 그 자체."""
    policy_dict = None
    if entry is not None:
        confidence = entry["alpha"] / (entry["alpha"] + entry["beta"])
        policy_dict = {
            "fruit_type": fruit_type,
            "condition": condition,
            "destination": entry["destination"],
            "alpha": entry["alpha"],
            "beta": entry["beta"],
            "confidence": confidence,
            "source": "replay_simulated",
            "updated_at": "",
        }
    query_human, _evpi, _cost = should_query_human(
        fruit_type, condition, policy_dict, n_t, tau_hold_t,
        settings.human_query_cost_k1, settings.human_query_cost_k2,
    )
    return query_human


def _apply_outcome(
    tally: ReplayTally,
    state: dict,
    key: tuple,
    entry: Optional[dict],
    event: DecisionEvent,
    query_human: bool,
    ask_duration_sec: float,
    execute_interval_sec: float,
) -> None:
    """한 이벤트에 대한 한 정책의 판정 결과(query_human)를 지표에 반영하고,
    query_human=True였다면 (모듈 docstring 3번 가정대로) 사람이 사후 라벨에
    맞는 정답을 줬다고 보고 시뮬레이션 상태(alpha/beta/destination)를 갱신한다."""
    tally.n_events += 1

    if query_human:
        tally.query_count += 1
        tally.total_time_sec += ask_duration_sec

        correct_destination = CORRECT_DESTINATION_FOR_LABEL.get(event.actual_label)
        if correct_destination is None:
            if event.actual_label is not None:
                logger.warning(
                    "알 수 없는 actual_label=%r (audit_id=%s) - 사후 라벨 없음과 동일하게 처리",
                    event.actual_label, event.audit_id,
                )
            tally.n_unlabeled += 1
            return

        if entry is not None:
            prev_alpha, prev_beta, prev_destination = entry["alpha"], entry["beta"], entry["destination"]
        else:
            prev_alpha, prev_beta, prev_destination = _INITIAL_ALPHA, _INITIAL_BETA, correct_destination

        new_alpha, new_beta, new_destination = apply_feedback_update(
            prev_alpha, prev_beta, prev_destination, correct_destination
        )
        state[key] = {"alpha": new_alpha, "beta": new_beta, "destination": new_destination}

    else:
        tally.total_time_sec += execute_interval_sec
        if event.actual_label is None:
            tally.n_unlabeled += 1
        else:
            tally.total_loss += L(event.actual_label, entry["destination"])


def replay(
    events: list[DecisionEvent],
    threshold: float = FIXED_THRESHOLD_DEFAULT,
    items_per_set: int = 6,
    ask_duration_sec: Optional[float] = None,
    execute_interval_sec: Optional[float] = None,
) -> list[SetComparisonRow]:
    """전체 이벤트 시퀀스를 한 번 순회하며 고정 threshold 정책과 동적 EVPI 정책을
    각각 독립적으로 시뮬레이션한다. 두 정책의 alpha/beta 상태는 세트 경계와
    무관하게 리플레이 전체 동안 이어지고, items_per_set개(§5.1 "1세트=6개")씩
    묶어 세트별 지표(paired 비교용)를 만든다.
    """
    if ask_duration_sec is None:
        query_timestamps = [e.timestamp for e in events if e.branch == "stage3_query_human"]
        ask_duration_sec = estimate_ask_duration_sec(query_timestamps, _load_resolved_timestamps())
    if execute_interval_sec is None:
        execute_interval_sec = estimate_execute_interval_sec(events)

    state_fixed: dict = {}
    state_dynamic: dict = {}
    set_rows: list[SetComparisonRow] = []
    current = SetComparisonRow(set_index=0)

    for i, event in enumerate(events):
        key = (event.fruit_type, event.condition)
        n_t = event.n_pending or 0
        tau_hold_t = event.tau_hold_sec or 0.0

        entry_fixed = state_fixed.get(key)
        query_fixed = _fixed_threshold_query_human(entry_fixed, threshold)
        _apply_outcome(
            current.fixed, state_fixed, key, entry_fixed, event, query_fixed,
            ask_duration_sec, execute_interval_sec,
        )

        entry_dynamic = state_dynamic.get(key)
        query_dynamic = _dynamic_evpi_query_human(entry_dynamic, event.fruit_type, event.condition, n_t, tau_hold_t)
        _apply_outcome(
            current.dynamic, state_dynamic, key, entry_dynamic, event, query_dynamic,
            ask_duration_sec, execute_interval_sec,
        )

        is_last_in_set = (i + 1) % items_per_set == 0 or i == len(events) - 1
        if is_last_in_set:
            set_rows.append(current)
            current = SetComparisonRow(set_index=len(set_rows))

    return set_rows


def _try_wilcoxon(fixed_losses: list[float], dynamic_losses: list[float]) -> str:
    """scipy.stats.wilcoxon을 시도하되, N=5 저검정력 한계를 항상 명시한다."""
    n = len(fixed_losses)
    if n < 1 or len(dynamic_losses) != n:
        return "Wilcoxon signed-rank test: 표본 부족으로 생략"

    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return "Wilcoxon signed-rank test: scipy 미설치로 생략"

    if all(f == d for f, d in zip(fixed_losses, dynamic_losses)):
        return "Wilcoxon signed-rank test: 모든 쌍의 차이가 0이라 계산 불가(두 정책 결과 완전히 동일)"

    try:
        statistic, p_value = scipy_stats.wilcoxon(fixed_losses, dynamic_losses)
    except ValueError as exc:
        return f"Wilcoxon signed-rank test: 계산 불가 ({exc})"

    return (
        f"Wilcoxon signed-rank test: statistic={statistic:.3f}, p={p_value:.4f} (N={n}"
        " - §5.1에서 미리 인지한 대로 저검정력 표본이라 이 p-value에 과도한 의미를 부여하지 말 것. "
        "유의하지 않다고 해서 '차이 없음'을 의미하지 않음)"
    )


def generate_report(set_rows: list[SetComparisonRow]) -> str:
    """paired 비교표 + wilcoxon 시도 + N=5 저검정력 한계를 리포트 본문에 무조건 출력."""
    lines = []
    lines.append("=" * 90)
    lines.append("§5.2 검증 리포트 — 고정 threshold(0.65) vs 동적 EVPI (paired 비교)")
    lines.append("=" * 90)

    header = (
        f"{'세트':>4} | {'손실(고정)':>10} | {'손실(동적)':>10} | "
        f"{'시간(고정,s)':>12} | {'시간(동적,s)':>12} | {'질문수(고정)':>10} | {'질문수(동적)':>10}"
    )
    lines.append(header)
    lines.append("-" * 90)

    fixed_losses, dynamic_losses = [], []
    for row in set_rows:
        fixed_losses.append(row.fixed.total_loss)
        dynamic_losses.append(row.dynamic.total_loss)
        lines.append(
            f"{row.set_index:>4} | {row.fixed.total_loss:>10.1f} | {row.dynamic.total_loss:>10.1f} | "
            f"{row.fixed.total_time_sec:>12.1f} | {row.dynamic.total_time_sec:>12.1f} | "
            f"{row.fixed.query_count:>10d} | {row.dynamic.query_count:>10d}"
        )
    lines.append("-" * 90)

    n = len(set_rows)
    lines.append(f"세트 수(N) = {n}")
    lines.append("")
    if n < 30:
        lines.append(
            f"! N={n}의 통계적 함의(§5.1): 이 정도 표본 수의 저검정력 표본으로는 정규성 가정 기반 "
            "검정(paired t-test)의 검정력이 극히 낮습니다. 아래 Wilcoxon signed-rank test는 참고용으로 '시도'하는 것이며, "
            "유의미한 p-value가 나오지 않을 수 있고 그 경우에도 이를 '차이 없음의 증거'로 해석해서는 "
            "안 됩니다. 주된 보고는 통계적 유의성 주장이 아니라 '관찰된 경향(observed trend)'의 정직한 "
            "서술이어야 합니다."
        )
        lines.append("")

    for label, fixed_vals, dynamic_vals in (
        ("총 손실 sum(L(y,d))", fixed_losses, dynamic_losses),
        ("세트 처리 총 소요 시간(초)", [r.fixed.total_time_sec for r in set_rows], [r.dynamic.total_time_sec for r in set_rows]),
        ("Stage3(사람 호출) 도달 횟수", [r.fixed.query_count for r in set_rows], [r.dynamic.query_count for r in set_rows]),
    ):
        fixed_mean = statistics.mean(fixed_vals) if fixed_vals else float("nan")
        dynamic_mean = statistics.mean(dynamic_vals) if dynamic_vals else float("nan")
        if dynamic_mean < fixed_mean:
            trend = "동적 EVPI가 더 낮음"
        elif dynamic_mean > fixed_mean:
            trend = "동적 EVPI가 더 높음"
        else:
            trend = "차이 없음"
        lines.append(f"[{label}]")
        lines.append(f"  평균(고정)={fixed_mean:.2f}, 평균(동적)={dynamic_mean:.2f}, 관찰된 경향: {trend}")

    lines.append("")
    lines.append("  " + _try_wilcoxon(fixed_losses, dynamic_losses))

    n_unlabeled_total = sum(r.fixed.n_unlabeled + r.dynamic.n_unlabeled for r in set_rows)
    if n_unlabeled_total > 0:
        lines.append("")
        lines.append(
            f"참고: 사후 라벨(actual_label)이 없는 사건 {n_unlabeled_total}건은 손실 계산에서 "
            "제외되었습니다 (§5.2 사후 라벨링이 아직 안 된 로그이거나, 데모/테스트용 더미 데이터)."
        )

    lines.append("")
    lines.append(
        "한계(§6): congestion은 정책별 재시뮬레이션 없이 기록된 사실 그대로 재생했다 - "
        "고정-threshold 반사실 정책이 실제로는 다른 시점에 질문했을 경우 혼잡도 자체가 "
        "달라졌을 수 있다는 근사임을 밝힌다."
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="§5.2 검증 리플레이: tb_decision_audit 로그를 고정 threshold(0.65)와 동적 EVPI 두 방식으로 재생해 paired 비교 리포트를 출력한다."
    )
    parser.add_argument("--items-per-set", type=int, default=6, help="세트당 개체 수 (기본 6, §5.1)")
    parser.add_argument("--fixed-threshold", type=float, default=FIXED_THRESHOLD_DEFAULT, help="과거 하드코딩 게이트의 confidence 임계값 (기본 0.65)")
    parser.add_argument("--session-ids", nargs="*", default=None, help="특정 세션(session_id)들만 재생하고 싶을 때 지정. 생략하면 전체 로그 사용")
    args = parser.parse_args()

    events = load_decision_events(args.session_ids)
    if not events:
        print("tb_decision_audit에 재생할 판정 이벤트가 없습니다.")
        return

    set_rows = replay(events, threshold=args.fixed_threshold, items_per_set=args.items_per_set)
    print(generate_report(set_rows))


if __name__ == "__main__":
    main()
