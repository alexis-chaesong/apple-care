# Task2: 사과 1개씩 순차 실물 드라이런 - 사후 검증 스크립트
#
# 드라이런이 끝난 뒤 그 세션의 tb_decision_audit 로그를 모아서,
#   1) 스키마 완전성 - branch별로 채워져 있어야 할 필드가 실제로 채워졌는지
#   2) Stage 1/2/3 중 어느 경로를 탔는지 요약
# 을 사람이 읽기 좋게 출력한다.
#
# 범위 밖: 이 스크립트는 "사과 1개를 순차적으로 처리"하는 드라이런에서 나온
# 로그를 전제로 요약한다. 여러 물체가 동시에 처리돼 audit_id 순서와 물리적
# 사건 순서가 어긋날 수 있는 다중 아이템 시나리오의 순서 재구성은 다루지
# 않는다 (§5.1 본 실험에서 별도로 필요).
#
# 사용법:
#   cd code/backend && python3 scripts/dry_run_verify.py
#   cd code/backend && python3 scripts/dry_run_verify.py --min-audit-id 42
#   cd code/backend && python3 scripts/dry_run_verify.py --since "2026-07-15T10:00:00"

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import database
from services.decision_audit import VALID_BRANCHES

SEPARATOR = "=" * 90

# branch별로 "이 값이 null이면 이상하다"고 볼 수 있는 필드 목록.
# stage3_query_human은 의도적으로 제외 - decision_planner.py의 Stage1(vision.confidence
# 낮음) 조기 반환 경로도 같은 branch로 기록되는데, 그 경로는 EVPI/Cost 계산 자체를
# 안 하므로 evpi_human/cost_human/n_pending/tau_hold_sec이 전부 null인 게 정상이다
# (재발방지 - 스펙 문서에 이미 명시된 특성, 이 스크립트가 오탐으로 잡으면 안 됨).
EXPECTED_NON_NULL_FIELDS = {
    "stage2_vlm_call": ["vlm_called", "final_action"],
    "stage3_risk_accept_execute": [
        "theta_exists", "dstored", "evpi_human", "cost_human", "final_destination",
    ],
    "stage3_human_resolved": [
        "alpha_before", "beta_before", "alpha_after", "beta_after", "final_destination",
    ],
    "prior_initialized": ["prior_init_method", "alpha_after", "beta_after"],
}

# "이 branch에 도달했다" = 대략 이 Stage를 거쳤다는 뜻으로 보여주기 위한 매핑.
# Stage1 fast-path(log_stage1_auto_execute)는 이 프로젝트에 실 호출부가 없으므로
# (Track4 "재발 방지" 항목) 여기 나타나면 오히려 그 자체가 특이사항이다.
BRANCH_TO_STAGE_LABEL = {
    "stage1_auto_execute": "Stage1 fast-path (이 프로젝트엔 원래 없어야 하는 경로!)",
    "stage2_vlm_call": "Stage2 (VLM 식별 시도)",
    "stage3_query_human": "Stage3 - 사람에게 확인(HOLD)",
    "stage3_risk_accept_execute": "Stage3 - 위험 감수 실행",
    "stage3_human_resolved": "Stage3 - 사람 답변 반영(posterior 갱신)",
    "prior_initialized": "Track3 - 신규 조합 prior 초기화",
}


def _load_rows(min_audit_id: Optional[int], since: Optional[str], max_audit_id: Optional[int]) -> list[sqlite3.Row]:
    conn = database.get_db_connection()
    try:
        query = "SELECT * FROM tb_decision_audit WHERE 1=1"
        params: list = []
        if min_audit_id is not None:
            query += " AND audit_id >= ?"
            params.append(min_audit_id)
        if max_audit_id is not None:
            query += " AND audit_id <= ?"
            params.append(max_audit_id)
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)
        query += " ORDER BY audit_id"
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def _print_row_table(rows: list[sqlite3.Row]) -> None:
    print(SEPARATOR)
    print(f"이번 세션 판정 로그 ({len(rows)}건)")
    print(SEPARATOR)
    header = f"{'audit_id':>8}  {'timestamp':<28}  {'branch':<28}  {'fruit_type':<18}  {'condition':<10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        d = dict(row)
        print(
            f"{d['audit_id']:>8}  {d['timestamp']:<28}  {d['branch']:<28}  "
            f"{str(d['fruit_type']):<18}  {str(d['condition']):<10}"
        )


def _check_branch_validity(rows: list[sqlite3.Row]) -> list[str]:
    """모든 branch 값이 VALID_BRANCHES(services/decision_audit.py) 안에 있는지 -
    스키마가 변경됐는데 이 스크립트가 안 따라간 경우를 조기에 알아채기 위함."""
    problems = []
    for row in rows:
        if row["branch"] not in VALID_BRANCHES:
            problems.append(
                f"audit_id={row['audit_id']}: 알 수 없는 branch={row['branch']!r} "
                f"(services/decision_audit.VALID_BRANCHES에 없음 - 스키마 불일치 의심)"
            )
    return problems


def _check_schema_completeness(rows: list[sqlite3.Row]) -> list[str]:
    """branch별로 채워져 있어야 할 필드가 실제로 null이 아닌지 확인."""
    problems = []
    for row in rows:
        d = dict(row)
        expected_fields = EXPECTED_NON_NULL_FIELDS.get(d["branch"], [])
        for field in expected_fields:
            if d.get(field) is None:
                problems.append(
                    f"audit_id={d['audit_id']} branch={d['branch']!r}: "
                    f"'{field}'가 null - 이 branch에서는 채워져 있어야 정상"
                )
    return problems


def _check_final_action_consistency(rows: list[sqlite3.Row]) -> list[str]:
    """query_human=True인데 final_action='execute'거나, query_human=False인데
    final_action='hold'인 경우처럼 논리적으로 모순되는 조합을 찾는다."""
    problems = []
    for row in rows:
        d = dict(row)
        if d["query_human"] is None or d["final_action"] is None:
            continue
        query_human = bool(d["query_human"])
        if query_human and d["final_action"] == "execute" and d["branch"] != "stage3_human_resolved":
            # stage3_human_resolved는 "이 사건이 애초에 query_human=True로 갔었다"는
            # 이력 표시로 query_human=True + final_action=execute 조합이 정상이다
            # (사람이 답변한 뒤 execute 되는 것이므로) - 그 외 branch에서만 이상함.
            problems.append(
                f"audit_id={d['audit_id']} branch={d['branch']!r}: "
                f"query_human=True인데 final_action='execute' (모순 의심)"
            )
        if not query_human and d["final_action"] == "hold":
            problems.append(
                f"audit_id={d['audit_id']} branch={d['branch']!r}: "
                f"query_human=False인데 final_action='hold' (모순 의심)"
            )
    return problems


def _summarize_stage_path(rows: list[sqlite3.Row]) -> None:
    print(SEPARATOR)
    print("Stage 1/2/3 경로 요약")
    print(SEPARATOR)

    if not rows:
        print("(로그 없음)")
        return

    branch_counts: dict[str, int] = {}
    for row in rows:
        branch_counts[row["branch"]] = branch_counts.get(row["branch"], 0) + 1

    for branch, count in branch_counts.items():
        label = BRANCH_TO_STAGE_LABEL.get(branch, branch)
        print(f"  {label}: {count}건")

    print()
    # 대표적인 흐름 서술 - 이번 세션에 어떤 branch들이 나타났는지로 대략적인
    # 내러티브를 만들어준다 (여러 물체가 섞였을 수 있으니 "정확한 순서"라기보다
    # "이런 일들이 있었다" 수준의 요약).
    saw_vlm = "stage2_vlm_call" in branch_counts
    saw_query = "stage3_query_human" in branch_counts
    saw_risk_accept = "stage3_risk_accept_execute" in branch_counts
    saw_resolved = "stage3_human_resolved" in branch_counts
    saw_stage1_fastpath = "stage1_auto_execute" in branch_counts

    if saw_stage1_fastpath:
        print("  ⚠ stage1_auto_execute가 나타났습니다 - 이 프로젝트엔 원래 이 경로가")
        print("    없어야 합니다(Track4 '재발 방지' 항목). decision_planner.py가")
        print("    바뀌었는지 확인이 필요합니다.")

    if saw_vlm:
        print("  - unknown 물체에 대해 VLM(GPT-4o) 식별이 최소 1회 시도됨")
    if saw_query:
        print("  - Stage3 게이트가 최소 1회 사람에게 확인을 요청함(HOLD)")
    if saw_resolved:
        print("  - 그 중 최소 1건은 사람이 실제로 답변해서 posterior가 갱신됨")
    elif saw_query:
        print("  - 다만 stage3_human_resolved가 없음 - 질문은 갔지만 아직 사람이")
        print("    답변을 안 했거나(세션 진행 중), STUCK 상태로 남았을 가능성")
    if saw_risk_accept:
        print("  - Stage3 게이트가 최소 1회 '위험 감수 실행'으로 자동 처리함")
    if not (saw_vlm or saw_query or saw_risk_accept or saw_resolved or saw_stage1_fastpath):
        print("  - 판정 관련 branch가 하나도 없음 - 드라이런이 실제로 실행됐는지")
        print("    확인 필요 (또는 --min-audit-id/--since 범위가 잘못됐을 수 있음)")


def main() -> None:
    parser = argparse.ArgumentParser(description="드라이런 세션의 tb_decision_audit 로그 사후 검증")
    parser.add_argument("--min-audit-id", type=int, default=None, help="이 audit_id 이상만 검사")
    parser.add_argument("--max-audit-id", type=int, default=None, help="이 audit_id 이하만 검사")
    parser.add_argument("--since", default=None, help="이 timestamp(ISO) 이후 행만 검사 (예: 2026-07-15T10:00:00)")
    args = parser.parse_args()

    rows = _load_rows(args.min_audit_id, args.since, args.max_audit_id)

    print(f"드라이런 사후 검증 (DB={database.DB_PATH})")
    print(
        "주의: 사과 1개 순차 처리 전제 - 다중 아이템 동시 처리 시나리오의 순서 재구성은 "
        "이 스크립트의 검증 범위 밖입니다."
    )

    _print_row_table(rows)

    problems = []
    problems += _check_branch_validity(rows)
    problems += _check_schema_completeness(rows)
    problems += _check_final_action_consistency(rows)

    print(SEPARATOR)
    print("스키마 완전성 / 논리 일관성 검사")
    print(SEPARATOR)
    if problems:
        for p in problems:
            print(f"  ⚠ {p}")
    else:
        print("  이상 없음 - 모든 행이 branch별 기대 필드를 채우고 있고, "
              "query_human/final_action 조합도 일관됩니다.")

    _summarize_stage_path(rows)

    print(SEPARATOR)
    print(f"최종 요약: {len(rows)}건 중 이상 {len(problems)}건 발견")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
