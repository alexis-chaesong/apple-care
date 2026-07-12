#!/usr/bin/env python3
# Track5: Calibration 인프라 (통계 분석/결론은 아직 내리지 않음 - 데이터 수집/계산 파이프라인만)

"""
scripts/calibration_report.py
================================
tb_decision_audit + tb_human_feedback을 조인해 "정책이 예측한 확률(p)"과
"사람이 실제로 준 정답(actual_label)"의 쌍을 뽑아 ECE(Expected Calibration
Error)/Brier score를 계산하고, 데이터가 충분하면 reliability diagram을 그린다.

핵심 설계: p는 새로 계산하지 않고 반드시 services.loss_matrix.L_error()를
그대로 재사용해서 역산한다 (evpi_human = (1-p)*L_error(c,dstored)라는 §4.3.3
공식의 역함수) - 캘리브레이션 스크립트가 게이트 로직과 다른 공식을 쓰면 애초에
"뭘 캘리브레이션하는지"가 어긋나므로 반드시 같은 소스를 참조해야 함.

짝짓기(pairing) 방법: tb_decision_audit는 append-only라 stage3_query_human
행에 session_id가 채워지지 않는다(hitl_state_machine.py가 세션을 아직 만들기
전에 이 행이 먼저 기록되기 때문). 그래서 각 stage3_human_resolved 행(사람이
실제로 답한 시점, actual_label 보유)에 대해 "같은 (fruit_type, condition)이고
이 시점보다 먼저 기록된 가장 최근 stage3_query_human 행"을 그 판정을 유발한
질문으로 간주해 짝짓는다. 이 휴리스틱의 한계: 같은 (fruit_type, condition)
조합에 대해 사람 응답을 기다리는 동안 다른 판정이 끼어들면 잘못 짝지어질 수
있음 - 표본이 늘어나면 session_id를 query_human 행에도 채우는 방향으로
개선하는 게 근본적 해결책(지금은 인프라만 준비하는 단계라 남겨둠).

이 스크립트는 "무엇이 맞다/틀렸다"는 결론을 내리지 않는다 - 계산 결과와 함께
표본 크기 경고를 항상 출력하고, 판단은 사람에게 맡긴다.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import DB_PATH  # noqa: E402
from services.loss_matrix import L_error  # noqa: E402

# 표본 크기가 이 미만이면 ECE/Brier 자체는 계산해서 보여주되, 통계적으로
# 신뢰하기엔 부족하다는 경고를 반드시 함께 출력한다.
MIN_SAMPLE_SIZE_FOR_TRUST = 30

# reliability diagram의 각 bin이 최소 이 개수의 표본을 가져야 그 bin을 그린다.
# (요청사항: "bin당 10건 이상 쌓였을 때만")
MIN_SAMPLES_PER_BIN = 10
N_BINS = 5  # bin당 최소 10건을 채우려면 표본이 적은 지금 단계에선 10bin보다 5bin이 현실적


def fetch_calibration_pairs() -> list[tuple[float, bool, dict]]:
    """(predicted_p, was_correct, context) 쌍의 리스트를 반환.

    predicted_p: 판정 시점에 정책이 갖고 있던 신뢰도(사후 확률) p.
    was_correct: 그 정책이 저장하고 있던 destination(dstored)이 사람의 실제
                 답(actual_label)과 일치했는지.
    context: 디버깅용 원본 필드(fruit_type/condition/timestamp 등).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT audit_id, fruit_type, condition, actual_label, timestamp, session_id
            FROM tb_decision_audit
            WHERE branch = 'stage3_human_resolved' AND actual_label IS NOT NULL
            ORDER BY timestamp
            """
        )
        resolved_rows = cursor.fetchall()

        pairs: list[tuple[float, bool, dict]] = []
        for resolved in resolved_rows:
            cursor.execute(
                """
                SELECT evpi_human, dstored, theta_exists, timestamp
                FROM tb_decision_audit
                WHERE branch = 'stage3_query_human'
                  AND fruit_type = ? AND condition = ?
                  AND timestamp < ?
                  AND theta_exists = 1
                  AND evpi_human IS NOT NULL
                  AND dstored IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (resolved["fruit_type"], resolved["condition"], resolved["timestamp"]),
            )
            query_row = cursor.fetchone()
            if query_row is None:
                # theta_exists=0(신규 조합이라 애초에 정책이 없던 경우)이면 "예측"
                # 자체가 없으므로 캘리브레이션 쌍에서 제외 - p=0.5는 "모른다"는
                # 뜻이지 검증 가능한 예측이 아님.
                continue

            evpi = query_row["evpi_human"]
            dstored = query_row["dstored"]
            loss = L_error(resolved["fruit_type"], resolved["condition"], dstored=dstored)
            if loss <= 0:
                continue
            predicted_p = 1.0 - (evpi / loss)
            was_correct = dstored == resolved["actual_label"]

            pairs.append((
                predicted_p, was_correct,
                {
                    "audit_id": resolved["audit_id"],
                    "fruit_type": resolved["fruit_type"],
                    "condition": resolved["condition"],
                    "dstored_predicted": dstored,
                    "actual_label": resolved["actual_label"],
                    "query_timestamp": query_row["timestamp"],
                    "resolved_timestamp": resolved["timestamp"],
                },
            ))
        return pairs
    finally:
        conn.close()


def expected_calibration_error(pairs: list[tuple[float, bool, dict]], n_bins: int = N_BINS) -> float:
    """표준 ECE(Expected Calibration Error) 계산.

    ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|
    각 bin b에 대해 "그 bin의 평균 예측확신도"와 "실제 정답률"의 차이를
    표본 비율로 가중평균한 값. 0에 가까울수록 확신도가 실제와 잘 맞는다는 뜻.
    """
    if not pairs:
        return float("nan")

    bins = [[] for _ in range(n_bins)]
    for p, correct, _ in pairs:
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, correct))

    n_total = len(pairs)
    ece = 0.0
    for bin_samples in bins:
        if not bin_samples:
            continue
        bin_conf = sum(p for p, _ in bin_samples) / len(bin_samples)
        bin_acc = sum(1 for _, c in bin_samples if c) / len(bin_samples)
        ece += (len(bin_samples) / n_total) * abs(bin_acc - bin_conf)
    return ece


def brier_score(pairs: list[tuple[float, bool, dict]]) -> float:
    """Brier score = mean((p - outcome)^2). 0에 가까울수록 좋음 (완벽=0, 최악=1)."""
    if not pairs:
        return float("nan")
    return sum((p - (1.0 if correct else 0.0)) ** 2 for p, correct, _ in pairs) / len(pairs)


def maybe_plot_reliability_diagram(pairs: list[tuple[float, bool, dict]], n_bins: int = N_BINS) -> None:
    """bin당 표본이 MIN_SAMPLES_PER_BIN 이상인 bin이 하나도 없으면 그리지 않는다."""
    bins = [[] for _ in range(n_bins)]
    for p, correct, _ in pairs:
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, correct))

    usable_bins = [b for b in bins if len(b) >= MIN_SAMPLES_PER_BIN]
    if not usable_bins:
        print(
            f"[reliability diagram] 스킵: bin당 {MIN_SAMPLES_PER_BIN}건 이상인 bin이 "
            f"하나도 없음 (현재 bin별 표본 수: {[len(b) for b in bins]}). "
            "데이터가 더 쌓이면 다시 실행하세요."
        )
        return

    try:
        import matplotlib
        matplotlib.use("Agg")  # 헤드리스 환경(로봇 서버)에서도 동작하도록 GUI 백엔드 안 씀
        import matplotlib.pyplot as plt
    except ImportError:
        print("[reliability diagram] 스킵: matplotlib이 설치되어 있지 않음 (pip install matplotlib).")
        return

    xs, ys, sizes = [], [], []
    for b in bins:
        if len(b) < MIN_SAMPLES_PER_BIN:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(1 for _, c in b if c) / len(b)
        xs.append(conf)
        ys.append(acc)
        sizes.append(len(b))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="완벽한 calibration")
    ax.scatter(xs, ys, s=[sz * 20 for sz in sizes], alpha=0.7, label="관측된 bin (점 크기=표본 수)")
    ax.set_xlabel("예측 확신도(p)")
    ax.set_ylabel("실제 정답률")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability Diagram (Track5 인프라 - 결론용 아님)")
    ax.legend()

    out_path = Path(__file__).resolve().parent / "calibration_reliability_diagram.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[reliability diagram] 저장 완료: {out_path}")


def main() -> None:
    pairs = fetch_calibration_pairs()
    n = len(pairs)

    print(f"=== Track5 Calibration 인프라 리포트 ===")
    print(f"짝지어진 (예측확신도, 실제일치여부) 표본 수: {n}")

    if n == 0:
        print(
            "표본이 0건입니다 - theta_exists=1인 정책이 사람 피드백으로 재검증된 사례가 "
            "아직 없습니다 (신규 조합 첫 질문들만 있으면 여기 안 잡힙니다 - 의도된 동작)."
        )
        return

    if n < MIN_SAMPLE_SIZE_FOR_TRUST:
        print(
            f"[경고] 표본 수({n}건)가 {MIN_SAMPLE_SIZE_FOR_TRUST}건 미만입니다. "
            "아래 ECE/Brier 값은 계산은 되지만 통계적으로 무의미할 수 있습니다 - "
            "이 수치로 '모델이 잘 보정되어 있다/아니다'를 판단하지 마세요. "
            "참고용 파이프라인 동작 확인 목적으로만 사용하세요."
        )

    ece = expected_calibration_error(pairs)
    brier = brier_score(pairs)
    print(f"ECE (Expected Calibration Error): {ece:.4f}  (0에 가까울수록 잘 보정됨)")
    print(f"Brier score: {brier:.4f}  (0에 가까울수록 좋음, 0.25=무작위 추측 수준)")

    print("\n--- 표본별 상세 (디버깅용) ---")
    for p, correct, ctx in pairs:
        mark = "O" if correct else "X"
        print(
            f"  [{mark}] p={p:.3f} fruit={ctx['fruit_type']} condition={ctx['condition']} "
            f"predicted={ctx['dstored_predicted']} actual={ctx['actual_label']} "
            f"(audit_id={ctx['audit_id']})"
        )

    maybe_plot_reliability_diagram(pairs)


if __name__ == "__main__":
    main()
