# Track 1: 손실 행렬 및 오분류 손실 근사

"""
loss_matrix.py
===============
리서치 문서("Congestion-Aware Bayesian Learning-to-Defer for Real-Time
Human-in-the-Loop Robotic Sorting") §4.1(Loss Matrix), §4.3.1(Condition
분류), §4.3.2(오분류 손실 근사)를 코드로 옮긴 파일.

이 파일은 Stage 3(Human Query Gate)의 EVPI_human(θ_f,c) 계산이 참조하는
"틀렸을 때 손실이 얼마인가"만 책임진다. Beta(α,β) posterior 자체
(bayesian_policy.py)나 게이팅 로직(decision_planner.py)은 다루지 않는다.

주의 (문서 원문과의 차이): 사용자 요청 시점에는 L_value를 "실제\\판정 4x4
(normal/ugly/processing/discard_box)"로 가정했으나, 문서 §4.1 원문의 L_value
행렬은 실제(행) 3종(normal/ugly/processing) × 판정(열) 4종
(normal_box/ugly_box/processing_box/discard_box)인 3x4 행렬이다. "실제
discard(=rotten/moldy)" 행이 없는 이유는, 그 경우의 손실은 L_value가 아니라
§4.1의 안전 위반 항(SAFETY_VIOLATION_LOSS)이 전담하기 때문이다
(L(y,d) = 1000·1[y∈{rotten,moldy} ∧ d≠discard_box] + L_value(y,d)).
아래 구현은 문서 원문 그대로를 코드화했다.
"""

from typing import Optional

import numpy as np

ACTUAL_ROWS = ("normal", "ugly", "processing")
DESTINATIONS = ("normal_box", "ugly_box", "processing_box", "discard_box")

# §4.1 원문 그대로의 L_value 3x4 행렬 (실제=행, 판정=열)
L_VALUE_MATRIX = np.array(
    [
        [0, 3, 4, 10],
        [2, 0, 1, 5],
        [5, 1, 0, 3],
    ],
    dtype=float,
)

SAFETY_VIOLATION_LOSS = 1000.0

_ROW_INDEX = {name: i for i, name in enumerate(ACTUAL_ROWS)}
_COLUMN_INDEX = {name: i for i, name in enumerate(DESTINATIONS)}
# dstored(예: "normal_box")를 L_error()의 "실제(actual) 행"으로 간주할 때 쓰는
# 역방향 매핑. discard_box는 대응하는 actual row가 없으므로(그 손실은 안전 위반
# 항이 전담) 의도적으로 이 딕셔너리에 없음 - L_error()가 .get()으로 None을 받아
# 최댓값 fallback으로 처리한다.
_DESTINATION_TO_ROW = {f"{name}_box": name for name in ACTUAL_ROWS}


def verify_dominance_condition() -> None:
    """§4.1 "지배 조건 검증: L_value 최댓값(10) ≪ 안전 위반 최솟값(1000)"을
    상수 하드코딩이 아니라 실행 시점에 검증한다.

    L_value 행렬이나 SAFETY_VIOLATION_LOSS가 향후 수정되어 지배 조건이
    깨지면(예: 안전 손실보다 가치 손실이 커지면), 세이프티 우선순위가
    보장되지 않으므로 즉시 실패해야 한다.
    """
    assert L_VALUE_MATRIX.max() < SAFETY_VIOLATION_LOSS, (
        f"지배조건 위반: L_value 최댓값({L_VALUE_MATRIX.max()}) >= "
        f"SAFETY_VIOLATION_LOSS({SAFETY_VIOLATION_LOSS})"
    )


# 모듈 임포트 시점에 즉시 실행 (fail fast) - §4.1 지배조건.
verify_dominance_condition()


def get_l_value(actual: str, destination: str) -> float:
    """L_value(y=actual, d=destination) 단일 값 조회."""
    return float(L_VALUE_MATRIX[_ROW_INDEX[actual], _COLUMN_INDEX[destination]])


def row_max(actual: str) -> float:
    """L_value 행렬에서 실제(actual) 행 하나의 최댓값(최악의 오분류 손실)."""
    return float(L_VALUE_MATRIX[_ROW_INDEX[actual]].max())


SAFETY_ACTUAL_LABELS = ("rotten", "moldy")


def L(y: str, d: str) -> float:
    """§4.1 전체 Loss 함수.

    "L(y, d) = 1000 · 1[y ∈ {rotten, moldy} ∧ d ≠ discard_box] + L_value(y, d)"

    y(실제/사후 라벨)가 안전 라벨(rotten/moldy)이 아니면 안전 위반 항은 0이고,
    ACTUAL_ROWS(normal/ugly/processing)에 속할 때만 L_value(y,d)가 더해진다
    (안전 라벨은 L_VALUE_MATRIX에 행이 없으므로 그 항은 0으로 취급).

    §5.2 리플레이(replay_session.py)의 "총 손실 sum(L(y,d))" 지표가 이 함수를 사용한다.
    """
    safety_violation = SAFETY_VIOLATION_LOSS if y in SAFETY_ACTUAL_LABELS and d != "discard_box" else 0
    value_term = get_l_value(y, d) if y in _ROW_INDEX else 0
    return safety_violation + value_term


# §4.3.1 — mold/rotten/unknown은 "정체 모름=최악" 가정으로 안전 계열 취급.
# 주의: 실제 코드 경로상 vision_bridge.py가 YOLO의 apple_rotten 클래스를
# defect_type="mold"로 정규화해버려서, condition 문자열로서의 "rotten"은 현재
# decision_planner.py에서 도달 불가능하다 (0단계 재확인 결과, 원래도 동일했음).
# 그럼에도 연구문서 §4.3.1과의 일치 및 향후 vision 매핑이 바뀔 가능성을 대비해
# 그대로 유지한다.
SAFETY_CONDITIONS = {"mold", "rotten", "unknown"}
VALUE_CONDITIONS = {"bruise", "scratch", "small"}


def L_error(
    f: str,
    c: str,
    theta_f_c: Optional[float] = None,
    dstored: Optional[str] = None,
) -> float:
    """§4.3.2 오분류 손실 근사.

    "정책 θ_f,c가 틀렸을 때의 기대 손실을 다음과 같이 근사한다:

        L_error(f, c) = 1000                                  if c ∈ {mold, rotten, unknown}
                      = max_d' L_value(row=d_stored, d')       if c ∈ {bruise, scratch, small}

     가치 계열의 경우, 현재 저장된 정책의 destination d_stored를 손실 행렬의
     '실제(actual)' 행으로 간주하고, 그 행의 최댓값(최악의 오분류 손실)을
     취하는 보수적 근사를 사용한다. 예: small 조건의 정책이 normal_box로
     저장되어 있으면, normal 행의 최댓값(10, normal→discard_box)을 사용한다.

     정책이 아직 존재하지 않는 완전 신규 조합은 p=0.5(최대 불확실성)로
     간주해 최대 EVPI를 반환한다." (§4.3.3)
     -> dstored가 없으면(정책 미존재) 해당 조건 계열(가치 계열)의 최댓값을
        동일한 취지로 보수적으로 반환한다.

    Args:
        f: fruit_type. θ_f,c 표기와의 대응을 위해 시그니처에 유지하나,
            dstored가 이미 특정 (f, c) 조합에 대해 조회된 값이므로 이 함수의
            계산 자체는 f를 사용하지 않는다.
        c: condition (SAFETY_CONDITIONS 또는 VALUE_CONDITIONS 중 하나여야 함).
        theta_f_c: 현재 정책 신뢰도. §4.3.3 EVPI_human(θ_f,c) = (1-p)·L_error(f,c)
            에서 p는 θ_f,c로부터 계산되지만, L_error 자체의 수식(§4.3.2)은
            θ_f,c에 의존하지 않는다. EVPI_human과의 시그니처 대칭을 위해
            받아두되 이 함수 내부에서는 사용하지 않는다.
        dstored: 현재 저장된 정책의 destination (예: "normal_box").
            None이면(정책 미존재) 가치 계열의 최댓값을 보수적으로 반환한다.

    Returns:
        오분류 손실 근사값 (float).

    Raises:
        ValueError: c가 SAFETY_CONDITIONS/VALUE_CONDITIONS 어디에도 속하지 않을 때.
    """
    if c in SAFETY_CONDITIONS:
        return SAFETY_VIOLATION_LOSS
    if c in VALUE_CONDITIONS:
        if dstored is None:
            return float(L_VALUE_MATRIX.max())
        actual_row = _DESTINATION_TO_ROW.get(dstored)
        if actual_row is None:
            return float(L_VALUE_MATRIX.max())
        return row_max(actual_row)
    raise ValueError(
        f"알 수 없는 condition: {c!r} (SAFETY_CONDITIONS={SAFETY_CONDITIONS} "
        f"또는 VALUE_CONDITIONS={VALUE_CONDITIONS} 중 하나여야 함)"
    )
