# Track 1: services/loss_matrix.py 단위 테스트

"""
test_loss_matrix.py
=====================
services/loss_matrix.py에 대한 pytest 단위 테스트.
리서치 문서 §4.1(지배 조건)과 §4.3.2(small→normal_box 예시)를 그대로 검증한다.

사용법:
    cd code/backend && pytest test_loss_matrix.py -v
"""

import pytest

from services.loss_matrix import (
    ACTUAL_ROWS,
    DESTINATIONS,
    L_VALUE_MATRIX,
    SAFETY_CONDITIONS,
    SAFETY_VIOLATION_LOSS,
    VALUE_CONDITIONS,
    L,
    L_error,
    get_l_value,
    row_max,
    verify_dominance_condition,
)


def test_dominance_condition_holds():
    """§4.1: "지배 조건 검증: L_value 최댓값(10) ≪ 안전 위반 최솟값(1000)"."""
    assert L_VALUE_MATRIX.max() < SAFETY_VIOLATION_LOSS
    verify_dominance_condition()


def test_dominance_condition_fails_if_violated(monkeypatch):
    """지배 조건이 깨지면 verify_dominance_condition()이 즉시 실패해야 한다."""
    monkeypatch.setattr("services.loss_matrix.SAFETY_VIOLATION_LOSS", 5)
    with pytest.raises(AssertionError):
        verify_dominance_condition()


def test_l_value_matrix_matches_document():
    """§4.1 L_value 행렬 (0~10 스케일) 원문 수치 그대로."""
    assert get_l_value("normal", "normal_box") == 0
    assert get_l_value("normal", "ugly_box") == 3
    assert get_l_value("normal", "processing_box") == 4
    assert get_l_value("normal", "discard_box") == 10

    assert get_l_value("ugly", "normal_box") == 2
    assert get_l_value("ugly", "ugly_box") == 0
    assert get_l_value("ugly", "processing_box") == 1
    assert get_l_value("ugly", "discard_box") == 5

    assert get_l_value("processing", "normal_box") == 5
    assert get_l_value("processing", "ugly_box") == 1
    assert get_l_value("processing", "processing_box") == 0
    assert get_l_value("processing", "discard_box") == 3


def test_condition_classification():
    """§4.3.1: 안전 계열 / 가치 계열 분류."""
    assert SAFETY_CONDITIONS == {"mold", "rotten", "unknown"}
    assert VALUE_CONDITIONS == {"small", "bruise", "scratch"}
    assert SAFETY_CONDITIONS.isdisjoint(VALUE_CONDITIONS)


@pytest.mark.parametrize("condition", ["mold", "rotten", "unknown"])
def test_l_error_safety_conditions_return_fixed_penalty(condition):
    """§4.3.2: c ∈ {mold, rotten, unknown}이면 L_error = 1000 (dstored와 무관)."""
    assert L_error("apple", condition) == SAFETY_VIOLATION_LOSS
    assert L_error("apple", condition, dstored="normal_box") == SAFETY_VIOLATION_LOSS


def test_l_error_small_normal_box_document_example():
    """§4.3.2 문서 예시 그대로: "small 조건의 정책이 normal_box로 저장되어
    있으면, normal 행의 최댓값(10, normal→discard_box)을 사용한다.".
    """
    assert row_max("normal") == 10
    assert L_error("apple", "small", dstored="normal_box") == 10


@pytest.mark.parametrize(
    "condition,dstored,expected",
    [
        ("bruise", "normal_box", 10),
        ("scratch", "ugly_box", 5),
        ("small", "processing_box", 5),
    ],
)
def test_l_error_value_conditions_use_stored_destination_as_actual_row(condition, dstored, expected):
    """§4.3.2: 가치 계열은 dstored를 '실제' 행으로 간주해 그 행의 최댓값을 반환."""
    assert L_error("apple", condition, dstored=dstored) == expected


def test_l_error_value_condition_without_policy_returns_series_max():
    """§4.3.2/§4.3.3: 정책 미존재(dstored=None) -> 가치 계열 최댓값을 보수적으로 반환."""
    assert L_error("apple", "small", dstored=None) == L_VALUE_MATRIX.max() == 10


def test_l_error_value_condition_with_unmapped_destination_falls_back_to_series_max():
    """dstored가 L_value 행렬에 대응 행이 없는 목적지(discard_box)이면
    정책 미존재와 동일하게 계열 최댓값으로 보수적 폴백한다.
    """
    assert L_error("apple", "small", dstored="discard_box") == L_VALUE_MATRIX.max() == 10


def test_l_error_unknown_condition_raises():
    with pytest.raises(ValueError):
        L_error("apple", "not_a_real_condition")


def test_actual_rows_and_destinations_shape():
    assert ACTUAL_ROWS == ("normal", "ugly", "processing")
    assert DESTINATIONS == ("normal_box", "ugly_box", "processing_box", "discard_box")
    assert L_VALUE_MATRIX.shape == (len(ACTUAL_ROWS), len(DESTINATIONS))


def test_l_full_loss_function_safety_label():
    """§4.1: y가 안전 라벨이고 d != discard_box이면 안전 위반 항(1000)이 더해진다."""
    assert L("rotten", "normal_box") == SAFETY_VIOLATION_LOSS
    assert L("moldy", "processing_box") == SAFETY_VIOLATION_LOSS
    assert L("rotten", "discard_box") == 0


def test_l_full_loss_function_value_label():
    """§4.1: y가 ACTUAL_ROWS에 속하면 안전 위반 항 없이 L_value(y,d)만 반환."""
    assert L("normal", "discard_box") == 10
    assert L("ugly", "processing_box") == 1
