# Track 3: 계층적 Prior 단위 테스트

"""
test_hierarchical_prior.py
=============================
§5.4 계층적 Prior Ablation 구현(services/bayesian_policy.py의 get_hierarchical_prior,
get_or_init_theta, config.settings.use_hierarchical_prior 스위치)에 대한 pytest.

검증 대상:
  1) 다른 fruit_type이 이미 같은 condition에서 높은 confidence를 쌓아뒀으면, 신규
     fruit_type의 그 condition 조합이 Beta(1,1)보다 더 informative한 prior로 시작하는지
  2) 폴백 조건(유효 표본 0개 -> Beta(1,1))
  3) USE_HIERARCHICAL_PRIOR=False일 때 기존 Beta(1,1) 동작이 그대로 보존되는지(회귀 방지)

실제 data/robot_system.db 대신 tmp_path 임시 SQLite 파일 사용.

사용법:
    cd code/backend && pytest test_hierarchical_prior.py -v
"""

import pytest

import config
import database
from services.bayesian_policy import get_hierarchical_prior, get_or_init_theta, record_human_feedback
from services.decision_audit import get_audit_log_by_id


def _set_use_hierarchical_prior(enabled: bool) -> None:
    """config.settings는 @dataclass(frozen=True)라 monkeypatch.setattr(인스턴스, ...)이
    바로는 통하지 않는다 (FrozenInstanceError). object.__setattr__로 우회한다.
    settings는 프로세스 전체에서 공유되는 단일 인스턴스라, 이렇게 한 번 바꾸면
    bayesian_policy.py 등 어디서 `from config import settings`로 가져다 썼든
    전부 같은 객체를 보므로 즉시 반영된다."""
    object.__setattr__(config.settings, "use_hierarchical_prior", enabled)


@pytest.fixture(autouse=True)
def _restore_hierarchical_prior_flag():
    """매 테스트 종료 후 플래그를 원래 값으로 복원 (다른 테스트 파일에 영향 안 가게)."""
    original = config.settings.use_hierarchical_prior
    yield
    object.__setattr__(config.settings, "use_hierarchical_prior", original)


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test_hierarchical_prior.db"))
    database.init_db()
    yield


def _seed_human_feedback_policy(fruit_type: str, condition: str, alpha: float, beta: float, destination: str) -> None:
    """다른 fruit_type이 이미 이 condition으로 사람 피드백을 여러 번 받아 학습된
    상태를 만들기 위한 테스트 헬퍼. tb_policy_memory에 source='human_feedback' 행을
    직접 심는다 (record_human_feedback()을 여러 번 호출하는 대신 alpha/beta를 바로
    지정해 테스트를 결정적으로 만듦)."""
    conn = database.get_db_connection()
    try:
        cursor = conn.cursor()
        confidence = alpha / (alpha + beta)
        cursor.execute(
            """
            INSERT INTO tb_policy_memory
                (fruit_type, condition, destination, alpha, beta, confidence, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'human_feedback', datetime('now', 'localtime'))
            ON CONFLICT(fruit_type, condition) DO UPDATE SET
                destination = excluded.destination,
                alpha = excluded.alpha,
                beta = excluded.beta,
                confidence = excluded.confidence,
                source = 'human_feedback',
                updated_at = datetime('now', 'localtime')
            """,
            (fruit_type, condition, destination, alpha, beta, confidence),
        )
        conn.commit()
    finally:
        conn.close()


def test_hierarchical_prior_more_informative_than_uniform_when_pooled_data_exists(isolated_db):
    _set_use_hierarchical_prior(True)
    _seed_human_feedback_policy("kiwi", "bruise", alpha=90, beta=10, destination="ugly_box")

    prior_alpha, prior_beta, pooled_mean, pooled_n = get_hierarchical_prior("mango", "bruise")

    assert pooled_n == 1
    assert pooled_mean == pytest.approx(0.9)

    k = config.settings.hierarchical_prior_pseudo_count
    assert prior_alpha == pytest.approx(k * 0.9)
    assert prior_beta == pytest.approx(k * 0.1)

    hierarchical_mean = prior_alpha / (prior_alpha + prior_beta)
    uniform_mean = config.settings.bayesian_prior_alpha / (
        config.settings.bayesian_prior_alpha + config.settings.bayesian_prior_beta
    )
    assert hierarchical_mean == pytest.approx(0.9)
    assert hierarchical_mean > uniform_mean


def test_get_or_init_theta_uses_hierarchical_prior_for_brand_new_combo(isolated_db):
    _set_use_hierarchical_prior(True)
    _seed_human_feedback_policy("kiwi", "bruise", alpha=90, beta=10, destination="ugly_box")

    theta_init = get_or_init_theta("mango", "bruise")

    assert theta_init["method"] == "hierarchical"
    assert theta_init["pooled_sample_size"] == 1
    assert theta_init["pooled_mean"] == pytest.approx(0.9)

    k = config.settings.hierarchical_prior_pseudo_count
    assert theta_init["alpha"] == pytest.approx(k * 0.9)
    assert theta_init["beta"] == pytest.approx(k * 0.1)


def test_record_human_feedback_logs_prior_initialized_with_hierarchical_values(isolated_db):
    """§5.4: "신규 과일 첫 판단 시점의 confidence 값"을 tb_decision_audit에서
    재구성할 수 있어야 한다 - log_prior_initialized가 실제 hierarchical prior
    값을 담아 기록하는지 확인."""
    _set_use_hierarchical_prior(True)
    _seed_human_feedback_policy("kiwi", "bruise", alpha=90, beta=10, destination="ugly_box")

    updated = record_human_feedback(
        fruit_type="mango", condition="bruise", destination="ugly_box", raw_answer="못난이로 보내"
    )

    k = config.settings.hierarchical_prior_pseudo_count
    assert updated["alpha"] == pytest.approx(k * 0.9 + 1)
    assert updated["beta"] == pytest.approx(k * 0.1)

    conn = database.get_db_connection()
    try:
        row = conn.execute(
            "SELECT audit_id FROM tb_decision_audit WHERE branch='prior_initialized' "
            "AND fruit_type='mango' AND condition='bruise'"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    audit = get_audit_log_by_id(row["audit_id"])
    assert audit["prior_init_method"] == "hierarchical"
    assert audit["prior_pooled_mean"] == pytest.approx(0.9)
    assert audit["prior_pooled_sample_size"] == 1
    assert audit["alpha_after"] == pytest.approx(k * 0.9)
    assert audit["beta_after"] == pytest.approx(k * 0.1)


def test_hierarchical_prior_falls_back_to_uniform_when_no_pooled_samples(isolated_db):
    _set_use_hierarchical_prior(True)

    prior_alpha, prior_beta, pooled_mean, pooled_n = get_hierarchical_prior("kiwi", "scratch")
    assert pooled_n == 0
    assert pooled_mean is None
    assert prior_alpha == pytest.approx(config.settings.bayesian_prior_alpha)
    assert prior_beta == pytest.approx(config.settings.bayesian_prior_beta)

    theta_init = get_or_init_theta("kiwi", "scratch")
    assert theta_init["method"] == "hierarchical_fallback_uniform"
    assert theta_init["alpha"] == pytest.approx(config.settings.bayesian_prior_alpha)
    assert theta_init["beta"] == pytest.approx(config.settings.bayesian_prior_beta)


def test_llm_policy_rows_excluded_from_pooling(isolated_db):
    """source='llm_policy' 행(confidence가 실제 학습이 아니라 강제 고정값)은
    풀링 대상에서 제외되어야 한다. 시드 데이터의 ("apple","mold","discard_box",
    "llm_policy",1.0)만 있는 상태에서 "mold" 조건을 풀링하면 유효 표본 0개여야 함."""
    _set_use_hierarchical_prior(True)
    _, _, pooled_mean, pooled_n = get_hierarchical_prior("kiwi", "mold")
    assert pooled_n == 0
    assert pooled_mean is None


def test_own_fruit_type_excluded_from_pooling(isolated_db):
    """"같은 condition의 *다른* fruit_type들"이라는 조건대로, 자기 자신의 fruit_type은
    풀링 대상에서 제외되어야 한다."""
    _set_use_hierarchical_prior(True)
    _seed_human_feedback_policy("apple", "bruise", alpha=90, beta=10, destination="ugly_box")

    _, _, pooled_mean, pooled_n = get_hierarchical_prior("apple", "bruise")
    assert pooled_n == 0
    assert pooled_mean is None


def test_use_hierarchical_prior_false_preserves_uniform_beta_behavior(isolated_db):
    _set_use_hierarchical_prior(False)
    _seed_human_feedback_policy("kiwi", "bruise", alpha=90, beta=10, destination="ugly_box")

    theta_init = get_or_init_theta("mango", "bruise")
    assert theta_init["method"] == "uniform"
    assert theta_init["alpha"] == pytest.approx(config.settings.bayesian_prior_alpha)
    assert theta_init["beta"] == pytest.approx(config.settings.bayesian_prior_beta)

    updated = record_human_feedback(
        fruit_type="mango", condition="bruise", destination="ugly_box", raw_answer="못난이로 보내"
    )
    assert updated["alpha"] == pytest.approx(config.settings.bayesian_prior_alpha + 1)
    assert updated["beta"] == pytest.approx(config.settings.bayesian_prior_beta)


def test_default_config_flag_is_false(isolated_db):
    """스위치 기본값 자체가 False라 기존 동작을 보존해야 한다 (완전 대체 금지 요건)."""
    assert config.settings.use_hierarchical_prior is False
