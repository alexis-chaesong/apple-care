# 통합 드라이런: synthetic_pipeline_run.py를 pytest 회귀 스위트에 포함
# Track 6: run_cases()가 async로 바뀌면서 asyncio.run()으로 호출하도록 갱신

"""
test_synthetic_pipeline_run.py
=================================
synthetic_pipeline_run.py의 5+1개 필수 케이스 + Stage2 순수성 확인 +
replay_session.py CLI 연결이 회귀 테스트로도 계속 통과하는지 확인한다.
실제 검증 로직(각 케이스의 기대값)은 synthetic_pipeline_run.py의 assert문 자체에
있고, 이 파일은 그 스크립트가 예외 없이 끝까지 실행되는지만 감싼다.

사용법:
    cd code/backend && pytest test_synthetic_pipeline_run.py -v
"""

import asyncio

import pytest

import database
import synthetic_pipeline_run as spr


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test_synthetic_pipeline.db"))
    database.init_db()
    yield


def test_six_required_cases_run_without_error(isolated_db):
    asyncio.run(spr.run_cases())


def test_stage2_vlm_log_never_mutates_posterior(isolated_db):
    spr.verify_stage2_never_updates_posterior()


def test_synthetic_log_feeds_into_replay_session_cli(isolated_db):
    asyncio.run(spr.run_cases())
    report = spr.run_replay_on_synthetic_log()
    assert "세트 수(N)" in report
