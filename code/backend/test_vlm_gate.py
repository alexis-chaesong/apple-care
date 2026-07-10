# Track 6: services/vlm_gate.py 단위 테스트 (OpenAI 클라이언트는 테스트에서만 patch)

"""
test_vlm_gate.py
===================
services/vlm_gate.py에 대한 pytest 단위 테스트.

중요: 이 테스트만 OpenAI 클라이언트를 patch한다 - 프로덕션 코드
(services/vlm_gate.py의 call_gpt4o_vlm)는 실제로 OpenAI API를 호출한다.
일반적인 소프트웨어 테스트 패턴(실제 코드는 진짜 호출, 테스트에서만 결정적
응답으로 대체)을 따른다.

사용법:
    cd code/backend && pytest test_vlm_gate.py -v
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIError, APITimeoutError

import services.vlm_gate as vlm_gate


def _fake_openai_response(content: str):
    """openai SDK의 ChatCompletion 응답 객체를 흉내낸 최소 스텁."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def test_get_captured_frame_for_vlm_returns_none_when_ros_bridge_not_started():
    """Task1부터 get_captured_frame_for_vlm()은 더 이상 항상 None을 반환하는
    스텁이 아니라 실제 capture_frame ROS2 서비스를 호출한다. 이 pytest
    프로세스는 vision_bridge_manager.start_bridge()를 호출하지 않으므로
    (ROS2 노드/실제 카메라가 없는 일반 CI 환경) capture_frame_client가 아직
    None이고, request_captured_frame()이 그 상태를 감지해 예외 없이 None으로
    폴백하는지 확인한다 - 실제 서비스 호출 자체는 여기서 검증하지 않는다
    (그건 scripts/ 쪽 물리 드라이런 하네스의 몫)."""
    assert asyncio.run(vlm_gate.get_captured_frame_for_vlm("any-request-id")) is None
    assert asyncio.run(vlm_gate.get_captured_frame_for_vlm("")) is None


def test_call_gpt4o_vlm_success_returns_identified_object(monkeypatch):
    response = _fake_openai_response(json.dumps({"identified_object": "망치"}, ensure_ascii=False))
    mock_create = AsyncMock(return_value=response)
    monkeypatch.setattr(vlm_gate._client.chat.completions, "create", mock_create)

    result = asyncio.run(vlm_gate.call_gpt4o_vlm(b"fake-jpeg-bytes"))

    assert result["identified_object"] == "망치"
    assert "망치" in result["raw_response"]
    mock_create.assert_awaited_once()
    # destination을 절대 묻지 않는다는 프롬프트 설계 검증 - 호출 인자에
    # "destination" 관련 문구가 섞여 들어가지 않아야 함
    call_kwargs = mock_create.await_args.kwargs
    system_message = call_kwargs["messages"][0]["content"]
    assert "destination" not in system_message.lower()
    assert "목적지" not in system_message or "판단하지 마라" in system_message


def test_call_gpt4o_vlm_timeout_swallows_exception_and_returns_none(monkeypatch):
    mock_create = AsyncMock(side_effect=APITimeoutError(_fake_request()))
    monkeypatch.setattr(vlm_gate._client.chat.completions, "create", mock_create)

    result = asyncio.run(vlm_gate.call_gpt4o_vlm(b"fake-jpeg-bytes"))

    assert result["identified_object"] is None
    assert result["raw_response"] is None


def test_call_gpt4o_vlm_api_error_swallows_exception_and_returns_none(monkeypatch):
    mock_create = AsyncMock(side_effect=APIError("boom", _fake_request(), body=None))
    monkeypatch.setattr(vlm_gate._client.chat.completions, "create", mock_create)

    result = asyncio.run(vlm_gate.call_gpt4o_vlm(b"fake-jpeg-bytes"))

    assert result["identified_object"] is None
    assert result["raw_response"] is None


def test_call_gpt4o_vlm_malformed_json_returns_none_but_keeps_raw_response(monkeypatch):
    response = _fake_openai_response("이건 JSON이 아님")
    mock_create = AsyncMock(return_value=response)
    monkeypatch.setattr(vlm_gate._client.chat.completions, "create", mock_create)

    result = asyncio.run(vlm_gate.call_gpt4o_vlm(b"fake-jpeg-bytes"))

    assert result["identified_object"] is None
    assert result["raw_response"] == "이건 JSON이 아님"


def test_call_gpt4o_vlm_explicit_null_identified_object_returns_none(monkeypatch):
    """모델이 스스로 "특정 불가"라고 답한 경우(identified_object: null)도
    실패와 동일하게 취급되어야 한다."""
    response = _fake_openai_response(json.dumps({"identified_object": None}))
    mock_create = AsyncMock(return_value=response)
    monkeypatch.setattr(vlm_gate._client.chat.completions, "create", mock_create)

    result = asyncio.run(vlm_gate.call_gpt4o_vlm(b"fake-jpeg-bytes"))

    assert result["identified_object"] is None


def test_call_gpt4o_vlm_passes_hard_timeout_from_settings(monkeypatch):
    """settings.vlm_call_timeout_sec(잠정치, config placeholder)이 실제로
    OpenAI 호출에 그대로 전달되는지 확인. config.settings는 frozen dataclass라
    monkeypatch.setattr이 아니라 object.__setattr__로 임시 변경한다
    (test_hierarchical_prior.py와 동일 패턴)."""
    response = _fake_openai_response(json.dumps({"identified_object": "망치"}, ensure_ascii=False))
    mock_create = AsyncMock(return_value=response)
    monkeypatch.setattr(vlm_gate._client.chat.completions, "create", mock_create)

    original_timeout = vlm_gate.settings.vlm_call_timeout_sec
    object.__setattr__(vlm_gate.settings, "vlm_call_timeout_sec", 3.5)
    try:
        asyncio.run(vlm_gate.call_gpt4o_vlm(b"fake-jpeg-bytes"))
    finally:
        object.__setattr__(vlm_gate.settings, "vlm_call_timeout_sec", original_timeout)

    assert mock_create.await_args.kwargs["timeout"] == 3.5
