# services/vlm_gate.py의 call_gpt4o_vlm()이 실제 OpenAI API를 상대로 정상 동작하는지 확인
#
# 중요: 이 스크립트는 pytest 스위트에 포함되지 않는다 (test_vlm_gate.py는 클라이언트를
# patch해서 결정적으로 검증하고, 이 스크립트는 그 반대 - 매번 진짜 네트워크 호출을 한다).
# 실행할 때마다 실제 OpenAI API 비용이 발생하고 네트워크/모델 상태에 따라 결과가 달라질
# 수 있으므로 CI에 넣지 않고 사람이 필요할 때 직접 실행해서 눈으로 확인하는 용도다.
#
# 사용법:
#   cd code/backend && python3 scripts/smoke_test_vlm_gate.py

import asyncio
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# scripts/ 하위에서 실행돼도 backend/ 루트(services, config가 있는 곳)를 import할 수 있도록
# sys.path에 추가 - "cd code/backend && python3 scripts/..."로 실행하면 원래도 되지만,
# 다른 위치에서 스크립트 경로로 직접 실행해도 깨지지 않게 방어적으로 넣어둠.

from PIL import Image, ImageDraw

from config import settings
from services.vlm_gate import call_gpt4o_vlm


def _make_sample_image() -> bytes:
    """저장소 안에 이 검증용으로 쓸 만한 샘플 이미지가 없어 PIL로 직접 생성한다.

    실제 사과/과일 사진일 필요는 없다 - 이 스모크 테스트가 확인하려는 건 (1) API
    호출 자체가 실제로 나가고 응답이 오는지, (2) 프롬프트가 물체명만 묻고
    destination을 안 묻는지, (3) 파싱 로직이 실제 응답 형태에 견고한지이지,
    YOLO/비전 모델의 인식 정확도가 아니다.
    """
    img = Image.new("RGB", (400, 400), color=(235, 235, 235))
    draw = ImageDraw.Draw(img)
    draw.ellipse((90, 100, 310, 320), fill=(190, 30, 30))  # 빨간 원 (사과 비슷한 실루엣)
    draw.rectangle((185, 55, 215, 105), fill=(80, 55, 20))  # 꼭지
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _print_header(title: str) -> None:
    print("=" * 90)
    print(title)


async def case1_normal_call(image_bytes: bytes) -> bool:
    _print_header("케이스 1 — 정상 호출 확인")
    print(f"  openai_model            = {settings.openai_model}")
    print(f"  openai_api_key 설정됨    = {bool(settings.openai_api_key)}")
    print(f"  vlm_call_timeout_sec     = {settings.vlm_call_timeout_sec}")

    try:
        result = await call_gpt4o_vlm(image_bytes)
    except Exception as exc:  # noqa: BLE001
        print(f"  [실패] 예외가 밖으로 샘(삼켜지지 않음): {exc!r}")
        return False

    print(f"  identified_object = {result['identified_object']!r}")
    print("  raw_response 원문 (destination 관련 언급 여부는 직접 확인):")
    print("  " + "-" * 86)
    print(f"  {result['raw_response']}")
    print("  " + "-" * 86)

    ok = result["identified_object"] is not None and result["raw_response"] is not None
    print(f"  [{'성공' if ok else '실패'}] identified_object/raw_response가 둘 다 채워짐: {ok}")
    return ok


async def case2_timeout(image_bytes: bytes) -> bool:
    _print_header("케이스 2 — 타임아웃 동작 확인 (vlm_call_timeout_sec를 0.01초로 일시 오버라이드)")

    # config.settings는 frozen dataclass라 monkeypatch 없이 직접 값을 바꾸려면
    # object.__setattr__로 우회해야 한다 (test_hierarchical_prior.py와 동일 패턴).
    original_timeout = settings.vlm_call_timeout_sec
    object.__setattr__(settings, "vlm_call_timeout_sec", 0.01)
    try:
        result = await call_gpt4o_vlm(image_bytes)
    except Exception as exc:  # noqa: BLE001
        print(f"  [실패] 예외가 밖으로 샘(타임아웃이 안전하게 삼켜지지 않음): {exc!r}")
        return False
    finally:
        object.__setattr__(settings, "vlm_call_timeout_sec", original_timeout)
        print(f"  타임아웃 원래 값으로 복원 완료: {settings.vlm_call_timeout_sec}")

    print(f"  identified_object = {result['identified_object']!r}  (기대값: None)")
    print(f"  raw_response      = {result['raw_response']!r}  (기대값: None)")

    ok = result["identified_object"] is None
    print(f"  [{'성공' if ok else '실패'}] 타임아웃이 예외 없이 identified_object=None으로 안전 폴백됨: {ok}")
    return ok


async def case3_parsing_robustness(image_bytes: bytes, repeats: int = 3) -> bool:
    _print_header(f"케이스 3 — 응답 파싱 견고성 확인 (같은 이미지로 {repeats}회 반복 호출)")

    all_parsed = True
    for i in range(1, repeats + 1):
        try:
            result = await call_gpt4o_vlm(image_bytes)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{repeats}] [실패] 예외가 밖으로 샘: {exc!r}")
            all_parsed = False
            continue

        print(f"  [{i}/{repeats}] identified_object={result['identified_object']!r}")
        print(f"           raw_response={result['raw_response']!r}")
        if result["identified_object"] is None:
            print(f"  [{i}/{repeats}] 주의: identified_object가 None - 파싱 실패 또는 모델이 스스로 null 응답")
            all_parsed = False

    print(f"  [{'성공' if all_parsed else '주의 필요'}] {repeats}회 모두 identified_object가 채워졌는가: {all_parsed}")
    return all_parsed


async def main() -> None:
    if not settings.openai_api_key:
        print("OPENAI_API_KEY가 비어 있습니다 (code/.env 확인). 스모크 테스트를 중단합니다.")
        return

    image_bytes = _make_sample_image()
    print(f"샘플 이미지 생성 완료: {len(image_bytes)} bytes (PIL로 그린 빨간 원 - 실제 사과 사진 아님)")

    results = {
        "케이스1 (정상 호출)": await case1_normal_call(image_bytes),
        "케이스2 (타임아웃 폴백)": await case2_timeout(image_bytes),
        "케이스3 (파싱 견고성)": await case3_parsing_robustness(image_bytes),
    }

    _print_header("최종 요약")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL/주의 필요 - 위 로그 확인'}")


if __name__ == "__main__":
    asyncio.run(main())
