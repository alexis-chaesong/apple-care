#!/usr/bin/env bash
# run_backend.sh
# ================
# 2026-07-11 사고 재발 방지용 wrapper: 백엔드가 vision_ws/robot_ws를 소싱하지 않은
# 셸(예: ~/.bashrc 기본값인 ~/cobot_ws만 소싱된 새 터미널)에서 그냥 `uvicorn
# main:app --reload`로 직접 떠버리면, apple_care_msgs를 import 못 해
# vision_bridge.py가 조용히 자기 자신을 비활성화한다(vision_bridge.py:56-71) -
# 그 결과 /fastapi_vision_bridge 노드 자체가 안 생기고 YOLO/unknown 감지 데이터가
# 백엔드로 한 건도 안 들어오는데, uvicorn은 정상 기동된 것처럼 보여서 원인 파악이
# 매우 오래 걸렸다(그날 겪은 실제 장애).
#
# 이 스크립트는 uvicorn을 직접 실행하지 않고 반드시 이 wrapper를 거치도록 강제해서,
# 필요한 워크스페이스가 이미 소싱된 상태인지 기동 전에 확인하고 아니면 즉시 에러로
# 중단한다. 자동으로 source하지 않는 이유: 조용히 알아서 고쳐버리면 "안 하고 넘어가도
# 되는구나"가 되어 같은 실수가 반복되기 쉬움 - 실패를 명확하게 드러내는 쪽을 택함.
#
# 사용법:
#   source /home/rokey/apple-care/code/vision_ws/install/setup.bash
#   source /home/rokey/apple-care/code/robot_ws/install/setup.bash
#   /home/rokey/apple-care/code/backend/run_backend.sh

set -euo pipefail

_fail() {
    echo "[run_backend.sh] FATAL: $1" >&2
    echo "[run_backend.sh] 아래 순서로 다시 소싱한 뒤 이 스크립트를 실행하세요:" >&2
    echo "    source /home/rokey/apple-care/code/vision_ws/install/setup.bash" >&2
    echo "    source /home/rokey/apple-care/code/robot_ws/install/setup.bash" >&2
    echo "    $0" >&2
    exit 1
}

case "${AMENT_PREFIX_PATH:-}" in
    *vision_ws*) ;;
    *) _fail "vision_ws가 소싱되지 않았습니다 (AMENT_PREFIX_PATH에 vision_ws 없음)." ;;
esac

case "${AMENT_PREFIX_PATH:-}" in
    *robot_ws*) ;;
    *) _fail "robot_ws가 소싱되지 않았습니다 (AMENT_PREFIX_PATH에 robot_ws 없음)." ;;
esac

# PATH 문자열 매칭만으로는 실제 import 가능 여부를 보장하지 못하므로(경로만 있고
# 빌드가 안 됐거나 깨진 경우 등) apple_care_msgs를 실제로 import해서 다시 한번 확인.
if ! python3 -c "import apple_care_msgs.srv" 2>/dev/null; then
    _fail "apple_care_msgs를 import할 수 없습니다 (vision_ws가 소싱됐지만 빌드가 안 됐을 수 있음)."
fi

echo "[run_backend.sh] vision_ws/robot_ws 소싱 확인 완료, apple_care_msgs import 성공 - 백엔드를 기동합니다."

cd "$(dirname "$0")"
exec uvicorn main:app --reload
