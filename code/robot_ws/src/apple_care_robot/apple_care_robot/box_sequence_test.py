"""
box_sequence_test
====================

원래는 백엔드(비전/음성/HITL) 연동 없이 "그리퍼가 제대로 잡는지, 박스 위치가
맞는지"만 확인하던 독립 하드웨어 검증 스크립트였음. 로컬에서 get_apple_status를
직접 호출하고, unknown이면 그냥 포기하는 자체 재시도 로직(STATUS_TO_BOX_NAME 직접
매핑, wait_for_confident_detection 등)을 갖고 있어서 /decision/result나
/robot/process_state 같은 백엔드 연동 토픽을 전혀 모르는 상태였음.

그 결과 이 스크립트를 실행 중일 때는 HITL 질문이 뜨지 않고, 음성으로 정책을
등록해도 반영할 방법이 없었음(백엔드와 아예 연결돼 있지 않았으므로) - 실제로
겪은 문제. ros2 run 진입점 이름(box_sequence_test)은 그대로 유지하되, 실행
내용은 백엔드와 완전히 연동된 apple_sorting_cycle과 동일하게 맞춤.

기존 독립 검증 로직(STATUS_TO_BOX_NAME 직접 매핑, MAX_DETECTION_ATTEMPTS=5 재시도
등)이 다시 필요해지면, 이 커밋 이전 버전을 git 이력에서 그대로 복구하면 됨.
"""

from apple_care_robot.apple_sorting_cycle import main

if __name__ == "__main__":
    main()
