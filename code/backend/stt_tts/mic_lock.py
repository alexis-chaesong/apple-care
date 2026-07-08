# 마이크 동시 사용 방지용 공유 Lock

"""
stt_tts/mic_lock.py
=====================
마이크를 쓰는 세 주체가 동시에 마이크 장치를 열지 않도록 공유하는 전역 Lock.

  1) wakeup_listener.py의 상시 웨이크워드 대기 루프 (pyaudio)
  2) state/hitl_state_machine.py의 HITL 질문-응답 STT 리스닝 (sounddevice)
  3) services/voice_policy.py의 음성 정책 명령 처리 (sounddevice, 웨이크워드
     감지 또는 HMI push-to-talk 버튼으로 트리거됨)

세 주체 모두 sounddevice(stt_service)와 pyaudio(wakeup_listener)라는 서로 다른
라이브러리로 같은 물리 마이크 장치를 열려고 시도할 수 있어, 하나가 마이크를
쓰는 동안 나머지는 반드시 대기해야 함. bridge_manager, connection_manager와
동일한 모듈 레벨 싱글톤 패턴.
"""

import asyncio

mic_lock = asyncio.Lock()
