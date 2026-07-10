# tb_policy_memory, tb_human_feedback 테이블 추가

import sqlite3
import os

# 데이터 저장 경로 지정 (프로젝트 루트의 data 폴더 내)
DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/robot_system.db")

def get_db_connection():
    """DB 연결 객체를 반환하는 함수"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 데이터를 읽어오기 설정
    return conn

def init_db():
    """앱 구동 시 테이블을 생성하고 초기 세팅값을 넣는 함수"""
    # data 폴더가 없으면 생성
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. 배출 모드 마스터 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tb_dump_modes (
            mode_id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode_name TEXT NOT NULL,
            tilt_angle INTEGER NOT NULL,
            shake_count INTEGER NOT NULL
        );
    """)
    
    # 2. 배출 작업 이력 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tb_dump_history (
            task_id TEXT PRIMARY KEY,
            mode_id INTEGER,
            start_time TEXT DEFAULT (datetime('now', 'localtime')),
            end_time TEXT,
            status TEXT NOT NULL,
            FOREIGN KEY (mode_id) REFERENCES tb_dump_modes(mode_id)
        );
    """)
    
    # 3. 에러 로그 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tb_error_log (
            error_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            error_code TEXT NOT NULL,
            error_msg TEXT NOT NULL,
            error_time TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (task_id) REFERENCES tb_dump_history(task_id)
        );
    """)
    
    # 초기 마스터 데이터 주입 (데이터가 없을 때만 초기화)
    cursor.execute("SELECT COUNT(*) FROM tb_dump_modes")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO tb_dump_modes (mode_name, tilt_angle, shake_count) VALUES ('유형1: 일반 배출 + 세척', 45, 5)")
        cursor.execute("INSERT INTO tb_dump_modes (mode_name, tilt_angle, shake_count) VALUES ('유형2: 강하게 털기 + 세척', 60, 10)")
        print("💡 [DB] 초기 배출 모드 데이터 등록 완료!")

    # ------------------------------------------------------------------
    # 아래부터 신규 추가: VLA 프로젝트용 테이블 2종
    # conn.commit() / conn.close()가 호출되기 전에 실행되어야 하므로
    # 기존 함수의 commit/close 호출부(맨 아래) 앞으로 이동시킴
    # ------------------------------------------------------------------

    # 4. Policy Memory 테이블 생성
    # fruit_type + condition 조합이 하나의 정책 단위.
    # 예: ("apple", "small") -> "normal_box", ("kiwi", "unknown") -> "processing_box"
    #
    # alpha, beta는 Bayesian Beta 분포 파라미터.
    # source가 "human_feedback"인 행만 매 피드백마다 alpha/beta가 갱신되고,
    # source가 "llm_policy"인 행(작업자가 자연어로 직접 지정한 규칙)은
    # alpha/beta를 쓰지 않고 항상 확정 규칙으로 취급함 (confidence=1.0 고정)
    #
    # PRIMARY KEY를 (fruit_type, condition) 복합키로 잡아서
    # 같은 조합에 대한 정책이 여러 행으로 중복 저장되는 것을 DB 레벨에서 원천 차단함
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tb_policy_memory (
            fruit_type TEXT NOT NULL,
            condition TEXT NOT NULL,
            destination TEXT NOT NULL,
            alpha REAL NOT NULL DEFAULT 1.0,
            beta REAL NOT NULL DEFAULT 1.0,
            confidence REAL NOT NULL DEFAULT 0.5,
            source TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now', 'localtime')),
            PRIMARY KEY (fruit_type, condition)
        );
    """)

    # 5. Human Feedback 이력 테이블 생성
    # tb_policy_memory가 "현재 최종 결론"만 담는 반면,
    # 이 테이블은 "언제 누가 무슨 답을 했는지"의 원본 이력을 전부 남김
    #
    # 목적: (1) LLM 해석이 잘못됐을 때 원인 추적
    #       (2) 데모/발표 때 "실제로 몇 번 물어보고 몇 번째부터 자동화됐는지" 시연 근거
    #       (3) 추후 alpha/beta 재계산이 필요할 때(버그 수정 등) 원본 데이터로 재구성 가능
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tb_human_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            fruit_type TEXT NOT NULL,
            destination TEXT NOT NULL,
            raw_answer TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
    """)

    # 초기 마스터 정책 데이터 주입 (데이터가 없을 때만 초기화)
    # 작업자가 아직 아무 자연어 명령도 안 내렸을 때를 대비한 Prior Policy.
    # Priority Rule(Discard > Damaged > Processing > Normal)에서
    # 가장 보수적인 기본값으로 세팅해둠 -> 위험한 오분류보다는 사람에게 묻는 쪽으로 기본값을 잡음
    cursor.execute("SELECT COUNT(*) FROM tb_policy_memory")
    if cursor.fetchone()[0] == 0:
        default_policies = [
            # source가 llm_policy인 행은 위쪽 주석대로 confidence=1.0 고정(확정 규칙)으로 넣어야
            # decision_planner.decide()가 bayesian_auto_threshold와 무관하게 바로 execute함
            ("apple", "normal", "normal_box", "llm_policy", 1.0),
            ("apple", "mold", "discard_box", "llm_policy", 1.0),
            ("apple", "scratch", "processing_box", "llm_policy", 1.0),
            ("apple", "bruise", "processing_box", "llm_policy", 1.0),
            # unknown 조건은 alpha/beta 학습 대상이므로 source를 human_feedback으로 잡지 않고
            # 시스템 기본 규칙(ask_human)으로 시작. 첫 피드백부터 학습이 시작됨
            ("apple", "unknown", "ask_human", "llm_policy", 1.0),
        ]
        cursor.executemany("""
            INSERT INTO tb_policy_memory (fruit_type, condition, destination, source, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, default_policies)
        print("💡 [DB] 초기 Policy Memory 기본값 등록 완료!")

    conn.commit()
    conn.close()
    print("✅ [DB] SQLite 데이터베이스 초기화 및 테이블 생성 성공")