# Task2: 사과 1개씩 순차 실물 드라이런 - 실시간 모니터링 하네스
#
# tb_decision_audit에 새 행이 쌓이는 것을 폴링으로 감지해 즉시 콘솔에 출력하고,
# (rclpy를 쓸 수 있는 환경이면) /decision/result, /robot/command ROS2 토픽도
# 함께 echo한다.
#
# 범위 밖: 이 스크립트는 "사과 1개를 순차적으로 처리"하는 드라이런 전용이다.
# 여러 개를 동시에 투입하는 시나리오에서는 tb_decision_audit 행들이 서로 다른
# 물체의 판정이 인터리빙되어 찍힐 수 있는데, 이 스크립트는 그런 다중 아이템
# 동시 처리 시나리오를 구분/정리해서 보여주도록 설계되지 않았다 (§5.1 본
# 실험에서 별도 하네스로 다뤄야 할 범위).
#
# 사용법:
#   cd code/backend && python3 scripts/dry_run_monitor.py
#   cd code/backend && python3 scripts/dry_run_monitor.py --poll-interval 0.3 --log-file /tmp/dry_run.log

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# scripts/ 하위에서 실행돼도 backend/ 루트(database, config)를 import할 수 있도록.

import database

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


SEPARATOR = "-" * 90


def _format_row(row: sqlite3.Row) -> str:
    """tb_decision_audit 행 하나를 branch별로 핵심 필드만 골라 사람이 읽기
    좋은 여러 줄로 정리. 전체 30개 컬럼을 다 찍으면 콘솔이 너무 시끄러워서,
    services/decision_audit.py의 각 log_*() 함수가 실제로 채우는 필드
    조합만 branch별로 골라 보여준다."""
    d = dict(row)
    lines = [
        f"[audit_id={d['audit_id']:>4}] {d['timestamp']}  branch={d['branch']}",
        f"    fruit_type={d['fruit_type']!r}  condition={d['condition']!r}  "
        f"c_yolo={d['c_yolo']}  position={d['position']}",
    ]

    if d["branch"] == "stage2_vlm_call":
        lines.append(
            f"    vlm_called={bool(d['vlm_called'])}  vlm_response={d['vlm_response']}"
        )
    if d["branch"] in ("stage3_query_human", "stage3_risk_accept_execute"):
        lines.append(
            f"    theta_exists={d['theta_exists']}  dstored={d['dstored']!r}  "
            f"evpi_human={d['evpi_human']}  cost_human={d['cost_human']}  "
            f"n_pending={d['n_pending']}  tau_hold_sec={d['tau_hold_sec']}"
        )
    if d["branch"] == "stage3_human_resolved":
        lines.append(
            f"    posterior: alpha {d['alpha_before']} -> {d['alpha_after']}   "
            f"beta {d['beta_before']} -> {d['beta_after']}   session_id={d['session_id']}"
        )
    if d["branch"] == "prior_initialized":
        lines.append(
            f"    init_method={d['prior_init_method']!r}  "
            f"pooled_mean={d['prior_pooled_mean']}  "
            f"pooled_sample_size={d['prior_pooled_sample_size']}  "
            f"alpha={d['alpha_after']}  beta={d['beta_after']}"
        )

    lines.append(
        f"    query_human={d['query_human']}  final_action={d['final_action']!r}  "
        f"final_destination={d['final_destination']!r}"
    )
    return "\n".join(lines)


class DryRunMonitor:
    """tb_decision_audit 폴링 + 콘솔/로그파일 출력을 담당하는 공용 헬퍼.
    ROS 사용 가능 여부와 무관하게 동일하게 동작 (TopicEchoNode가 있으면 타이머
    콜백으로, 없으면 단순 while 루프로 poll_once()를 호출)."""

    def __init__(self, poll_interval: float, log_file: Optional[TextIO]):
        self.poll_interval = poll_interval
        self.log_file = log_file
        self._last_audit_id = self._get_max_audit_id()
        self._emit(
            f"드라이런 모니터링 시작 (poll_interval={poll_interval}s, "
            f"DB={database.DB_PATH}, 시작 시점 최신 audit_id={self._last_audit_id})"
        )
        self._emit(
            "주의: 사과 1개 순차 처리 전제 - 다중 아이템 동시 투입 시나리오는 "
            "이 하네스의 검증 범위 밖입니다."
        )

    def _get_max_audit_id(self) -> int:
        conn = database.get_db_connection()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(audit_id), 0) AS m FROM tb_decision_audit"
            ).fetchone()
            return row["m"]
        finally:
            conn.close()

    def poll_once(self) -> None:
        conn = database.get_db_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM tb_decision_audit WHERE audit_id > ? ORDER BY audit_id",
                (self._last_audit_id,),
            ).fetchall()
        finally:
            conn.close()

        for row in rows:
            self._last_audit_id = row["audit_id"]
            self._emit(_format_row(row))

    def emit_ros_message(self, topic: str, data: str) -> None:
        self._emit(f"[ROS {topic}] {data}")

    def _emit(self, text: str) -> None:
        print(text)
        print(SEPARATOR)
        if self.log_file:
            self.log_file.write(text + "\n" + SEPARATOR + "\n")
            self.log_file.flush()


if ROS_AVAILABLE:

    class TopicEchoNode(Node):
        """/decision/result, /robot/command를 구독해서 monitor로 그대로 넘기고,
        타이머로 monitor.poll_once()도 같이 돌린다 (스레드 없이 rclpy 콜백
        하나로 DB 폴링 + 토픽 echo를 모두 처리 - 이 코드베이스의 다른 노드들이
        느린 작업과 빠른 작업을 콜백/타이머로 나누는 관례와 동일)."""

        def __init__(self, monitor: DryRunMonitor):
            super().__init__("dry_run_monitor")
            self.monitor = monitor
            self.create_subscription(
                String, "/decision/result",
                lambda msg: self.monitor.emit_ros_message("/decision/result", msg.data),
                10,
            )
            self.create_subscription(
                String, "/robot/command",
                lambda msg: self.monitor.emit_ros_message("/robot/command", msg.data),
                10,
            )
            self.create_timer(monitor.poll_interval, monitor.poll_once)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tb_decision_audit 실시간 tail + (가능하면) ROS 토픽 echo"
    )
    parser.add_argument("--poll-interval", type=float, default=0.5, help="DB 폴링 주기(초)")
    parser.add_argument("--log-file", default=None, help="콘솔 출력을 그대로 append할 로그 파일 경로")
    args = parser.parse_args()

    log_file = open(args.log_file, "a", encoding="utf-8") if args.log_file else None
    monitor = DryRunMonitor(args.poll_interval, log_file)

    try:
        if ROS_AVAILABLE:
            rclpy.init()
            node = TopicEchoNode(monitor)
            print("ROS2 토픽 echo 활성화: /decision/result, /robot/command")
            print(SEPARATOR)
            try:
                rclpy.spin(node)
            finally:
                node.destroy_node()
                rclpy.shutdown()
        else:
            print("rclpy를 가져올 수 없어 DB 폴링만 수행합니다 (ROS2 소싱이 안 됐다면 "
                  "source /opt/ros/humble/setup.bash 후 재실행하면 토픽 echo도 켜집니다).")
            print(SEPARATOR)
            while True:
                monitor.poll_once()
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n모니터링 종료 (Ctrl+C).")
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
