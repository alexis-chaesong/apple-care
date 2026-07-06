import tkinter as tk
from tkinter import messagebox, ttk
import json
import os
import queue
import sys
import threading
from datetime import datetime
import cv2  # OpenCV 추가
from PIL import Image, ImageTk  # Pillow 추가
import requests

# dashboard.py는 code/hmi_app/view/ 아래에 있음. hmi_app을 패키지로 import하려면
# 그 부모 디렉토리인 code/ 가 sys.path에 있어야 하므로, cwd가 어디든 상관없이
# python3 dashboard.py로 직접 실행해도 동작하도록 부트스트랩해둠
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from hmi_app.api import client

# UI 표시용 라벨(4분류: 판매/못난이/가공용/폐기)과 backend destination 값을 분리하는 매핑.
# "못난이"는 backend에도 4번째 destination(ugly_box)으로 추가되어 있음 (models.py,
# services/llm_service.py의 interpret_human_answer/translate_policy_command 참고)
CATEGORY_TO_DESTINATION = {
    "판매": "normal_box",
    "못난이": "ugly_box",
    "가공용": "processing_box",
    "폐기": "discard_box",
}


def _category_to_raw_answer(category: str) -> str:
    """카테고리 라벨을 자연어 raw_answer로 변환. 받침(종성) 유무에 따라
    "으로"/"로" 조사를 올바르게 붙임 (예: "가공용" -> "가공용으로 보내")"""
    code = ord(category[-1]) - 0xAC00
    has_batchim = 0 <= code < 11172 and code % 28 != 0
    particle = "으로" if has_batchim else "로"
    return f"{category}{particle} 보내"
DESTINATION_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_DESTINATION.items()}


class VLASorterDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RGB-D Multi-View 기반 예외 농산물 판단 및 로봇 분류 시스템")
        self.root.configure(bg="#2c3e50")

        # 분류 통계 데이터
        self.counts = {"판매": 124, "못난이": 42, "가공용": 15, "폐기": 8}

        # 현재 화면에 떠 있는 HITL 팝업이 어떤 판정에 대한 것인지 추적하는 상태.
        # WebSocket으로 받은 VLA_ASK_HUMAN payload의 fruit_type/condition/session_id를
        # 팝업이 살아있는 동안 들고 있다가 manual_sort()에서 피드백 전송 시 사용
        self.hitl_fruit_type = None
        self.hitl_condition = None
        self.hitl_session_id = None
        self.hitl_popup = None  # 현재 열려있는 HITL 팝업 창(Toplevel) 참조. HITL_RESOLVED 도착 시 자동으로 닫는 데 사용

        # UI 스타일 테마 설정
        style = ttk.Style()
        style.theme_use('clam')

        # backend WebSocket(ws://localhost:8000/ws/robot/status) 연결을 백그라운드
        # daemon thread에서 유지하고, 수신 메시지는 ws_queue에 쌓인다.
        # Tkinter는 메인 스레드만 위젯을 건드릴 수 있으므로, 이 스레드는 큐에
        # 적재만 하고 위젯 갱신은 poll_ws_queue()가 메인 스레드에서 담당한다.
        self.ws_queue = queue.Queue()
        self.ws_client = client.WebSocketClient(self.ws_queue)
        self.ws_client.start()
        self.root.after(100, self.poll_ws_queue)

        # [추가] OpenCV 웹캠 초기화 (0은 기본 내장 캠)
        self.cap = cv2.VideoCapture(0)
        
        # 메인 컨테이너 레이아웃 구성
        self.create_top_bar()
        self.create_side_bar()
        
        # 메인 콘텐츠 영역 (사이드바 메뉴 전환용 포인터)
        self.content_area = tk.Frame(self.root, bg="#2c3e50")
        self.content_area.pack(side="right", fill="both", expand=True, padx=15, pady=15)
        
        # 각 탭 프레임 딕셔너리
        self.frames = {}
        self.create_frames()
        
        # 최초 기본 화면: 모니터링 대시보드
        self.show_frame("monitor")
        
        # 메인 윈도우 중앙 정렬 (1400x800)
        self.center_window(1400, 800)
        
        # [추가] 프로그램 종료 시 웹캠 자원을 안전하게 해제하기 위한 바인딩
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.log_message("System initialized. UI navigation elements loaded.")

    def center_window(self, width=1400, height=800):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def center_popup(self, popup, width=500, height=350):
        popup.update_idletasks()
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        popup.geometry(f"{width}x{height}+{x}+{y}")

    # 0. WebSocket 수신 큐 폴링 (100ms마다 메인 스레드에서 실행)
    def poll_ws_queue(self):
        try:
            while True:
                msg = self.ws_queue.get_nowait()
                self._handle_ws_message(msg)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_ws_queue)

    def _handle_ws_message(self, msg):
        msg_type = msg.get("type")

        if msg_type == "VLA_DECISION":
            payload = msg.get("payload", {}) or {}
            destination = payload.get("destination")
            category = DESTINATION_TO_CATEGORY.get(destination)
            if category is not None:
                self.counts[category] = self.counts.get(category, 0) + 1
                if category in self.count_labels:
                    self.count_labels[category].config(text=f"{self.counts[category]} 개")
            self.log_message(
                f"[VLA_DECISION] fruit={payload.get('fruit_type')} "
                f"condition={payload.get('condition')} destination={destination} "
                f"reason={payload.get('reason')} confidence={payload.get('confidence')}"
            )

        elif msg_type == "VLA_ASK_HUMAN":
            payload = msg.get("payload", {}) or {}
            self.log_message(
                f"[VLA_ASK_HUMAN] fruit={payload.get('fruit_type')} "
                f"condition={payload.get('condition')} confidence={payload.get('confidence')} "
                "- HITL 팝업 자동 표시"
            )
            self.trigger_hitl_popup(payload)

        elif msg_type == "HITL_RESOLVED":
            payload = msg.get("payload", {}) or {}
            self.log_message(
                f"[HITL_RESOLVED] 해결됨: {payload.get('fruit_type')} → {payload.get('destination')} "
                f"(condition={payload.get('condition')}, confidence={payload.get('confidence')})"
            )
            if self.hitl_popup is not None:
                try:
                    if self.hitl_popup.winfo_exists():
                        self.hitl_popup.destroy()
                except tk.TclError:
                    pass
                self.hitl_popup = None

        elif msg_type == "HITL_STUCK":
            payload = msg.get("payload", {}) or {}
            fruit_type = payload.get("fruit_type")
            self.log_message(
                f"[HITL_STUCK] 자동 재질문 실패, 수동 개입 필요: fruit={fruit_type} "
                f"condition={payload.get('condition')} session_id={payload.get('session_id')}"
            )
            messagebox.showwarning(
                "HITL 재질문 실패",
                f"자동 재질문 실패, 수동 개입 필요: {fruit_type}\n"
                "로봇/시스템 제어 탭에서 강제 복구가 필요합니다.",
            )

    # 1. 상단 글로벌 제어 및 상태 바 (상시 노출)
    def create_top_bar(self):
        top_bar = tk.Frame(self.root, bg="#1a252f", height=60)
        top_bar.pack(fill="x", side="top")
        
        title_label = tk.Label(top_bar, text="★ VLA SORTING SYSTEM HMI ★", 
                               font=("Helvetica", 14, "bold"), fg="#ecf0f1", bg="#1a252f")
        title_label.pack(side="left", padx=20, pady=15)
        
        self.current_tab_lbl = tk.Label(top_bar, text="- 메인 모니터링", font=("Helvetica", 12), fg="#3498db", bg="#1a252f")
        self.current_tab_lbl.pack(side="left", pady=17)
        
        self.status_label = tk.Label(top_bar, text="SYSTEM STATUS: RUNNING", 
                                     font=("Helvetica", 11, "bold"), fg="#2ecc71", bg="#1a252f")
        self.status_label.pack(side="right", padx=20, pady=18)

    # 2. 좌측 메뉴 내비게이션 사이드바 (상시 노출)
    def create_side_bar(self):
        side_bar = tk.Frame(self.root, bg="#34495e", width=200)
        side_bar.pack(side="left", fill="y")
        side_bar.pack_propagate(False)

        logo_lbl = tk.Label(side_bar, text="VLA Control", font=("Helvetica", 14, "bold"), fg="#ecf0f1", bg="#2c3e50", pady=15)
        logo_lbl.pack(fill="x", side="top")

        def menu_btn(text, target):
            return tk.Button(
                side_bar, text=text, font=("Helvetica", 11, "bold"), bg="#34495e", fg="white",
                activebackground="#1abc9c", activeforeground="white", bd=0, relief="flat", height=2, anchor="w", padx=20,
                command=lambda: self.show_frame(target)
            )

        menu_btn("실시간 모니터링", "monitor").pack(fill="x", pady=2)
        menu_btn("VLA 정책 관리", "policy").pack(fill="x", pady=2)
        menu_btn("로봇/시스템 제어", "control").pack(fill="x", pady=2)

        lbl_div = tk.Label(side_bar, text="", bg="#2c3e50", height=1)
        lbl_div.pack(fill="x", pady=20)

        btn_hitl_wait = tk.Button(side_bar, text="작업자 지시\n대기", bg="#e67e22", fg="white", font=("Helvetica", 11, "bold"), height=3, command=self.trigger_hitl_popup)
        btn_hitl_wait.pack(fill="x", side="bottom", padx=10, pady=(5, 10))

        btn_estop = tk.Button(side_bar, text="EMERGENCY\nSTOP", bg="#e74c3c", fg="white", font=("Helvetica", 11, "bold"), height=3, command=self.emergency_stop)
        btn_estop.pack(fill="x", side="bottom", padx=10, pady=(5, 20))

    # 3. 사이드바 전환을 위한 프레임 구조셋 생성
    def create_frames(self):
        f_monitor = tk.Frame(self.content_area, bg="#2c3e50")
        self.setup_monitor_tab(f_monitor)
        self.frames["monitor"] = (f_monitor, "- 메인 모니터링")
        
        f_policy = tk.Frame(self.content_area, bg="#2c3e50")
        self.setup_policy_tab(f_policy)
        self.frames["policy"] = (f_policy, "- VLA 자연어 정책 관리")
        
        f_control = tk.Frame(self.content_area, bg="#2c3e50")
        self.setup_control_tab(f_control)
        self.frames["control"] = (f_control, "- 시스템 운영 및 제어")

    def show_frame(self, target):
        for f, _ in self.frames.values():
            f.pack_forget()
        frame, title = self.frames[target]
        frame.pack(fill="both", expand=True)
        self.current_tab_lbl.config(text=title)

    def setup_monitor_tab(self, parent):
        cam_frame = tk.Frame(parent, bg="#2c3e50")
        cam_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # [수정] Cam 1: Intel RealSense D455 대용 -> 노트북 웹캠 출력용 Label로 변경
        self.frame_d455 = tk.LabelFrame(cam_frame, text="Cam 1: Laptop Webcam (Real-time View)", font=("Helvetica", 11, "bold"), fg="#ecf0f1", bg="#34495e", bd=2)
        self.frame_d455.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # [수정] 웹캠 영상 컨테이너: 창 크기에 맞춰 '채우도록'(fill+expand) 배치하되,
        # pack_propagate(False)로 안에 들어가는 이미지 크기가 이 칸 크기에 영향을 주지 못하게 못박음
        # -> "틀은 방 크기에 맞춰 늘어나지만, 사진이 틀을 다시 늘리진 못한다"
        webcam_container = tk.Frame(self.frame_d455, bg="#000000")
        webcam_container.pack(fill="both", expand=True, padx=5, pady=5)
        webcam_container.pack_propagate(False)
        self.webcam_container = webcam_container

        # [추가] 컨테이너의 실제 크기가 바뀔 때만(=창 리사이즈 등 레이아웃 변경 시에만) 갱신
        # 매 프레임마다 크기를 재는 대신, 크기가 "진짜" 바뀔 때 이벤트로 받아서 저장해둠
        self.cam_box_w = 480
        self.cam_box_h = 360
        webcam_container.bind("<Configure>", self.on_webcam_container_resize)

        self.lbl_webcam = tk.Label(webcam_container, bg="#000000")
        self.lbl_webcam.pack(fill="both", expand=True)
        
        # Cam 2: C720 (가상 데이터 스캔 영역 유지)
        frame_c720 = tk.LabelFrame(cam_frame, text="Cam 2: Logitech C720 (Side View)", font=("Helvetica", 11, "bold"), fg="#ecf0f1", bg="#34495e", bd=2)
        frame_c720.pack(side="right", fill="both", expand=True, padx=(5, 0))
        lbl_c720 = tk.Label(frame_c720, text="Side Defect Scan Area\n- Secondary inspection ready.\n- Normal Angle Alignment.", font=("Consolas", 11), fg="#bdc3c7", bg="#000000")
        lbl_c720.pack(fill="both", expand=True, padx=5, pady=5)

        self.create_counter_table(parent)
        
        # [수정] 창이 완전히 그려져서 위젯 크기가 확정된 뒤에 스트리밍 루프 시작
        # (레이아웃이 안 잡힌 상태에서 바로 시작하면 winfo_width가 엉뚱한 값을 줘서 화면이 찌그러짐)
        self.root.update_idletasks()
        self.root.after(100, self.update_webcam)

    # [추가] 컨테이너 크기가 "진짜" 바뀔 때만 호출됨 (창 리사이즈, 레이아웃 변경 시)
    # 이미지 갱신 때문에 호출되는 게 아니라, pack_propagate(False)로 막아뒀기 때문에
    # 오직 창/레이아웃 변화로만 트리거됨 -> 피드백 루프 없음
    def on_webcam_container_resize(self, event):
        self.cam_box_w = max(event.width, 10)
        self.cam_box_h = max(event.height, 10)

    # [수정] 실시간 웹캠 프레임 업데이트 함수 - 비율 유지(letterbox) 방식으로 변경
    def update_webcam(self):
        if hasattr(self, 'lbl_webcam') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 1. 좌우 반전 (거울 모드)
                frame = cv2.flip(frame, 1)

                # 2. 담을 칸의 크기 - Configure 이벤트로 갱신된 값 사용 (창 크기에 맞춰 채우되, 이미지가 이 값을 다시 바꾸진 못함)
                box_w = self.cam_box_w
                box_h = self.cam_box_h

                frame_h, frame_w = frame.shape[:2]

                # 3. 원본 비율을 유지하면서 칸 안에 '꽉 차게, 잘리지 않게' 들어갈 배율 계산
                #    (가로에 맞추면 세로가 남고, 세로에 맞추면 가로가 남을 수 있음 -> 더 작은 쪽 배율 선택)
                scale = min(box_w / frame_w, box_h / frame_h)
                new_w, new_h = int(frame_w * scale), int(frame_h * scale)
                resized = cv2.resize(frame, (new_w, new_h))

                # 4. 칸 크기의 검은 캔버스를 만들고 가운데에 영상 붙이기 (레터박싱)
                #    -> 세로로 긴 액자에 정사각 사진을 억지로 늘리는 대신,
                #       액자 크기에 맞는 검은 매트를 깔고 사진은 원래 비율 그대로 가운데 배치
                canvas = cv2.copyMakeBorder(
                    resized,
                    top=(box_h - new_h) // 2,
                    bottom=box_h - new_h - (box_h - new_h) // 2,
                    left=(box_w - new_w) // 2,
                    right=box_w - new_w - (box_w - new_w) // 2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)
                )

                # 5. OpenCV(BGR) -> Tkinter(RGB) 변환 후 표시
                cv2image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)

                self.lbl_webcam.imgtk = imgtk  # 가비지 컬렉션 방지 참조 유지
                self.lbl_webcam.config(image=imgtk)

        # 15ms(약 60fps) 마다 자기 자신을 재호출
        self.root.after(15, self.update_webcam)

    def setup_policy_tab(self, parent):
        frame = tk.LabelFrame(parent, text="■ VLA NATURAL LANGUAGE POLICY INPUT", font=("Helvetica", 12, "bold"), fg="#ecf0f1", bg="#34495e", bd=2)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        lbl_guide = tk.Label(frame, text="작업 현장의 가변적인 처리 기준을 자연어로 입력하세요:", font=("Malgun Gothic", 11), fg="#ecf0f1", bg="#34495e")
        lbl_guide.pack(anchor="w", padx=15, pady=10)

        self.policy_text = tk.Text(frame, height=5, font=("Malgun Gothic", 11))
        self.policy_text.insert("1.0", "오늘은 작은 사과도 판매하고, 곰팡이 의심되는 것만 무조건 폐기해줘. 애매하면 물어봐.")
        self.policy_text.pack(fill="x", padx=15, pady=5)

        self.btn_convert = tk.Button(frame, text="자연어 정책 파싱 및 JSON 변환", bg="#3498db", fg="white", font=("Helvetica", 11, "bold"), pady=8, command=self.apply_policy)
        self.btn_convert.pack(anchor="e", padx=15, pady=10)

        lbl_json_title = tk.Label(frame, text="▼ ROS2 가동용 의사결정 Rule 변환 결과 (JSON)", font=("Helvetica", 11, "bold"), fg="#ecf0f1", bg="#34495e")
        lbl_json_title.pack(anchor="w", padx=15, pady=(10, 5))

        self.json_output = tk.Label(frame, text="{\n  \"target_policy\": \"flexible_grading\",\n  \"rules\": {\n    \"small_size\": \"sale\",\n    \"mold_suspect\": \"discard\",\n    \"low_confidence\": \"ask_human\"\n  }\n}", 
                                    font=("Consolas", 11), fg="#2ecc71", bg="#1a252f", justify="left", anchor="w", padx=15, pady=15)
        self.json_output.pack(fill="both", expand=True, padx=15, pady=15)

    def setup_control_tab(self, parent):
        ctrl_frame = tk.LabelFrame(parent, text="SYSTEM HARDWARE CONTROL", font=("Helvetica", 12, "bold"), fg="#ecf0f1", bg="#34495e", bd=2)
        ctrl_frame.pack(fill="x", padx=10, pady=10)

        btn_start = tk.Button(ctrl_frame, text="START ROBOT SYSTEM", bg="#2ecc71", fg="white", font=("Helvetica", 11, "bold"), width=22, pady=10, command=self.on_start_robot)
        btn_start.pack(side="left", padx=15, pady=15)

        btn_pause = tk.Button(ctrl_frame, text="PAUSE ROBOT MOTION", bg="#f1c40f", fg="black", font=("Helvetica", 11, "bold"), width=22, pady=10, command=self.on_pause_robot)
        btn_pause.pack(side="left", padx=15, pady=15)

        log_frame = tk.LabelFrame(parent, text="실시간 데이터 및 이벤트 로그 (ROS2 로그 매핑 가능)", font=("Helvetica", 12, "bold"), fg="#ecf0f1", bg="#34495e", bd=2)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.log_box = tk.Text(log_frame, font=("Consolas", 10), fg="#ecf0f1", bg="#1a252f")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)

    def create_counter_table(self, parent):
        frame = tk.LabelFrame(parent, text="SORTING COUNT (실시간 분류 통계)", font=("Helvetica", 11, "bold"), fg="#ecf0f1", bg="#34495e", bd=2)
        frame.pack(fill="x", side="bottom", pady=(10, 0))

        self.count_labels = {}
        for i, (category, count) in enumerate(self.counts.items()):
            frame.grid_columnconfigure(i, weight=1)
            
            lbl_title = tk.Label(frame, text=category, font=("Helvetica", 11, "bold"), fg="#ecf0f1", bg="#34495e")
            lbl_title.grid(row=0, column=i, padx=5, pady=8, sticky="nsew")
            
            lbl_cnt = tk.Label(frame, text=f"{count} 개", font=("Helvetica", 16, "bold"), fg="#f1c40f", bg="#2c3e50", relief="sunken", pady=10)
            lbl_cnt.grid(row=1, column=i, padx=5, pady=8, sticky="nsew")
            self.count_labels[category] = lbl_cnt

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'log_box'):
            self.log_box.insert("end", f"[{timestamp}] {message}\n")
            self.log_box.see("end")

    def apply_policy(self):
        text = self.policy_text.get("1.0", "end-1c").strip()
        if not text:
            return

        self.log_message(f"[VLA NLP Update] 정책 명령 전송 중: '{text}'")
        self.btn_convert.config(state="disabled")

        def worker():
            try:
                result = client.post_policy_command(text)
                pretty = json.dumps(result.get("applied_policies", result), ensure_ascii=False, indent=2)

                def on_success():
                    self.json_output.config(text=pretty)
                    self.log_message(f"[VLA NLP Update] 정책 반영 완료: {result.get('applied_policies')}")
                    self.btn_convert.config(state="normal")

                self.root.after(0, on_success)

            except requests.exceptions.RequestException as exc:
                def on_error():
                    self.log_message(f"[VLA NLP Error] 정책 명령 처리 실패: {exc}")
                    self.btn_convert.config(state="normal")

                self.root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    # TODO: routers/robot_router.py에 실제 START 엔드포인트가 생기면
    # client.post_robot_start()를 threading.Thread로 호출하도록 연결 필요
    def on_start_robot(self):
        self.log_message(
            "System process tracking active. ROS2 node online. "
            "(TODO: 실제 로봇 START 엔드포인트 확인 후 연결 필요 - robot_router.py가 현재 빈 라우터)"
        )

    # TODO: routers/robot_router.py에 실제 PAUSE 엔드포인트가 생기면
    # client.post_robot_pause()를 threading.Thread로 호출하도록 연결 필요
    def on_pause_robot(self):
        self.log_message(
            "Robot motion paused safe stance. "
            "(TODO: 실제 로봇 PAUSE 엔드포인트 확인 후 연결 필요 - robot_router.py가 현재 빈 라우터)"
        )

    # TODO: routers/robot_router.py에 실제 EMERGENCY STOP 엔드포인트가 생기면
    # client.post_robot_estop()을 threading.Thread로 호출하도록 연결 필요
    def emergency_stop(self):
        self.status_label.config(text="SYSTEM STATUS: E-STOPPED", fg="#e74c3c")
        self.log_message(
            "[CRITICAL ERROR] EMERGENCY STOP ACTIVATED. Hardware Interlock triggered. "
            "(TODO: 실제 EMERGENCY STOP 엔드포인트 확인 후 연결 필요 - robot_router.py가 현재 빈 라우터)"
        )
        messagebox.showerror("긴급 정지 (E-STOP)", "로봇 및 선별 컨베이어 제어 노드가 즉각 차단되었습니다.")

    def trigger_hitl_popup(self, payload=None):
        # 사이드바 "작업자 지시 대기" 버튼으로 수동 테스트할 때는 payload가 없으므로
        # 예전과 동일한 더미 데이터를 사용. WebSocket으로 VLA_ASK_HUMAN이 오면
        # main.py의 decision_planner.DecisionResult를 그대로 담은 실제 payload가 들어옴
        if payload is None:
            payload = {"fruit_type": "apple", "condition": "unknown", "confidence": 0.482}

        fruit_type = payload.get("fruit_type", "unknown")
        condition = payload.get("condition", "unknown")
        confidence = payload.get("confidence") or 0.0
        session_id = payload.get("session_id")  # 현재 backend 브로드캐스트엔 없음(아래 참고), 향후 대비

        # 팝업이 떠 있는 동안 어떤 판정에 대한 응답인지 상태로 들고 있다가
        # manual_sort()에서 post_vla_feedback() 호출 시 사용
        self.hitl_fruit_type = fruit_type
        self.hitl_condition = condition
        self.hitl_session_id = session_id

        self.log_message(
            f"[HITL Alert] {fruit_type} Classification (condition={condition}) "
            f"Confidence ({confidence * 100:.1f}%) below threshold. Waiting human prompt."
        )

        hitl_win = tk.Toplevel(self.root)
        hitl_win.title("Human-In-The-Loop Exception Manager")
        hitl_win.configure(bg="#34495e")
        self.hitl_popup = hitl_win

        def _on_popup_closed():
            if self.hitl_popup is hitl_win:
                self.hitl_popup = None
            hitl_win.destroy()

        hitl_win.protocol("WM_DELETE_WINDOW", _on_popup_closed)

        self.center_popup(hitl_win, width=500, height=350)
        hitl_win.grab_set()

        lbl_alert = tk.Label(hitl_win, text="AI 신뢰도 저하 예외 상황 발생 - 작업자 지시 대기", font=("Helvetica", 11, "bold"), fg="#e67e22", bg="#34495e")
        lbl_alert.pack(pady=12)

        img_dummy = tk.Label(
            hitl_win,
            text=(
                f"[ {fruit_type} 판정 데이터 ]\n\n"
                f"- Condition: {condition}\n"
                f"- Confidence Score: {confidence * 100:.1f}%"
            ),
            font=("Consolas", 10), bg="#000000", fg="#bdc3c7", width=52, height=6,
        )
        img_dummy.pack(pady=8)

        lbl_question = tk.Label(hitl_win, text="해당 예외 농산물을 어떤 분류 범주로 매핑할까요?", font=("Malgun Gothic", 11), fg="#ecf0f1", bg="#34495e")
        lbl_question.pack(pady=8)

        def manual_sort(category):
            destination = CATEGORY_TO_DESTINATION[category]
            raw_answer = _category_to_raw_answer(category)
            self.log_message(f"[HITL] '{category}' 선택 -> raw_answer='{raw_answer}' 전송 중... (destination={destination})")

            def worker():
                try:
                    result = client.post_vla_feedback(
                        fruit_type=self.hitl_fruit_type,
                        condition=self.hitl_condition,
                        raw_answer=raw_answer,
                        session_id=self.hitl_session_id,
                    )
                    # /api/feedback은 "접수됐다"는 것만 알려줌. 실제 해석/학습 결과는
                    # backend가 비동기로 처리한 뒤 WS로 HITL_RESOLVED(성공) 또는
                    # HITL_STUCK(재질문 실패)을 브로드캐스트하므로, 팝업은 여기서 바로
                    # 닫지 않고 그 이벤트가 올 때(_handle_ws_message)까지 열어둠
                    def on_success():
                        self.log_message(f"[HITL Submitted] {result} - 처리 결과 대기 중...")

                    self.root.after(0, on_success)

                except requests.exceptions.RequestException as exc:
                    def on_error():
                        self.log_message(f"[HITL Error] 피드백 전송 실패: {exc} (팝업 유지, 재시도 가능)")

                    self.root.after(0, on_error)

            threading.Thread(target=worker, daemon=True).start()

        btn_box = tk.Frame(hitl_win, bg="#34495e")
        btn_box.pack(fill="x", pady=15, padx=20)

        colors = {"판매": "#2ecc71", "못난이": "#f1c40f", "가공용": "#3498db", "폐기": "#e74c3c"}
        for cat, color in colors.items():
            btn = tk.Button(btn_box, text=cat, bg=color, fg="white" if cat != "못난이" else "black",
                            font=("Helvetica", 10, "bold"), width=9, command=lambda c=cat: manual_sort(c))
            btn.pack(side="left", padx=5, expand=True)

    # [추가] 윈도우 창 닫을 때 카메라 프로세스 릴리즈용 함수
    def on_closing(self):
        self.ws_client.stop()
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VLASorterDashboard(root)
    root.mainloop()