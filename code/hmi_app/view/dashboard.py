import tkinter as tk
from tkinter import messagebox, ttk
import base64
import os
import queue
import sys
import threading
import time
from datetime import datetime
import cv2  # OpenCV 추가
import numpy as np
from PIL import Image, ImageTk  # Pillow 추가
import requests

# dashboard.py는 code/hmi_app/view/ 아래에 있음. hmi_app을 패키지로 import하려면
# 그 부모 디렉토리인 code/ 가 sys.path에 있어야 하므로, cwd가 어디든 상관없이
# python3 dashboard.py로 직접 실행해도 동작하도록 부트스트랩해둠
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from hmi_app.api import client

CATEGORY_TO_DESTINATION = {
    "판매": "normal_box",
    "못난이": "ugly_box",
    "가공용": "processing_box",
    "폐기": "discard_box",
}

def _category_to_raw_answer(category: str) -> str:
    code = ord(category[-1]) - 0xAC00
    has_batchim = 0 <= code < 11172 and code % 28 != 0
    particle = "으로" if has_batchim else "로"
    return f"{category}{particle} 보내"

DESTINATION_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_DESTINATION.items()}

# llm_service.interpret_human_answer()의 few-shot 예시 문구를 그대로 사용 -
# "skip" destination으로 해석되는 걸 사실상 보장하기 위해 임의 문구 대신
# 프롬프트에 이미 등록된 정확한 예시 텍스트를 그대로 보낸다.
HITL_SKIP_RAW_ANSWER = "무시해, 나중에 할게"


class VLASorterDashboard:
    # ── iOS 시스템 컬러 기반 라이트 테마 ──────────────────────────
    COLOR_BG = "#F2F2F7"          # iOS systemGroupedBackground
    COLOR_SIDEBAR = "#F9F9FB"     # 순백 대신 눈부심 줄인 오프화이트
    COLOR_CARD = "#F9F9FB"
    COLOR_CARD_DARK = "#E9E9EB"   # 로그창/입력창 등 인셋 박스 (iOS systemGray5)
    COLOR_ROW = "#F2F2F7"
    COLOR_BORDER = "#D1D1D6"      # iOS separator
    COLOR_TEXT = "#1C1C1E"        # iOS label
    COLOR_TEXT_MUTED = "#6E6E73"  # iOS secondaryLabel
    COLOR_TEXT_DIM = "#2C2C2E"    # 잘 안 보여서 어둡게 조정 (원래 iOS tertiaryLabel은 #AEAEB2였음)
    COLOR_BLUE = "#007AFF"        # iOS systemBlue
    COLOR_BLUE_DEEP = "#0060DF"
    COLOR_GREEN = "#34C759"       # iOS systemGreen
    COLOR_ORANGE = "#FF9500"      # iOS systemOrange
    COLOR_RED = "#FF3B30"         # iOS systemRed
    COLOR_EMERGENCY_BG = "#7A0C0C"  # 진한 알림용 레드 (긴급 상황 강조 유지)

    # E-STOP 오버레이 재개 버튼 라벨. 로봇이 실제로 wait_for_resume()에 들어가기
    # 전(물리 복구 중)엔 WAITING, PROCESS_STATE="waiting_for_resume"을 받은 뒤엔 READY -
    # _show_emergency_overlay/_enable_resume_button 참고.
    BTN_TEXT_RESUME_WAITING = "물리 복구 중 - 잠시만 기다려주세요..."
    BTN_TEXT_RESUME_READY = "현장 확인 완료 - 재개(RESUME)"

    FONT_TITLE = ("NanumGothic", 20, "bold") 
    FONT_SECTION = ("NanumGothic", 13, "bold") 
    FONT_BODY = ("NanumGothic", 11) 
    FONT_BODY_BOLD = ("NanumGothic", 11, "bold") 
    FONT_MONO = ("DejaVu Sans Mono", 11) 
    FONT_MONO_BOLD = ("DejaVu Sans Mono", 12, "bold") 
    FONT_MONO_SMALL = ("DejaVu Sans Mono", 10) 

    DEFAULT_JOINT_ANGLES = {
        "J1 (Base)": 0,
        "J2 (Shoulder)": 0,
        "J3 (Elbow)": 90,
        "J4 (Wrist 1)": 0,
        "J5 (Wrist 2)": 90,
        "J6 (Wrist 3)": 0,
    } 
    JOINT_SPECS = [
        ("J1 (Base)", -180, 180, "정면 기준 좌우 180°"),
        ("J2 (Shoulder)", -360, 360, "전후 무제한 가동"),
        ("J3 (Elbow)", -160, 160, "상하 굴곡 제어"),
        ("J4 (Wrist 1)", -360, 360, "요골/척골 회전 대체"),
        ("J5 (Wrist 2)", -160, 160, "그리퍼 꺾임 각도"),
        ("J6 (Wrist 3)", -360, 360, "그리퍼 무한 회전축"),
    ] 

    def __init__(self, root):
        self.root = root
        self.root.title("RGB-D Multi-View 기반 예외 농산물 판단 및 로봇 분류 시스템")
        self.root.configure(bg=self.COLOR_BG) 

        self.counts = {"판매": 0, "못난이": 0, "가공용": 0, "폐기": 0}
        self.is_estopped = False
        self.is_paused = False
        self.current_frame = None

        # basket_camera/obj_detection 노드가 꺼지거나 죽으면 해당 CAMERA_FRAME이
        # 더 이상 안 오는데, 예전에는 라벨에 마지막으로 그렸던 이미지가 그대로
        # 남아서 마치 계속 살아있는 것처럼 보이는 문제가 있었음. 마지막으로
        # 프레임을 받은 시각을 기록해두고 poll_ws_queue에서 주기적으로 확인해서,
        # 일정 시간 이상 새 프레임이 없으면 대기 화면으로 되돌린다.
        self._basket_frame_last_seen = None
        self._basket_frame_stale = False
        self._main_frame_last_seen = None
        self._main_frame_stale = False

        self.hitl_fruit_type = None 
        self.hitl_condition = None 
        self.hitl_session_id = None 
        self.hitl_popup = None 
        self.hitl_cam_label = None  

        self.joint_sliders = {} 
        self.joint_entries = {} 

        style = ttk.Style()
        style.theme_use('clam')

        self.ws_queue = queue.Queue()
        self.ws_client = client.WebSocketClient(self.ws_queue)
        self.ws_client.start()
        self.root.after(100, self.poll_ws_queue)

        self.create_top_bar()
        self.create_side_bar()
        
        self.content_area = tk.Frame(self.root, bg=self.COLOR_BG) 
        self.content_area.pack(side="right", fill="both", expand=True, padx=15, pady=15) 
        
        self.frames = {} 
        self.create_frames()
        self.show_frame("monitor") 
        
        actual_w, actual_h = self.center_window(1400, 850)
        # 어떤 해상도로 줄여도 사이드바/카드 텍스트가 겹치거나 잘리지 않도록
        # 화면 구성이 무너지지 않는 최소 크기를 강제한다 (그 아래로는 스크롤/축소 대신
        # 창 자체가 더 안 줄어들게 막는 게 산업용 HMI에서 더 안전한 선택).
        #
        # 실측으로 확인된 문제: 이 최소 크기(1180x720)를 고정값으로 두면, 화면
        # 해상도 자체가 그보다 작은 노트북/모니터에서는 창이 화면보다 커지면서
        # 오른쪽/아래쪽이 화면 밖으로 밀려나 버튼이나 글자가 실제로 안 보이는
        # 문제가 있었음(Tk의 minsize는 화면 크기와 무관하게 강제되는 하한이라
        # 화면이 더 작아도 그대로 적용됨). center_window()가 이미 화면 크기에
        # 맞춰 clamp한 실제 창 크기(actual_w/actual_h)를 넘지 않는 값으로
        # minsize를 다시 계산해서, 최소 크기가 화면보다 커지는 일이 없게 한다.
        min_w = min(1180, actual_w)
        min_h = min(720, actual_h)
        self.root.minsize(min_w, min_h)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.log_message("System initialized. UI navigation elements loaded.")

    def center_window(self, width=1400, height=850):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        # 요청한 크기가 실제 화면보다 크면(작은 해상도 모니터/노트북), 화면
        # 크기에 맞춰 줄인다 - 안 그러면 창의 일부(오른쪽/아래쪽 버튼, 글자)가
        # 화면 밖으로 밀려나서 실제로는 안 보이는 문제가 있었음. taskbar 등을
        # 고려해 화면 크기보다 살짝 여유(margin)를 둔다.
        margin_w, margin_h = 40, 80
        width = min(width, max(screen_width - margin_w, 320))
        height = min(height, max(screen_height - margin_h, 240))
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        return width, height

    def center_popup(self, popup, width=750, height=420):
        popup.update_idletasks()
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        # center_window()와 동일한 이유로, 팝업도 화면보다 커지지 않게 clamp함.
        margin_w, margin_h = 40, 80
        width = min(width, max(screen_width - margin_w, 320))
        height = min(height, max(screen_height - margin_h, 240))
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        popup.geometry(f"{width}x{height}+{x}+{y}")
        return width, height

    def _card_frame(self, parent, bg=None):
        return tk.Frame( 
            parent, bg=bg or self.COLOR_CARD, 
            highlightbackground=self.COLOR_BORDER, highlightthickness=1, bd=0, 
        ) 

    def _section_header(self, parent, text, color=None, font=None, bg=None):
        bg = bg or (parent["bg"] if isinstance(parent, (tk.Frame, tk.LabelFrame)) else self.COLOR_CARD) 
        row = tk.Frame(parent, bg=bg) 
        bar = tk.Frame(row, width=5, height=18, bg=color or self.COLOR_BLUE) 
        bar.pack(side="left", padx=(0, 8), pady=2) 
        lbl = tk.Label(row, text=text, font=font or self.FONT_SECTION, fg=self.COLOR_TEXT, bg=bg) 
        lbl.pack(side="left") 
        return row 

    def _flat_button(self, parent, text, bg, active_bg, fg="white", font=None):
        return tk.Button( 
            parent, text=text, bg=bg, fg=fg, activebackground=active_bg, activeforeground=fg, 
            font=font or self.FONT_BODY_BOLD, bd=0, relief="flat", pady=10, cursor="hand2", 
        ) 

    # 카메라 노드가 죽거나 꺼진 뒤 이만큼(초) 새 프레임이 없으면 마지막 프레임을
    # 화면에 그대로 두지 않고 대기 화면으로 되돌린다.
    CAMERA_FRAME_STALE_TIMEOUT_SEC = 2.0

    def poll_ws_queue(self):
        try:
            while True:
                msg = self.ws_queue.get_nowait()
                self._handle_ws_message(msg)
        except queue.Empty:
            pass
        self._check_basket_frame_stale()
        self._check_main_frame_stale()
        self.root.after(100, self.poll_ws_queue)

    def _check_basket_frame_stale(self):
        if self._basket_frame_stale or self._basket_frame_last_seen is None:
            return
        if time.monotonic() - self._basket_frame_last_seen < self.CAMERA_FRAME_STALE_TIMEOUT_SEC:
            return
        self._basket_frame_stale = True
        if self.lbl_c720 is not None and self.lbl_c720.winfo_exists():
            self.lbl_c720.config(
                image="",
                text="⚠ Basket Camera 연결 끊김\nbasket_camera 노드를 실행해 연결하세요.",
            )
            self.lbl_c720.imgtk = None

    def _check_main_frame_stale(self):
        if self._main_frame_stale or self._main_frame_last_seen is None:
            return
        if time.monotonic() - self._main_frame_last_seen < self.CAMERA_FRAME_STALE_TIMEOUT_SEC:
            return
        self._main_frame_stale = True
        if self.lbl_webcam is not None and self.lbl_webcam.winfo_exists():
            self.lbl_webcam.config(
                image="",
                text="⚠ Pick Camera 연결 끊김\nobj_detection(RealSense) 노드를 실행해 연결하세요.",
                font=self.FONT_MONO_SMALL, fg=self.COLOR_TEXT_MUTED, justify="left",
            )
            self.lbl_webcam.imgtk = None

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
            self.trigger_hitl_popup(payload)

        elif msg_type == "HITL_RESOLVED":
            payload = msg.get("payload", {}) or {}
            fruit_type = payload.get('fruit_type')
            destination = payload.get('destination')
            self.log_message(
                f"[HITL_RESOLVED] 해결됨: {fruit_type} → {destination} "
                f"(condition={payload.get('condition')}, confidence={payload.get('confidence')})"
            )

            # VLA_DECISION(자동 판정)과 동일하게, 사람이 확정한 매핑도 카운트에 반영
            category = DESTINATION_TO_CATEGORY.get(destination)
            if category is not None:
                self.counts[category] = self.counts.get(category, 0) + 1
                if category in self.count_labels:
                    self.count_labels[category].config(text=f"{self.counts[category]} 개")

            # 매핑 완료를 사람이 직접 확인하도록 안내 창을 띄우고, "확인"을 눌러야
            # 팝업이 닫히게 함 (messagebox.showinfo는 사용자가 닫을 때까지 블로킹됨).
            #
            # parent를 안 주면 이 안내 창이 self.root를 기준으로 뜨는데, HITL
            # 팝업(hitl_win)이 grab_set()으로 입력을 독점하고 있는 동안이라 안내
            # 창이 그 뒤에 가려서 안 보이는 문제가 있었음(실제로 겪은 문제 - 매핑
            # 완료를 눌러도 아무 반응이 없는 것처럼 보였음). HITL 팝업을 parent로
            # 지정하면 그 위(transient)로 확실히 뜬다.
            hitl_popup_still_open = (
                self.hitl_popup is not None and self.hitl_popup.winfo_exists()
            )
            messagebox.showinfo(
                "매핑 완료",
                f"'{fruit_type}'을(를) [{category or destination}]로 매핑 완료했습니다.",
                parent=self.hitl_popup if hitl_popup_still_open else self.root,
            )

            if self.hitl_popup is not None:
                try:
                    if self.hitl_popup.winfo_exists():
                        self.hitl_popup.destroy()
                except tk.TclError:
                    pass
                self.hitl_popup = None
                self.hitl_cam_label = None

        elif msg_type == "HITL_SKIPPED":
            payload = msg.get("payload", {}) or {}
            fruit_type = payload.get('fruit_type')
            self.log_message(
                f"[HITL_SKIPPED] 보류: {fruit_type} (condition={payload.get('condition')}) - "
                "정책 학습 없이 다음 사과부터 처리합니다."
            )
            # HITL_RESOLVED와 달리 아무것도 확정된 게 없으므로(카운트 미반영,
            # 정책 미학습) 확인 팝업 없이 조용히 닫기만 한다.
            if self.hitl_popup is not None:
                try:
                    if self.hitl_popup.winfo_exists():
                        self.hitl_popup.destroy()
                except tk.TclError:
                    pass
                self.hitl_popup = None
                self.hitl_cam_label = None

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

        elif msg_type == "PROCESS_STATE":
            # StatusBus.set_state()가 로봇 쪽에서 발행 -> robot_bridge.py가 그대로
            # 중계. 특히 check_and_recover()가 비상정지 복구 완료 시 "MOVING"을
            # 다시 보내주므로, 로봇이 이미 자동 재개된 걸 여기서 실시간으로 볼 수 있음
            # (오버레이 화면을 직접 닫아주지는 않음 - 그건 사람이 버튼으로 확인).
            status_text = msg.get("payload", "")

            # 로봇단(estop_handler.check_and_recover)이 자체적으로 정지했을 때
            # 보내는 상태 문자열들 - "emergency_stop"(정지 직후), "waiting_for_resume"
            # (물리 복구까지 끝내고 재개 신호 대기 중), "controller_safety_stop:..."
            # (펜던트 물리 비상정지로 하드웨어 레벨 정지) 세 가지 모두 프론트 E-STOP
            # 버튼을 누르지 않고도 발생할 수 있음 - 이 경우 오버레이가 안 뜨면
            # 재개(RESUME) 버튼 자체가 화면 어디에도 없어서 운영자가 재개시킬 방법이
            # 없었음(실제로 겪은 문제). is_estopped가 아직 아니면 여기서 오버레이를 띄움.
            if status_text in ("emergency_stop", "waiting_for_resume") or status_text.startswith(
                "controller_safety_stop"
            ):
                self._handle_robot_triggered_estop(status_text)

            if status_text == "waiting_for_resume":
                # 로봇 쪽이 실제로 wait_for_resume()에 들어가기 직전에만 발행하는
                # 상태 - 이 시점부터 눌러야 재개 신호가 로봇에 살아서 전달되므로,
                # 그 전까지 비활성화해뒀던 오버레이 버튼을 여기서 활성화한다
                # (_show_emergency_overlay/_enable_resume_button 참고).
                self._enable_resume_button()

            if self.is_estopped:
                # E-STOP 오버레이가 떠 있는 동안은(위에서 방금 띄웠든, 이전부터
                # 떠 있었든) 상단 상태 라벨도 항상 위험 색으로 고정 - "waiting_for_resume"
                # 처럼 위 키워드 목록에 안 걸리는 상태 문자열 때문에 오버레이는 빨간데
                # 상태 라벨만 회색으로 보이는 불일치를 막기 위함.
                color = self.COLOR_RED
            elif any(k in status_text for k in ["오류", "실패", "emergency", "ERROR"]):
                color = self.COLOR_RED
            elif any(k in status_text for k in ["완료", "RUNNING", "이동", "구동", "MOVING"]):
                color = self.COLOR_GREEN
            else:
                color = self.COLOR_TEXT_MUTED
            self.status_label.config(text=f"SYSTEM STATUS: {status_text}", fg=color)
            self.log_message(f"[PROCESS_STATE] {status_text}")

        elif msg_type == "MOTION_STATUS":
            motion = msg.get("motion", "")
            message = msg.get("message", "")
            self.log_message(f"[MOTION_STATUS] {motion}" + (f" - {message}" if message else ""))

        elif msg_type == "SAFETY_EVENT":
            error_code = msg.get("error_code", "")
            error_msg = msg.get("error_msg", "")
            self.log_message(f"[SAFETY_EVENT] {error_code}: {error_msg}")

        elif msg_type == "CAMERA_FRAME":
            try: 
                jpg_bytes = base64.b64decode(msg.get("image", "")) 
                jpg_array = np.frombuffer(jpg_bytes, dtype=np.uint8) 
                frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR) 
                if frame is not None:
                    self.current_frame = frame
                    self._main_frame_last_seen = time.monotonic()
                    self._main_frame_stale = False
                    self.render_camera_frame(frame, self.lbl_webcam, self.cam_box_w, self.cam_box_h)
                    
                    if self.hitl_popup and self.hitl_cam_label:
                        self.render_camera_frame(frame, self.hitl_cam_label, 360, 240)
            except Exception as e:
                self.log_message(f"[CAMERA_FRAME] 디코딩 실패: {e}")

        elif msg_type == "BASKET_CAMERA_FRAME":
            # basket_bridge.py가 /basket_camera/debug_image(바스켓 b1~b4 bbox +
            # 사과 유무 오버레이)를 중계하는 메시지. Cam 1(CAMERA_FRAME)과 완전히
            # 별개의 스트림이라 Cam 2 라벨(self.lbl_c720)에만 그린다.
            try:
                jpg_bytes = base64.b64decode(msg.get("image", ""))
                jpg_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    self._basket_frame_last_seen = time.monotonic()
                    self._basket_frame_stale = False
                    self.render_camera_frame(
                        frame, self.lbl_c720, self.basket_cam_box_w, self.basket_cam_box_h
                    )
            except Exception as e:
                self.log_message(f"[BASKET_CAMERA_FRAME] 디코딩 실패: {e}")

    def create_top_bar(self):
        self.top_bar = tk.Frame(self.root, bg=self.COLOR_SIDEBAR, height=60) 
        self.top_bar.pack(fill="x", side="top") 
        self.top_bar.pack_propagate(False) 

        self.title_label = tk.Label(self.top_bar, text="VLA SORTING SYSTEM HMI", 
                               font=self.FONT_MONO_BOLD, fg=self.COLOR_BLUE, bg=self.COLOR_SIDEBAR) 
        self.title_label.pack(side="left", padx=20, pady=15) 

        self.current_tab_lbl = tk.Label(self.top_bar, text="- 메인 모니터링", font=self.FONT_BODY, fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_SIDEBAR) 
        self.current_tab_lbl.pack(side="left", pady=17) 

        self.status_label = tk.Label(self.top_bar, text="SYSTEM STATUS: RUNNING", 
                                     font=self.FONT_MONO_BOLD, fg=self.COLOR_GREEN, bg=self.COLOR_SIDEBAR) 
        self.status_label.pack(side="right", padx=20, pady=18) 

    def create_side_bar(self):
        self.side_bar = tk.Frame(self.root, bg=self.COLOR_SIDEBAR, width=210) 
        self.side_bar.pack(side="left", fill="y") 
        self.side_bar.pack_propagate(False) 

        logo_lbl = tk.Label(self.side_bar, text="[SYS] CONTROL PANEL", font=self.FONT_MONO_BOLD, fg=self.COLOR_BLUE, bg=self.COLOR_SIDEBAR, pady=25) 
        logo_lbl.pack(fill="x", side="top") 

        menu_defs = [
            ("실시간 모니터링", "monitor", self.COLOR_BLUE), 
            ("VLA 정책 관리", "policy", self.COLOR_ORANGE), 
            ("시스템 및 로봇 제어", "control", self.COLOR_GREEN), 
            ("시스템 로그", "log", self.COLOR_TEXT_MUTED), 
        ]

        def menu_btn(text, target, accent_color):
            row = tk.Frame(self.side_bar, bg=self.COLOR_SIDEBAR) 
            btn = tk.Button( 
                row, text=f"   {text}", font=self.FONT_BODY_BOLD, bg=self.COLOR_SIDEBAR, fg=self.COLOR_TEXT_MUTED, 
                activebackground=self.COLOR_CARD, activeforeground=self.COLOR_TEXT, bd=0, relief="flat", 
                height=2, anchor="w", padx=20, cursor="hand2", 
                command=lambda: self.show_frame(target), 
            ) 
            btn.pack(fill="x") 
            dot = tk.Frame(btn, width=4, height=16, bg=accent_color) 
            dot.place(x=15, rely=0.5, anchor="w") 
            return row 

        for text, target, accent_color in menu_defs:
            menu_btn(text, target, accent_color).pack(fill="x", pady=2, padx=8) 

        lbl_div = tk.Frame(self.side_bar, bg=self.COLOR_BORDER, height=1)
        lbl_div.pack(fill="x", pady=20, padx=15)

    def create_frames(self):
        f_monitor = tk.Frame(self.content_area, bg=self.COLOR_BG) 
        self.setup_monitor_tab(f_monitor) 
        self.frames["monitor"] = (f_monitor, "- 메인 모니터링") 

        f_policy = tk.Frame(self.content_area, bg=self.COLOR_BG) 
        self.setup_policy_tab(f_policy) 
        self.frames["policy"] = (f_policy, "- VLA 자연어 정책 관리") 

        f_control = tk.Frame(self.content_area, bg=self.COLOR_BG) 
        self.setup_combined_control_tab(f_control)
        self.frames["control"] = (f_control, "- 시스템 운영 및 하드웨어 제어")

        f_log = tk.Frame(self.content_area, bg=self.COLOR_BG) 
        self.setup_log_tab(f_log) 
        self.frames["log"] = (f_log, "- 시스템 로그") 

    def show_frame(self, target):
        if self.is_estopped:
            return
        for f, _ in self.frames.values():
            f.pack_forget() 
        frame, title = self.frames[target] 
        frame.pack(fill="both", expand=True) 
        self.current_tab_lbl.config(text=title) 

    def setup_monitor_tab(self, parent):
        cam_frame = tk.Frame(parent, bg=self.COLOR_BG) 
        cam_frame.pack(fill="both", expand=True, pady=(0, 10)) 

        self.frame_d455 = self._card_frame(cam_frame)
        self.frame_d455.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self._section_header(self.frame_d455, "Cam 1: D455 + YOLO Detection (Real-time View)", self.COLOR_BLUE).pack(anchor="w", padx=15, pady=(12, 8))

        webcam_container = tk.Frame(self.frame_d455, bg="#000000")
        webcam_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        webcam_container.pack_propagate(False)
        self.webcam_container = webcam_container

        self.cam_box_w = 480
        self.cam_box_h = 360
        webcam_container.bind("<Configure>", self.on_webcam_container_resize)

        self.lbl_webcam = tk.Label(webcam_container, bg="#000000")
        self.lbl_webcam.pack(fill="both", expand=True)

        frame_c720 = self._card_frame(cam_frame)
        frame_c720.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self._section_header(frame_c720, "Cam 2: Basket Camera (B1~B4 Occupancy)", self.COLOR_ORANGE).pack(anchor="w", padx=15, pady=(12, 8))

        basket_cam_container = tk.Frame(frame_c720, bg="#000000")
        basket_cam_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        basket_cam_container.pack_propagate(False)
        self.basket_cam_container = basket_cam_container

        self.basket_cam_box_w = 480
        self.basket_cam_box_h = 360
        basket_cam_container.bind("<Configure>", self.on_basket_cam_container_resize)

        self.lbl_c720 = tk.Label(
            basket_cam_container, text="Basket Camera 연결 대기 중...\n- basket_camera 노드가 켜지면 자동으로 표시됩니다.",
            font=self.FONT_MONO_SMALL, fg=self.COLOR_TEXT_MUTED, bg="#000000", justify="left",
        )
        self.lbl_c720.pack(fill="both", expand=True)

        self.create_counter_table(parent)

    def on_webcam_container_resize(self, event):
        self.cam_box_w = max(event.width, 10)
        self.cam_box_h = max(event.height, 10)

    def on_basket_cam_container_resize(self, event):
        self.basket_cam_box_w = max(event.width, 10)
        self.basket_cam_box_h = max(event.height, 10)

    def render_camera_frame(self, frame, label_widget, box_w, box_h):
        if label_widget is None or not label_widget.winfo_exists(): 
            return

        frame_h, frame_w = frame.shape[:2] 
        scale = min(box_w / frame_w, box_h / frame_h) 
        new_w, new_h = int(frame_w * scale), int(frame_h * scale) 
        
        if new_w <= 0 or new_h <= 0:
            return

        resized = cv2.resize(frame, (new_w, new_h)) 
        canvas = cv2.copyMakeBorder( 
            resized, 
            top=(box_h - new_h) // 2, 
            bottom=box_h - new_h - (box_h - new_h) // 2, 
            left=(box_w - new_w) // 2, 
            right=box_w - new_w - (box_w - new_w) // 2, 
            borderType=cv2.BORDER_CONSTANT, 
            value=(0, 0, 0) 
        ) 

        cv2image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB) 
        img = Image.fromarray(canvas) if cv2image is None else Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img) 

        label_widget.imgtk = imgtk 
        label_widget.config(image=imgtk) 

    def setup_policy_tab(self, parent):
        frame = self._card_frame(parent) 
        frame.pack(fill="both", expand=True, padx=10, pady=10) 

        self._section_header(frame, "VLA NATURAL LANGUAGE POLICY INPUT", self.COLOR_ORANGE).pack(anchor="w", padx=20, pady=(18, 10)) 

        lbl_guide = tk.Label(frame, text="작업 현장의 가변적인 처리 기준을 자연어로 입력하세요:", font=self.FONT_BODY, fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_CARD) 
        lbl_guide.pack(anchor="w", padx=20, pady=(0, 10)) 

        self.policy_text = tk.Text( 
            frame, height=5, font=self.FONT_BODY, bg=self.COLOR_CARD_DARK, fg=self.COLOR_TEXT, 
            insertbackground=self.COLOR_TEXT, relief="flat", highlightbackground=self.COLOR_BORDER, highlightthickness=1, 
        ) 
        self.policy_text.insert("1.0", "오늘은 작은 사과도 판매하고, 곰팡이 의심되는 것만 무조건 폐기해줘. 애매하면 물어봐.") 
        self.policy_text.pack(fill="x", padx=20, pady=5) 

        self.btn_convert = self._flat_button(frame, "자연어 정책 적용", self.COLOR_BLUE_DEEP, self.COLOR_BLUE_DEEP)
        self.btn_convert.configure(command=self.apply_policy)
        self.btn_convert.pack(anchor="e", padx=20, pady=10)

        self._section_header(frame, "로봇 TTS 질의 로그 (미확인 물체 발생 시 작업자에게 음성으로 질의한 내용)", self.COLOR_GREEN).pack(anchor="w", padx=20, pady=(10, 5))

        self.tts_question_log = tk.Text(
            frame, height=10, font=self.FONT_MONO, fg=self.COLOR_GREEN, bg=self.COLOR_CARD_DARK,
            relief="flat", highlightbackground=self.COLOR_BORDER, highlightthickness=1, state="disabled",
        )
        self.tts_question_log.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def setup_combined_control_tab(self, parent):
        container = tk.Frame(parent, bg=self.COLOR_BG)
        container.pack(fill="both", expand=True, padx=5, pady=5)

        ctrl_frame = self._card_frame(container) 
        ctrl_frame.pack(fill="x", padx=5, pady=(0, 10)) 
        self._section_header(ctrl_frame, "SYSTEM HARDWARE CORE CONTROL (시스템 기본 제어)", self.COLOR_BLUE).pack(anchor="w", padx=20, pady=(15, 8)) 

        btn_row = tk.Frame(ctrl_frame, bg=self.COLOR_CARD) 
        btn_row.pack(fill="x", padx=20, pady=(0, 20)) 

        self.btn_main_start = self._flat_button(btn_row, "START ROBOT SYSTEM", self.COLOR_GREEN, "#248A3D", fg="white")
        self.btn_main_start.configure(width=22, command=self.on_start_robot)
        self.btn_main_start.pack(side="left", padx=(0, 12), expand=True, fill="x")

        self.btn_main_pause = self._flat_button(btn_row, "PAUSE ROBOT MOTION", self.COLOR_ORANGE, "#C76E00", fg="white")
        self.btn_main_pause.configure(width=22, command=self.on_pause_robot)
        self.btn_main_pause.pack(side="left", padx=(0, 12), expand=True, fill="x")

        self.btn_main_estop = self._flat_button(btn_row, "EMERGENCY STOP", self.COLOR_RED, "#C0281F", fg="white")
        self.btn_main_estop.configure(width=22, command=self.emergency_stop)
        self.btn_main_estop.pack(side="left", expand=True, fill="x")

        self.create_joint_override_panel(container) 
        self.create_end_effector_panel(container) 

    def setup_log_tab(self, parent):
        log_frame = self._card_frame(parent) 
        log_frame.pack(fill="both", expand=True, padx=10, pady=10) 
        self._section_header(log_frame, "실시간 데이터 및 이벤트 로그 (ROS2 로그 매핑 가능)", self.COLOR_TEXT_MUTED).pack(anchor="w", padx=20, pady=(15, 8)) 

        self.log_box = tk.Text( 
            log_frame, font=self.FONT_MONO_SMALL, fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_CARD_DARK, 
            relief="flat", highlightbackground=self.COLOR_BORDER, highlightthickness=1, 
        ) 
        self.log_box.pack(fill="both", expand=True, padx=20, pady=(0, 20)) 
        self.log_box.tag_configure("log_ok", foreground=self.COLOR_GREEN) 
        self.log_box.tag_configure("log_info", foreground=self.COLOR_BLUE) 
        self.log_box.tag_configure("log_err", foreground=self.COLOR_RED) 
        self.log_box.tag_configure("log_default", foreground=self.COLOR_TEXT_MUTED) 

    # ── [핵심 추가]: 슬라이더 클릭 시 버튼을 누르고 있지 않아도 해당 지점으로 바로 점프하는 헬퍼 함수 ────
    def _jump_to_value(self, event, slider_widget):
        try:
            min_val = float(slider_widget.cget("from"))
            max_val = float(slider_widget.cget("to"))
            x0, _ = slider_widget.coords(min_val)
            x1, _ = slider_widget.coords(max_val)
            if x1 != x0:
                fraction = (event.x - x0) / (x1 - x0)
                fraction = max(0.0, min(1.0, fraction))
                slider_widget.set(min_val + fraction * (max_val - min_val))
        except Exception:
            pass
        # 기본 클래스 바인딩(버튼을 누른 채 드래그해야 이동하는 동작) 전파를 막아
        # 클릭만으로도 즉시 값이 반영되도록 함
        return "break"

    def create_joint_override_panel(self, parent):
        joint_frame = self._card_frame(parent) 
        joint_frame.pack(fill="x", padx=5, pady=(0, 10)) 
        self._section_header(joint_frame, "JOINT OVERRIDE (협동로봇 관절별 수동 제어 조작)", self.COLOR_BLUE).pack(anchor="w", padx=20, pady=(15, 10)) 

        self.joint_sliders.clear() 
        self.joint_entries.clear() 

        rows_wrap = tk.Frame(joint_frame, bg=self.COLOR_CARD) 
        rows_wrap.pack(fill="x", padx=15, pady=(0, 15)) 

        for name, min_val, max_val, desc in self.JOINT_SPECS: 
            j_row = tk.Frame(rows_wrap, bg=self.COLOR_ROW, highlightbackground=self.COLOR_BORDER, highlightthickness=1) 
            j_row.pack(fill="x", padx=5, pady=4) 

            lbl_name = tk.Label(j_row, text=name, font=self.FONT_MONO_BOLD, fg=self.COLOR_BLUE, bg=self.COLOR_ROW, width=13, anchor="w") 
            lbl_name.pack(side="left", padx=(10, 5), pady=8)

            lbl_desc = tk.Label(j_row, text=f"({desc})", font=("NanumGothic", 9), fg=self.COLOR_TEXT_DIM, bg=self.COLOR_ROW, width=20, anchor="w") 
            lbl_desc.pack(side="left", padx=5) 

            initial_value = self.DEFAULT_JOINT_ANGLES.get(name, 0) 

            entry = tk.Entry( 
                j_row, width=7, font=self.FONT_MONO_SMALL, justify="center", bg=self.COLOR_CARD_DARK, fg=self.COLOR_TEXT, 
                insertbackground=self.COLOR_TEXT, relief="flat", highlightbackground=self.COLOR_BORDER, highlightthickness=1, 
            ) 
            entry.pack(side="right", padx=(5, 15)) 
            entry.insert(0, f"{initial_value:.1f}") 
            self.joint_entries[name] = entry 

            slider = tk.Scale(
                j_row, from_=min_val, to=max_val, orient="horizontal",
                bg=self.COLOR_BLUE_DEEP, fg=self.COLOR_TEXT_MUTED, troughcolor=self.COLOR_CARD_DARK, highlightthickness=0,
                activebackground=self.COLOR_BLUE_DEEP, showvalue=False, bd=0,
                command=lambda val, e=entry: self._sync_slider_to_entry(val, e), 
            ) 
            slider.set(initial_value) 
            slider.pack(side="left", fill="x", expand=True, padx=15) 
            
            # [기능 개선]: 슬라이더의 선 영역 클릭 시 버튼 조작 필요 없이 좌표 기준 즉시 이동 바인딩
            slider.bind("<Button-1>", lambda event, s=slider: self._jump_to_value(event, s))
            
            self.joint_sliders[name] = slider 

            entry.bind( 
                "<Return>", 
                lambda event, s=slider, e=entry, mn=min_val, mx=max_val: self._sync_entry_to_slider(s, e, mn, mx), 
            ) 

    def create_end_effector_panel(self, parent):
        tool_frame = self._card_frame(parent) 
        tool_frame.pack(fill="x", padx=5, pady=(0, 5)) 
        self._section_header(tool_frame, "END EFFECTOR & ACTUATOR CONTROL (그리퍼 및 액추에이터 제어)", self.COLOR_GREEN).pack(anchor="w", padx=20, pady=(15, 8)) 

        btn_zone = tk.Frame(tool_frame, bg=self.COLOR_CARD) 
        btn_zone.pack(fill="x", padx=15, pady=(0, 15)) 

        # 하단 조작계 버튼 색상 상시 고정 유지
        btn_open = self._flat_button(btn_zone, "그리퍼 개방 (OPEN)", "#007AFF", "#0051D0")
        btn_open.configure(command=lambda: self.control_hardware_action("그리퍼 OPEN", "OPEN"))
        btn_open.pack(side="left", expand=True, fill="x", padx=(0, 6))

        btn_close = self._flat_button(btn_zone, "그리퍼 파지 (CLOSE)", "#0051D0", "#00308F")
        btn_close.configure(command=lambda: self.control_hardware_action("그리퍼 CLOSE", "CLOSE"))
        btn_close.pack(side="left", expand=True, fill="x", padx=(0, 6))

        btn_movej = self._flat_button(btn_zone, "관절각 일괄 전송 (MoveJ)", "#30B0C7", "#2694A8")
        btn_movej.configure(command=self.send_all_joints_command)
        btn_movej.pack(side="left", expand=True, fill="x", padx=(0, 6))

        btn_home = self._flat_button(btn_zone, "로봇 원상복구 (Go Home)", self.COLOR_GREEN, "#248A3D", fg="white")
        btn_home.configure(command=self.reset_all_joints)
        btn_home.pack(side="left", expand=True, fill="x")

    def _sync_slider_to_entry(self, value, entry_widget):
        if entry_widget.winfo_exists(): 
            entry_widget.delete(0, "end") 
            entry_widget.insert(0, f"{float(value):.1f}") 

    def _sync_entry_to_slider(self, slider_widget, entry_widget, min_val, max_val):
        try: 
            val = float(entry_widget.get()) 
            val = max(min_val, min(val, max_val)) 
            if entry_widget.winfo_exists(): 
                entry_widget.delete(0, "end") 
                entry_widget.insert(0, f"{val:.1f}") 
            if slider_widget.winfo_exists(): 
                slider_widget.set(val) 
        except ValueError: 
            if slider_widget.winfo_exists() and entry_widget.winfo_exists(): 
                entry_widget.delete(0, "end") 
                entry_widget.insert(0, f"{slider_widget.get():.1f}") 

    def control_hardware_action(self, action_name, action):
        self.log_message(f"[Gripper] {action_name} 명령 전송 중...")

        def worker():
            try:
                result = client.post_gripper_command(action)

                def on_success():
                    self.log_message(f"[Gripper] {action_name} 명령 전송 성공: {result}")
                    messagebox.showinfo("명령 전송 성공", f"{action_name} 명령이 정상적으로 전송되었습니다.")

                self.root.after(0, on_success)

            except requests.exceptions.RequestException as exc:
                error_text = str(exc)

                def on_error():
                    self.log_message(f"[Gripper] {action_name} 명령 전송 실패: {error_text}")
                    messagebox.showerror("서버 연결 실패", f"서버와 연결되지 않았습니다.")

                self.root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    def send_all_joints_command(self):
        joint_angles = {name.split(" ")[0]: round(slider.get(), 1) for name, slider in self.joint_sliders.items()}
        self.log_message(f"[MoveJ] 관절 일괄 이동 명령 전송 중: {joint_angles}")

        def worker():
            try:
                result = client.post_move_joint(joint_angles)

                def on_success():
                    self.log_message(f"[MoveJ] 관절 일괄 이동 명령 전송 성공: {result}")
                    messagebox.showinfo("명령 전송 성공", "관절각 일괄 전송 명령이 정상적으로 전송되었습니다.")

                self.root.after(0, on_success)

            except requests.exceptions.RequestException as exc:
                error_text = str(exc)

                def on_error():
                    self.log_message(f"[MoveJ] 관절 일괄 이동 명령 전송 실패: {error_text}")
                    messagebox.showerror("서버 연결 실패", f"서버와 연결되지 않았습니다.")

                self.root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    def reset_all_joints(self):
        for name, slider in self.joint_sliders.items(): 
            if slider.winfo_exists(): 
                slider.set(self.DEFAULT_JOINT_ANGLES.get(name, 0.0)) 
        for name, entry in self.joint_entries.items(): 
            if entry.winfo_exists(): 
                value = self.DEFAULT_JOINT_ANGLES.get(name, 0.0) 
                entry.delete(0, "end") 
                entry.insert(0, f"{value:.1f}") 
        self.log_message("[Go Home] 로봇 원점 복귀 명령 완료") 

    def create_counter_table(self, parent):
        frame = self._card_frame(parent) 
        frame.pack(fill="x", side="bottom", pady=(10, 0)) 
        self._section_header(frame, "SORTING COUNT (실시간 분류 통계)", self.COLOR_GREEN).pack(anchor="w", padx=20, pady=(12, 8)) 

        row_wrap = tk.Frame(frame, bg=self.COLOR_CARD) 
        row_wrap.pack(fill="x", padx=20, pady=(0, 15)) 

        self.count_labels = {} 
        for i, (category, count) in enumerate(self.counts.items()): 
            row_wrap.grid_columnconfigure(i, weight=1) 

            cell = tk.Frame(row_wrap, bg=self.COLOR_ROW, highlightbackground=self.COLOR_BORDER, highlightthickness=1) 
            cell.grid(row=0, column=i, padx=6, sticky="nsew") 

            lbl_title = tk.Label(cell, text=category, font=self.FONT_BODY_BOLD, fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_ROW) 
            lbl_title.pack(pady=(10, 4)) 

            lbl_cnt = tk.Label(cell, text=f"{count} 개", font=self.FONT_MONO_BOLD, fg=self.COLOR_GREEN, bg=self.COLOR_ROW) 
            lbl_cnt.pack(pady=(0, 10)) 
            self.count_labels[category] = lbl_cnt 

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S") 
        if hasattr(self, 'log_box') and self.log_box.winfo_exists(): 
            if any(k in message for k in ["Error", "오류", "실패", "STOPPED", "CRITICAL"]): 
                tag = "log_err" 
            elif any(k in message for k in ["완료", "성공", "RUNNING", "OK", "복구"]): 
                tag = "log_ok" 
            elif any(k in message for k in ["전송", "진행", "동작", "명령"]): 
                tag = "log_info" 
            else:
                tag = "log_default" 
            self.log_box.config(state="normal")
            self.log_box.insert("end", f"[{timestamp}] {message}\n", tag) 
            self.log_box.see("end") 
            self.log_box.config(state="disabled")

    def _build_ask_human_tts_text(self, fruit_type, condition, confidence):
        return (
            f"작업자님, 확인이 필요합니다. 감지된 물체는 '{fruit_type}'이며 상태는 '{condition}'으로 보이는데, "
            f"판단 신뢰도가 {confidence * 100:.1f}%로 낮습니다. 어떤 물체인지, 어떻게 분류할지 알려주세요."
        )

    def append_tts_question(self, text):
        if not hasattr(self, "tts_question_log") or not self.tts_question_log.winfo_exists():
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.tts_question_log.config(state="normal")
        self.tts_question_log.insert("end", f"[{timestamp}] [ROBOT TTS] {text}\n\n")
        self.tts_question_log.see("end")
        self.tts_question_log.config(state="disabled")

    def apply_policy(self):
        text = self.policy_text.get("1.0", "end-1c").strip()
        if not text:
            return

        self.log_message(f"[VLA NLP Update] 정책 명령 전송 중: '{text}'")
        self.btn_convert.config(state="disabled")

        def worker():
            try:
                result = client.post_policy_command(text)

                def on_success():
                    self.log_message(f"[VLA NLP Update] 정책 반영 완료: {result.get('applied_policies')}")
                    self.btn_convert.config(state="normal")

                self.root.after(0, on_success)

            except requests.exceptions.RequestException as exc:
                error_text = str(exc)

                def on_error():
                    self.log_message(f"[VLA NLP Error] 서버 연결 실패: {error_text}")
                    self.btn_convert.config(state="normal")

                self.root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    def on_start_robot(self):
        if self.is_estopped:
            messagebox.showerror("명령 거부됨", "긴급 정지(E-STOP) 상황에서는 시스템을 가동할 수 없습니다. 안전 확인 후 복구 리셋을 먼저 수행하십시오.")
            return

        self.log_message("[Robot] START 명령 전송 중...")

        def worker():
            try:
                result = client.post_robot_start()

                def on_success():
                    self.status_label.config(text="SYSTEM STATUS: RUNNING", fg=self.COLOR_GREEN)
                    self.log_message(f"[Robot] START 명령 전송 성공: {result}")

                self.root.after(0, on_success)

            except requests.exceptions.RequestException as exc:
                error_text = str(exc)

                def on_error():
                    self.log_message(f"[Robot] START 명령 전송 실패: {error_text}")
                    messagebox.showerror("서버 연결 실패", f"서버와 연결되지 않았습니다.")

                self.root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    def on_pause_robot(self):
        # 토글 버튼: 처음 누르면 일시정지(MANUAL_PAUSE), 이미 일시정지 중이면
        # 같은 버튼을 다시 눌러 재개(RESUME)함 - box_sequence_test.py의
        # robot_command_callback이 RESUME을 받으면 pause_requested도 함께
        # clear()하도록 되어 있어서(같은 명령을 재사용), 별도 엔드포인트 없이
        # 이 버튼 하나로 일시정지/재개를 오갈 수 있음.
        if self.is_estopped:
            return

        if self.is_paused:
            self._send_pause_resume(resume=True)
        else:
            self._send_pause_resume(resume=False)

    def _send_pause_resume(self, resume: bool):
        action_label = "재개(RESUME)" if resume else "일시정지(PAUSE)"
        self.log_message(f"[Robot] {action_label} 명령 전송 중...")
        self.btn_main_pause.config(state="disabled")

        def worker():
            try:
                result = client.post_robot_resume() if resume else client.post_robot_pause()

                def on_success():
                    self.is_paused = not resume
                    if self.is_paused:
                        self.status_label.config(text="SYSTEM STATUS: PAUSED", fg=self.COLOR_ORANGE)
                        self.btn_main_pause.config(text="RESUME ROBOT MOTION")
                    else:
                        self.status_label.config(text="SYSTEM STATUS: RUNNING", fg=self.COLOR_GREEN)
                        self.btn_main_pause.config(text="PAUSE ROBOT MOTION")
                    self.btn_main_pause.config(state="normal")
                    self.log_message(f"[Robot] {action_label} 명령 전송 성공: {result}")

                self.root.after(0, on_success)

            except requests.exceptions.RequestException as exc:
                error_text = str(exc)

                def on_error():
                    self.btn_main_pause.config(state="normal")
                    self.log_message(f"[Robot] {action_label} 명령 전송 실패: {error_text}")
                    messagebox.showerror("서버 연결 실패", f"서버와 연결되지 않았습니다.")

                self.root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    def _enter_estop_ui(self, log_text):
        """
        E-STOP 화면 잠금(오버레이 표시 + 상단바/상태 라벨 빨간색 전환)만 담당.
        emergency_stop()(운영자가 직접 프론트 버튼을 눌러 정지시키는 경로)과
        _handle_robot_triggered_estop()(로봇/백엔드가 스스로 정지해서 알려주는
        경로) 둘 다 여기를 거치게 해서, "누가 정지시켰든" 오버레이와 재개(RESUME)
        버튼이 항상 뜨도록 함 - 예전에는 이 UI 잠금이 emergency_stop()(프론트 버튼)
        안에만 있어서, 로봇이 스스로(하드웨어 인터락/자체 안전정지) 멈춘 경우
        오버레이 자체가 안 떠서 재개 버튼이 어디에도 없는 문제가 있었음
        (실제로 겪은 문제 - 재개를 누를 방법이 없어 사이클이 영원히 멈춰있었음).
        """
        if self.is_estopped:
            return
        self.is_estopped = True

        # 오리지널 버건디 레드 테마 안전 전환 복구 완료
        self.top_bar.config(bg=self.COLOR_EMERGENCY_BG)
        self.title_label.config(bg=self.COLOR_EMERGENCY_BG, fg="white")
        self.current_tab_lbl.config(bg=self.COLOR_EMERGENCY_BG, fg="#ffccd2")
        self.status_label.config(text="!!! HARDWARE CRITICAL E-STOPPED !!!", fg=self.COLOR_RED, bg=self.COLOR_EMERGENCY_BG)

        self.side_bar.config(bg=self.COLOR_SIDEBAR)
        self.log_message(log_text)

        self._show_emergency_overlay()

    def _handle_robot_triggered_estop(self, reason: str):
        """
        WebSocket으로 들어온 PROCESS_STATE가 "로봇/백엔드 쪽에서 이미 정지했음"을
        알리는 경우 호출. 프론트 E-STOP 버튼을 누른 게 아니므로 client.post_robot_estop()은
        다시 보내지 않고(이미 멈춘 로봇에 중복 정지 명령을 보낼 이유가 없음)
        화면 잠금/오버레이만 띄운다 - 오버레이의 재개 버튼이 실제 RESUME 경로
        (client.post_robot_resume())로 이어지므로, 여기서부터도 정상적으로
        재개할 수 있게 됨.
        """
        self._enter_estop_ui(f"[CRITICAL ERROR] 로봇이 자체적으로 비상정지했습니다 ({reason}).")

    def emergency_stop(self):
        self._enter_estop_ui("[CRITICAL ERROR] EMERGENCY STOP ACTIVATED. Hardware Interlock triggered.")

        # 안전 문제이므로 서버 응답을 기다리지 않고 UI는 즉시 잠근다(위 코드).
        # 실제 정지 명령은 별도 스레드로 전송하고, 전송 자체가 실패하면(네트워크 단절 등)
        # 소프트웨어 경로로 로봇에 정지 신호가 전달되지 않았을 수 있으므로 크게 경고한다.
        def worker():
            try:
                result = client.post_robot_estop()

                def on_success():
                    self.log_message(f"[Robot] EMERGENCY STOP 명령 전송 성공: {result}")

                self.root.after(0, on_success)

            except requests.exceptions.RequestException as exc:
                error_text = str(exc)

                def on_error():
                    self.log_message(f"[Robot] EMERGENCY STOP 명령 전송 실패: {error_text}")
                    messagebox.showerror(
                        "서버 연결 실패",
                        f"서버와 연결되지 않았습니다."
                        "소프트웨어 경로로 로봇이 정지되지 않았을 수 있으니, 즉시 물리 비상정지 버튼을 누르세요.",
                    )

                self.root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    def _show_emergency_overlay(self):
        if getattr(self, "emergency_overlay", None) is not None and self.emergency_overlay.winfo_exists(): 
            return 

        overlay = tk.Frame(self.content_area, bg=self.COLOR_EMERGENCY_BG, highlightbackground=self.COLOR_RED, highlightthickness=2)
        overlay.place(x=0, y=0, relwidth=1, relheight=1) 
        self.emergency_overlay = overlay 

        icon_lbl = tk.Label(overlay, text="[CRITICAL ALERT]", font=("DejaVu Sans Mono", 38, "bold"), fg=self.COLOR_RED, bg=self.COLOR_EMERGENCY_BG)
        icon_lbl.place(relx=0.5, rely=0.32, anchor="center") 

        title_lbl = tk.Label(overlay, text="EMERGENCY INTERLOCK ACTIVATED", font=("DejaVu Sans Mono", 22, "bold"), fg="white", bg=self.COLOR_EMERGENCY_BG)
        title_lbl.place(relx=0.5, rely=0.45, anchor="center") 

        sub_lbl = tk.Label(
            overlay, text=(
                "로봇은 정지 후 스스로 안전 위치(카메라 위치)로 물러나 대기합니다.\n"
                "물리 복구가 끝나면 아래 버튼이 자동으로 활성화됩니다 - 현장 안전을\n"
                "직접 확인한 뒤 누르면 로봇에 재개(RESUME) 신호가 전달되어 다음\n"
                "사이클(비전 재탐지)이 다시 시작됩니다."
            ),
            font=self.FONT_BODY_BOLD, fg="#ffccd2", bg=self.COLOR_EMERGENCY_BG, justify="center", wraplength=800
        )
        sub_lbl.place(relx=0.5, rely=0.55, anchor="center")
        overlay.bind("<Configure>", lambda e: sub_lbl.config(wraplength=max(e.width - 80, 240)))

        # 처음엔 비활성 상태로 띄움 - 로봇이 아직 물리 복구(상승->홈->카메라) 중일 수
        # 있어서, 그 이동이 다 끝나기 전에 눌러버리면 아직 wait_for_resume()을 호출하지도
        # 않은 로봇 쪽 resume_requested 이벤트가 조기 set()된 뒤 그대로 버려지는 문제가
        # 있었음(실제로 겪은 문제 - 버튼을 눌러도 로봇이 안 움직임). 로봇이 실제로
        # wait_for_resume()에 들어가기 직전에만 발행하는 PROCESS_STATE="waiting_for_resume"을
        # 받았을 때(_enable_resume_button) 비로소 눌러도 의미가 있으므로, 그 전까지는
        # 버튼 자체를 눌러도 아무 신호가 전달되지 않게 비활성화해둔다.
        btn_resume = tk.Button(
            overlay, text=self.BTN_TEXT_RESUME_WAITING, font=self.FONT_BODY_BOLD, bg=self.COLOR_TEXT_MUTED, fg="white",
            activebackground="white", activeforeground=self.COLOR_EMERGENCY_BG, bd=0, relief="flat", padx=25, pady=12,
            cursor="watch", state="disabled", command=self._confirm_and_resume,
        )
        btn_resume.place(relx=0.5, rely=0.68, anchor="center")
        self.btn_emergency_resume = btn_resume

        overlay.lift()

    def _enable_resume_button(self):
        # 로봇 쪽이 물리 복구를 끝내고 실제로 wait_for_resume()에 들어가기 직전에만
        # 발행하는 PROCESS_STATE="waiting_for_resume"을 받았을 때 호출됨 - 그 전까지
        # 비활성화해뒀던 재개 버튼을 이제서야 눌러도 되게 활성화한다 (_show_emergency_overlay
        # 참고). 이미 활성화돼 있거나(중복 수신) 사용자가 이미 눌러 전송 중인 상태
        # ("재개 신호 전송 중...")면 건드리지 않음.
        btn = getattr(self, "btn_emergency_resume", None)
        if btn is None or not btn.winfo_exists():
            return
        if btn.cget("text") != self.BTN_TEXT_RESUME_WAITING:
            return
        btn.config(state="normal", text=self.BTN_TEXT_RESUME_READY, bg=self.COLOR_RED, cursor="hand2")

    def _confirm_and_resume(self):
        # 재개 신호가 실제로 로봇에 전달됐는지 확인하기 전까지는 오버레이를
        # 닫지 않음(요구사항: 버튼이 로봇 동작과 무관하게 화면만 닫던 이전 동작을
        # 실제 재개 트리거로 바꿈). 이중 클릭/중복 요청 방지를 위해 전송 중엔
        # 버튼을 비활성화함.
        btn = getattr(self, "btn_emergency_resume", None)
        if btn is not None and btn.winfo_exists():
            btn.config(state="disabled", text="재개 신호 전송 중...")

        def worker():
            try:
                result = client.post_robot_resume()

                def on_success():
                    self.log_message(f"[Robot] RESUME 명령 전송 성공: {result}")
                    self._clear_emergency_overlay()

                self.root.after(0, on_success)

            except requests.exceptions.RequestException as exc:
                error_text = str(exc)

                def on_error():
                    self.log_message(f"[Robot] RESUME 명령 전송 실패: {error_text}")
                    messagebox.showerror("서버 연결 실패", "서버와 연결되지 않아 재개 신호를 보내지 못했습니다. 다시 시도해주세요.")
                    if btn is not None and btn.winfo_exists():
                        btn.config(state="normal", text=self.BTN_TEXT_RESUME_READY)

                self.root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    def _clear_emergency_overlay(self):
        self.is_estopped = False

        if getattr(self, "emergency_overlay", None) is not None and self.emergency_overlay.winfo_exists():
            self.emergency_overlay.destroy()
        self.emergency_overlay = None
        self.btn_emergency_resume = None

        self.top_bar.config(bg=self.COLOR_SIDEBAR)
        self.title_label.config(bg=self.COLOR_SIDEBAR, fg=self.COLOR_BLUE)
        self.current_tab_lbl.config(bg=self.COLOR_SIDEBAR, fg=self.COLOR_TEXT_MUTED)
        self.status_label.config(text="SYSTEM STATUS: RUNNING", fg=self.COLOR_GREEN, bg=self.COLOR_SIDEBAR)

        # 로봇 쪽 robot_command_callback은 RESUME을 받으면 E-STOP 게이트뿐 아니라
        # MANUAL_PAUSE(pause_requested)도 함께 clear()함 - 그래서 이 재개가 끝나면
        # 실제로는 일시정지도 같이 풀린 상태이므로, 버튼 라벨도 그에 맞춰 되돌린다.
        self.is_paused = False
        self.btn_main_pause.config(text="PAUSE ROBOT MOTION", state="normal")

        self.show_frame("monitor")
        self.log_message("[RECOVERY] 재개(RESUME) 신호 전송 완료. 시스템 정상 모니터링 상태로 복귀되었습니다.")

    def trigger_hitl_popup(self, payload=None):
        if self.is_estopped:
            return
        # 이미 팝업이 떠 있으면 새 창을 또 띄우지 않는다. VLA_ASK_HUMAN이 짧은 간격으로
        # 여러 번 와도(같은 물체에 대한 재질문/디바운스 실패 등) 창이 계속 쌓이는 걸 방지.
        if self.hitl_popup is not None:
            try:
                if self.hitl_popup.winfo_exists():
                    self.log_message("[HITL] 이미 열려있는 팝업이 있어 새 팝업을 건너뜀")
                    return
            except tk.TclError:
                pass
            self.hitl_popup = None
        if payload is None:
            payload = {"fruit_type": "사과 (Apple)", "condition": "색상 모호성 및 미세 멍", "confidence": 0.512} 

        fruit_type = payload.get("fruit_type", "unknown") 
        condition = payload.get("condition", "unknown") 
        confidence = payload.get("confidence") or 0.0 
        session_id = payload.get("session_id") 

        self.hitl_fruit_type = fruit_type
        self.hitl_condition = condition
        self.hitl_session_id = session_id

        tts_text = payload.get("tts_text") or payload.get("question_text") or self._build_ask_human_tts_text(fruit_type, condition, confidence)
        self.append_tts_question(tts_text)

        self.log_message(f"[HITL Alert] 예외 물체 감지 수동 개입 필요 ({fruit_type})")

        hitl_win = tk.Toplevel(self.root) 
        hitl_win.title("Human-In-The-Loop Exception Manager Multi-Viewer") 
        hitl_win.configure(bg=self.COLOR_BG) 
        self.hitl_popup = hitl_win 

        def _on_popup_closed():
            if self.hitl_popup is hitl_win: 
                self.hitl_popup = None 
                self.hitl_cam_label = None
            hitl_win.destroy() 

        hitl_win.protocol("WM_DELETE_WINDOW", _on_popup_closed)
        popup_w, popup_h = self.center_popup(hitl_win, width=750, height=420)
        # 팝업을 작게 줄여도 안내문/버튼이 겹치지 않도록 최소 크기 강제하되,
        # center_window()와 동일한 이유로 실제 화면 크기(popup_w/popup_h로 이미
        # clamp됨)보다 커지지 않게 함.
        hitl_win.minsize(min(650, popup_w), min(420, popup_h))
        hitl_win.grab_set()

        lbl_alert = tk.Label(hitl_win, text="[WARNING] AI 판단 신뢰도 저하 상황 발생 - 작업자 수동 지시 대기", font=self.FONT_BODY_BOLD, fg=self.COLOR_ORANGE, bg=self.COLOR_BG, pady=10, wraplength=700, justify="center")
        lbl_alert.pack(fill="x", side="top")
        hitl_win.bind("<Configure>", lambda e: lbl_alert.config(wraplength=max(e.width - 40, 200)))

        main_split = tk.Frame(hitl_win, bg=self.COLOR_BG)
        main_split.pack(fill="both", expand=True, padx=15, pady=10)

        left_panel = tk.Frame(main_split, bg=self.COLOR_BG)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        info_card = self._card_frame(left_panel, bg=self.COLOR_CARD_DARK)
        info_card.pack(fill="x", pady=(0, 10))

        img_dummy = tk.Label(
            info_card,
            text=(
                f"[ 판정 예외 농산물 데이터 ]\n\n"
                f"• 품목 종류: {fruit_type}\n"
                f"• 현재 상태: {condition}\n"
                f"• 신뢰도 점수: {confidence * 100:.1f}%"
            ),
            font=self.FONT_MONO, bg=self.COLOR_CARD_DARK, fg=self.COLOR_TEXT_MUTED, justify="left", anchor="w", padx=10, pady=15
        )
        img_dummy.pack(fill="x")

        lbl_question = tk.Label(left_panel, text="해당 농산물을 어떤 범주로 매핑할까요?", font=self.FONT_BODY_BOLD, fg=self.COLOR_TEXT, bg=self.COLOR_BG)
        lbl_question.pack(anchor="w", pady=5)

        right_panel = tk.LabelFrame(main_split, text="주 카메라 데이터 실시간 크롭 뷰", font=self.FONT_BODY_BOLD, fg=self.COLOR_TEXT, bg=self.COLOR_CARD, bd=2)
        right_panel.pack(side="right", fill="both", expand=True)

        self.hitl_cam_label = tk.Label(right_panel, bg="#000000", text="영상 동기화 대기 중...")
        self.hitl_cam_label.pack(fill="both", expand=True, padx=5, pady=5)

        if self.current_frame is not None:
            self.render_camera_frame(self.current_frame, self.hitl_cam_label, 360, 240)

        def _close_hitl_popup():
            if self.hitl_popup is not None:
                try:
                    if self.hitl_popup.winfo_exists():
                        self.hitl_popup.destroy()
                except tk.TclError:
                    pass
                self.hitl_popup = None
                self.hitl_cam_label = None

        def _submit_raw_answer(raw_answer, log_label):
            # session_id가 없다는 건 실제 세션이 아니라 로컬 테스트 팝업이라는 뜻
            # (실제 VLA_ASK_HUMAN 이벤트는 항상 session_id를 담아서 옴). 백엔드에
            # 대응하는 세션이 없어 POST /api/feedback이 항상 409로 실패하므로,
            # 서버 왕복 없이 그대로 반환해서 호출부가 로컬 처리를 하게 한다.
            if self.hitl_session_id is None:
                return False

            self.log_message(f"[HITL] '{log_label}' 선택 -> raw_answer='{raw_answer}' 전송 중...")

            def worker():
                try:
                    result = client.post_vla_feedback(
                        fruit_type=self.hitl_fruit_type,
                        condition=self.hitl_condition,
                        raw_answer=raw_answer,
                        session_id=self.hitl_session_id,
                    )
                    # /api/feedback은 "접수됐다"는 것만 알려줌. 실제 해석/학습 결과는
                    # backend가 비동기로 처리한 뒤 WS로 HITL_RESOLVED/HITL_SKIPPED(성공)
                    # 또는 HITL_STUCK(재질문 실패)을 브로드캐스트하므로, 팝업은 여기서
                    # 바로 닫지 않고 그 이벤트가 올 때(_handle_ws_message)까지 열어둠
                    def on_success():
                        self.log_message(f"[HITL Submitted] {result} - 처리 결과 대기 중...")

                    self.root.after(0, on_success)

                except requests.exceptions.RequestException as exc:
                    error_text = str(exc)

                    def on_error():
                        self.log_message(f"[HITL Error] 서버 연결 실패: {error_text} (팝업 유지, 재시도 가능)")

                    self.root.after(0, on_error)

            threading.Thread(target=worker, daemon=True).start()
            return True

        def manual_sort(category):
            destination = CATEGORY_TO_DESTINATION[category]
            raw_answer = _category_to_raw_answer(category)

            if _submit_raw_answer(raw_answer, category):
                return

            # 로컬 테스트 모드: 서버 왕복 없이 즉시 매핑 완료 처리
            self.counts[category] = self.counts.get(category, 0) + 1
            if category in self.count_labels:
                self.count_labels[category].config(text=f"{self.counts[category]} 개")
            self.log_message(f"[HITL Test] '{category}' 선택 -> destination={destination} (테스트 모드, 로컬 처리)")
            # parent를 hitl_win으로 지정 - hitl_win이 grab_set()으로 입력을
            # 독점하고 있는 동안 안내 창이 그 뒤에 가려 안 보이는 문제를 막음.
            messagebox.showinfo(
                "매핑 완료",
                f"'{self.hitl_fruit_type}'을(를) [{category}]로 매핑 완료했습니다. (테스트 모드)",
                parent=hitl_win if hitl_win.winfo_exists() else self.root,
            )
            _close_hitl_popup()

        def manual_skip():
            # 정책 학습도, 박스 이동 결정도 하지 않고 이 사과는 보류한 채 다음
            # 사과부터 처리하도록 넘어감 (hitl_state_machine.py의
            # destination=="skip" 분기 - vision_bridge에 좌표만 잠시 제외
            # 등록하고 세션을 종료함. 완료되면 WS로 HITL_SKIPPED가 옴).
            if _submit_raw_answer(HITL_SKIP_RAW_ANSWER, "무시"):
                return

            # 로컬 테스트 모드: 서버 왕복 없이 즉시 종료 (카운트는 증가시키지 않음 -
            # 아직 아무 분류도 정해지지 않았으므로)
            self.log_message(f"[HITL Test] '무시' 선택 (테스트 모드, 로컬 처리)")
            messagebox.showinfo(
                "무시 처리",
                f"'{self.hitl_fruit_type}'을(를) 보류하고 다른 사과부터 처리합니다. (테스트 모드)",
                parent=hitl_win if hitl_win.winfo_exists() else self.root,
            )
            _close_hitl_popup()

        btn_box = tk.Frame(left_panel, bg=self.COLOR_BG)
        btn_box.pack(fill="x", pady=5)

        colors = {"판매": self.COLOR_GREEN, "못난이": self.COLOR_ORANGE, "가공용": self.COLOR_BLUE, "폐기": self.COLOR_RED}
        for cat, color in colors.items():
            btn = tk.Button(
                btn_box, text=cat, bg=color, fg="white", activebackground=color, bd=0, relief="flat",
                font=self.FONT_BODY_BOLD, width=7, pady=10, cursor="hand2", command=lambda c=cat: manual_sort(c),
            )
            btn.pack(side="left", padx=3, expand=True, fill="x")

        btn_skip = tk.Button(
            btn_box, text="무시\n(다음 사과)", bg=self.COLOR_TEXT_MUTED, fg="white", activebackground=self.COLOR_TEXT_MUTED,
            bd=0, relief="flat", font=self.FONT_BODY_BOLD, width=7, pady=10, cursor="hand2", command=manual_skip,
        )
        btn_skip.pack(side="left", padx=3, expand=True, fill="x") 

    def on_closing(self):
        self.ws_client.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VLASorterDashboard(root)
    root.mainloop()
    