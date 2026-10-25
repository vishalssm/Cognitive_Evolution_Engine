import tkinter as tk
import paho.mqtt.client as mqtt
import json
import time

BROKER = "broker.hivemq.com"
TOPIC = "cee/telemetry"

# --- GLOBAL DATA ---
current_rpm, current_imu, current_pitch, current_ultra, system_temp = 0.0, 0.0, 0.0, 0.0, 0.0
knowledge_graph = []
rule_count = 1
was_anomalous = False

# UI Latch Variables
alert_hold_counter = 0  
last_error_trigger = "Awaiting initial event..."
last_rule_action = "..."

# --- MQTT CALLBACK ---
def on_message(client, userdata, message):
    global current_rpm, current_imu, current_pitch, current_ultra, system_temp
    global was_anomalous, knowledge_graph, rule_count
    global alert_hold_counter, last_error_trigger, last_rule_action
    
    try:
        data = json.loads(message.payload.decode("utf-8"))
        current_rpm = data.get("wheel_rpm_avg", 0)
        current_imu = data.get("imu_accel_x", 0)
        current_pitch = data.get("imu_pitch_deg", 0)
        current_ultra = data.get("ultrasonic_cm", 0)
        system_temp = data.get("system_temp_c", 0)
        
        detected_rule = None
        alert_msg = ""

        # --- REASONING ENGINE ---
        if current_rpm == 0 and current_imu == 0 and current_ultra == 0 and system_temp == 0:
            alert_msg = "CRITICAL: TELEMETRY PIPELINE SEVERED"
            detected_rule = "IF data_stream_lost THEN trigger_hardware_safing_mode()"
        elif system_temp > 75.0:
            alert_msg = "CRITICAL: THERMAL OVERLOAD"
            detected_rule = "IF cpu_temp > 75C THEN throttle_cognitive_hz(1)"
        elif current_ultra > 900:
            alert_msg = "PERCEPTION ERROR: SENSOR BLINDNESS"
            detected_rule = "IF radar_invalid THEN navigate_by_imu_only()"
        elif current_ultra < 10.0 and current_imu < 0:
            alert_msg = "IMPACT: WALL COLLISION"
            detected_rule = "IF impact_detected THEN reverse_and_recalculate()"
        elif current_ultra < 25.0 and current_rpm > 50:
            alert_msg = "DYNAMIC OBSTACLE CROSSED"
            detected_rule = "IF proximity < 25cm THEN emergency_brake()"
        elif current_rpm > 120 and current_imu < 0.8:
            alert_msg = "KINEMATIC MISMATCH: WHEEL SLIP"
            detected_rule = "IF slip_detected THEN cap_motor_rpm(50%)"
        elif current_rpm < 50 and current_imu < 0.3 and current_ultra > 30:
            alert_msg = "HIGH RESISTANCE: MUD / SAND"
            detected_rule = "IF traction_loss THEN engage_max_torque_burst()"
        elif abs(current_pitch) > 15.0:
            alert_msg = "SEVERE INCLINE: OFF-ROAD"
            detected_rule = "IF pitch_variance_high THEN enable_4x4_crawl_mode()"

        # --- EVENT LATCHING ---
        if detected_rule:
            if not was_anomalous:
                # Add new rule to the TOP of the list
                rule_string = f"RULE_{rule_count:03d}: {detected_rule}"
                knowledge_graph.insert(0, rule_string)
                
                # Lock in the display data for the UI
                last_error_trigger = alert_msg
                last_rule_action = rule_string
                alert_hold_counter = 20  # 2 seconds of forced red screen
                
                rule_count += 1
                
            was_anomalous = True
        else:
            was_anomalous = False
            
    except Exception:
        pass

# --- START MQTT ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "CEE_Visualizer_UI")
client.on_message = on_message
client.connect(BROKER)
client.subscribe(TOPIC)
client.loop_start()

# --- BUILD ADVANCED TKINTER UI ---
root = tk.Tk()
root.title("CEE: Omni-Hazard Digital Twin")
root.geometry("1100x750")
root.configure(bg="#090c10")

# Fonts & Colors
F_TITLE = ("Consolas", 24, "bold")
F_HDR = ("Consolas", 14, "bold")
F_TXT = ("Consolas", 13)
F_CODE = ("Courier New", 14, "bold")

C_BG, C_PANEL, C_TEXT = "#090c10", "#161b22", "#c9d1d9"
C_GREEN, C_RED, C_BLUE, C_PURPLE = "#3fb950", "#ff7b72", "#58a6ff", "#bc8cff"

tk.Label(root, text="COGNITIVE EVOLUTION ENGINE (CEE)", font=F_TITLE, bg=C_BG, fg=C_BLUE).pack(pady=(15, 5))

# Top Frame (Telemetry + Status)
top_frame = tk.Frame(root, bg=C_BG)
top_frame.pack(fill="x", padx=25, pady=5)

# 1. Telemetry Panel
telemetry_frame = tk.Frame(top_frame, bg=C_PANEL, highlightbackground=C_BLUE, highlightthickness=2, padx=20, pady=15)
telemetry_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
tk.Label(telemetry_frame, text="12-DOF TELEMETRY", font=F_HDR, bg=C_PANEL, fg=C_TEXT).pack(anchor="w", pady=(0, 10))

lbl_rpm = tk.Label(telemetry_frame, text="Wheel Avg : 0.0 RPM", font=F_TXT, bg=C_PANEL, fg=C_GREEN)
lbl_rpm.pack(anchor="w")
lbl_imu = tk.Label(telemetry_frame, text="IMU X-Axis: 0.0 m/s²", font=F_TXT, bg=C_PANEL, fg=C_GREEN)
lbl_imu.pack(anchor="w")
lbl_pitch = tk.Label(telemetry_frame, text="IMU Pitch : 0.0 °", font=F_TXT, bg=C_PANEL, fg=C_GREEN)
lbl_pitch.pack(anchor="w")
lbl_ultra = tk.Label(telemetry_frame, text="Front RADAR: 0.0 cm", font=F_TXT, bg=C_PANEL, fg=C_GREEN)
lbl_ultra.pack(anchor="w")
lbl_temp = tk.Label(telemetry_frame, text="CPU Temp  : 0.0 °C", font=F_TXT, bg=C_PANEL, fg=C_GREEN)
lbl_temp.pack(anchor="w")

# 2. Engine Status Panel
engine_frame = tk.Frame(top_frame, bg=C_PANEL, highlightbackground=C_GREEN, highlightthickness=2, padx=20, pady=15)
engine_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
tk.Label(engine_frame, text="LIVE SYSTEM STATUS", font=F_HDR, bg=C_PANEL, fg=C_TEXT).pack(anchor="center", pady=(0, 10))

lbl_status = tk.Label(engine_frame, text="NOMINAL", font=("Consolas", 26, "bold"), bg=C_PANEL, fg=C_GREEN)
lbl_status.pack(anchor="center", pady=10)

# 3. Cognitive Mapping Panel
map_frame = tk.Frame(root, bg=C_PANEL, highlightbackground=C_PURPLE, highlightthickness=2, padx=20, pady=15)
map_frame.pack(fill="x", padx=25, pady=10)
tk.Label(map_frame, text="LATEST COGNITIVE MAPPING (ERROR -> SOLUTION)", font=F_HDR, bg=C_PANEL, fg=C_PURPLE).pack(anchor="w", pady=(0, 5))

lbl_last_error = tk.Label(map_frame, text="Trigger: None", font=F_CODE, bg=C_PANEL, fg=C_RED, justify="left")
lbl_last_error.pack(anchor="w", pady=2)
lbl_last_rule = tk.Label(map_frame, text="Action : None", font=F_CODE, bg=C_PANEL, fg=C_GREEN, justify="left")
lbl_last_rule.pack(anchor="w", pady=2)

# 4. Scrollable Active Memory Panel
graph_frame = tk.Frame(root, bg=C_PANEL, highlightbackground=C_BLUE, highlightthickness=2, padx=20, pady=15)
graph_frame.pack(fill="both", expand=True, padx=25, pady=5)
tk.Label(graph_frame, text="OPERATIONAL KNOWLEDGE GRAPH (ACTIVE MEMORY)", font=F_HDR, bg=C_PANEL, fg=C_TEXT).pack(anchor="w", pady=(0, 10))

# Create Scrollbar and Text Box
scroll_y = tk.Scrollbar(graph_frame)
scroll_y.pack(side="right", fill="y")

txt_graph = tk.Text(graph_frame, font=F_TXT, bg=C_PANEL, fg=C_TEXT, yscrollcommand=scroll_y.set, wrap="word", relief="flat", state="disabled", height=6)
txt_graph.pack(side="left", fill="both", expand=True)
scroll_y.config(command=txt_graph.yview)

# --- UI UPDATE LOOP ---
def update_dashboard():
    global alert_hold_counter
    
    lbl_rpm.config(text=f"Wheel Avg  : {current_rpm:6.1f} RPM")
    lbl_imu.config(text=f"IMU X-Axis : {current_imu:6.2f} m/s²")
    lbl_pitch.config(text=f"IMU Pitch  : {current_pitch:6.1f} °")
    lbl_ultra.config(text=f"Front RADAR: {current_ultra:6.1f} cm")
    lbl_temp.config(text=f"CPU Temp   : {system_temp:6.1f} °C")
    
    lbl_temp.config(fg=C_RED if system_temp > 75 else C_GREEN)
    lbl_ultra.config(fg=C_RED if current_ultra > 900 or current_ultra < 10 else C_GREEN)
    lbl_pitch.config(fg=C_RED if abs(current_pitch) > 15 else C_GREEN)
    
    if alert_hold_counter > 0:
        engine_frame.config(highlightbackground=C_RED)
        lbl_status.config(text="!! ANOMALY DETECTED !!", fg=C_RED)
        alert_hold_counter -= 1
    else:
        engine_frame.config(highlightbackground=C_GREEN)
        lbl_status.config(text="SYSTEM NOMINAL", fg=C_GREEN)
        
    lbl_last_error.config(text=f"Detected Hazard  -> {last_error_trigger}")
    lbl_last_rule.config(text=f"Generated Logic  -> {last_rule_action}")
        
    # Safely update the scrollable text box
    if knowledge_graph:
        graph_text = "\n\n".join([f"  [+] {rule}" for rule in knowledge_graph])
        txt_graph.config(state="normal")       # Unlock box
        txt_graph.delete("1.0", tk.END)        # Clear old text
        txt_graph.insert(tk.END, graph_text)   # Inject new text
        txt_graph.config(state="disabled")     # Lock box again so user can't type in it
        
    root.after(100, update_dashboard)

update_dashboard()
root.mainloop()
client.loop_stop()