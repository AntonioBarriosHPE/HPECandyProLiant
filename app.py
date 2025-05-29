import asyncio
import numpy as np
import base64
import json
import logging
import time
import os
from pathlib import Path
import platform
import sys
import random
from enum import Enum
from collections import deque
# import cProfile # Profiling removed for now
# import pstats   # Profiling removed for now
# from pstats import SortKey # Profiling removed for now
from openvino.runtime import Core

from aiohttp import web, WSCloseCode

import cv2
from PIL import Image
import io

from hardware_stubs import GPIO_CONTROLLER, is_el300_present

# --- Configuration & Globals ---
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 9000
WEBSOCKET_PATH = "/ws"
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_IR_BASE_PATH = SCRIPT_DIR / "openvino_models" / "ir" / "public"
STATIC_FILES_PATH = SCRIPT_DIR / "static"

FACE_DETECTION_MODEL_XML = MODEL_IR_BASE_PATH / "face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
EMOTION_RECOGNITION_MODEL_XML = MODEL_IR_BASE_PATH / "emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml"

DETECTION_THRESHOLD = 0.5
EMOTIONS_CLASSES = ['neutral', 'happy', 'sad', 'surprise', 'angry']
EMOTION_SMOOTHING_WINDOW_SIZE = 3
SMOOTHING_CONSENSUS_THRESHOLD = 2

app_settings = {
    "happy_time_ms": 5000,
    "motor_on_ms": 2000,
    "cooldown_s": 5
}

class GameState(Enum):
    IDLE = 0
    INITIAL_SMILE = 1
    MINIGAME_BUFFER_SMILE = 2
    MINIGAME_TRANSITION = 3
    MINIGAME_PLAYING = 4
    GAME_OVER = 5

minigame_settings = {
    "initial_smile_buffer_ms": 3000,
    "initial_hold_duration_ms": 5000,
    "initial_transition_ms": 2500,
    "duration_decay_factor": 0.95,
    "min_hold_duration_ms": 800,
    "min_transition_ms": 700,
    "max_total_game_duration_s": 300,
    "allowed_emotions": ['sad', 'happy', 'angry', 'neutral']
}
GAME_OVER_RESET_DELAY_S = 5.0

# --- Global State Variables ---
is_demo_active = False
video_producer_task = None
active_websockets = set()
current_game_state = GameState.IDLE
is_initial_happy_streak = False
initial_happy_streak_start_time_ms = 0
leaderboard_snapshot_b64 = None
leaderboard_snapshot_taken_this_session = False
candy_eligible_again_time_s = 0
dispenser_motor_stop_time_s = 0
minigame_start_time_overall_ms = 0
minigame_current_score_ms = 0
current_target_emotion = None
current_target_emotion_set_time_ms = 0
current_round_deadline_ms = 0
current_hold_duration_ms = minigame_settings["initial_hold_duration_ms"]
current_transition_ms = minigame_settings["initial_transition_ms"]
last_game_over_reason = ""
game_over_display_start_time_s = 0
emotion_history = deque(maxlen=EMOTION_SMOOTHING_WINDOW_SIZE)
stable_primary_emotion_name = None

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger("hpe_candy_proliant")

ie_core = None
face_detection_net = None
face_detection_exec_net = None
emotion_net = None
emotion_exec_net = None
face_det_input_shape_nchw = {}
emotion_input_shape_nchw = {}
models_initialized_successfully = False

def initialize_openvino_models():
    global ie_core, face_detection_net, face_detection_exec_net, emotion_net, emotion_exec_net
    global face_det_input_shape_nchw, emotion_input_shape_nchw, models_initialized_successfully
    models_initialized_successfully = False
    logger.info("Initializing OpenVINO Core and models...")
    try:
        ie_core = Core()
        logger.info(f"Available OpenVINO devices: {ie_core.available_devices}")
        if not FACE_DETECTION_MODEL_XML.exists():
            logger.error(f"FD XML not found: {FACE_DETECTION_MODEL_XML}"); return False
        if not EMOTION_RECOGNITION_MODEL_XML.exists():
            logger.error(f"ER XML not found: {EMOTION_RECOGNITION_MODEL_XML}"); return False
        
        logger.info(f"Loading FD model: {FACE_DETECTION_MODEL_XML}")
        face_detection_net = ie_core.read_model(model=str(FACE_DETECTION_MODEL_XML))
        fd_input_layer = face_detection_net.input(0)
        n, c, h, w = map(int, fd_input_layer.shape)
        face_det_input_shape_nchw = {'n': n, 'c': c, 'h': h, 'w': w}
        face_detection_exec_net = ie_core.compile_model(model=face_detection_net, device_name="AUTO")
        logger.info(f"FD model compiled. Requested: AUTO. Shape: {face_det_input_shape_nchw}")

        logger.info(f"Loading ER model: {EMOTION_RECOGNITION_MODEL_XML}")
        emotion_net = ie_core.read_model(model=str(EMOTION_RECOGNITION_MODEL_XML))
        em_input_layer = emotion_net.input(0)
        n, c, h, w = map(int, em_input_layer.shape)
        emotion_input_shape_nchw = {'n': n, 'c': c, 'h': h, 'w': w}
        emotion_exec_net = ie_core.compile_model(model=emotion_net, device_name="AUTO")
        logger.info(f"ER model compiled. Requested: AUTO. Shape: {emotion_input_shape_nchw}")
        
        models_initialized_successfully = True; return True
    except Exception as e:
        logger.error(f"Error initializing OpenVINO models: {e}", exc_info=True)
        models_initialized_successfully = False; return False

def preprocess_frame_face_detection(frame_hwc):
    if not models_initialized_successfully or not face_det_input_shape_nchw: return None, None
    h, w = face_det_input_shape_nchw['h'], face_det_input_shape_nchw['w']
    resized_frame = cv2.resize(frame_hwc, (w, h))
    transposed_frame = resized_frame.transpose(2, 0, 1)
    return np.expand_dims(transposed_frame, 0), resized_frame

def detect_faces_from_input_frame(frame_for_face_det_nchw):
    if not models_initialized_successfully or frame_for_face_det_nchw is None or \
       not face_detection_net or not face_detection_exec_net: return []
    results = face_detection_exec_net.infer_new_request({face_detection_net.input(0).get_any_name(): frame_for_face_det_nchw})
    detections = results[face_detection_net.output(0).get_any_name()]
    faces = []
    model_input_h, model_input_w = face_det_input_shape_nchw['h'], face_det_input_shape_nchw['w']
    for detection in detections[0][0]:
        confidence = float(detection[2])
        if confidence > DETECTION_THRESHOLD:
            x1, y1 = int(detection[3] * model_input_w), int(detection[4] * model_input_h)
            x2, y2 = int(detection[5] * model_input_w), int(detection[6] * model_input_h)
            area = (x2 - x1) * (y2 - y1)
            faces.append({'rect_model_input_coords': (x1, y1, x2, y2), 'confidence': confidence, 'area': area})
    faces.sort(key=lambda x: x['area'], reverse=True)
    return faces

def preprocess_face_emotion_recognition(face_roi_hwc):
    if not models_initialized_successfully or not emotion_input_shape_nchw or 'h' not in emotion_input_shape_nchw: return None
    h, w = emotion_input_shape_nchw['h'], emotion_input_shape_nchw['w']
    resized_face = cv2.resize(face_roi_hwc, (w, h))
    transposed_face = resized_face.transpose(2, 0, 1)
    return np.expand_dims(transposed_face, 0)

def recognize_emotion(face_roi_for_emotion_hwc): # Changed param name for clarity
    if not models_initialized_successfully or not emotion_input_shape_nchw or \
       'h' not in emotion_input_shape_nchw or not emotion_net or not emotion_exec_net: return None, 0.0
    if face_roi_for_emotion_hwc is None or face_roi_for_emotion_hwc.size == 0: return None, 0.0
    try:
        input_for_emotion_model = preprocess_face_emotion_recognition(face_roi_for_emotion_hwc)
        if input_for_emotion_model is None: return None, 0.0
        results = emotion_exec_net.infer_new_request({emotion_net.input(0).get_any_name(): input_for_emotion_model})
        probabilities = results[emotion_net.output(0).get_any_name()][0].flatten()
        emotion_index = np.argmax(probabilities)
        return EMOTIONS_CLASSES[emotion_index], float(probabilities[emotion_index])
    except Exception as e:
        logger.warning(f"Emotion recognition error: {e}", exc_info=False); return None, 0.0

def frame_to_base64(frame, quality=75):
    if frame is None: logger.warning("frame_to_base64 received a None frame."); return None
    try:
        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and buffer is not None and buffer.size > 0:
            return base64.b64encode(buffer).decode('utf-8')
        else:
            logger.warning('cv2.imencode failed. Using Pillow fallback.')
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with io.BytesIO() as byte_stream:
                Image.fromarray(rgb_frame).save(byte_stream, format='JPEG', quality=quality)
                return base64.b64encode(byte_stream.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Exception in frame_to_base64: {e}", exc_info=True); return None

async def broadcast(message_dict):
    if active_websockets:
        results = await asyncio.gather(
            *[ws.send_str(json.dumps(message_dict)) for ws in active_websockets if not ws.closed],
            return_exceptions=True
        )
        # Error logging for send failures removed for brevity, but can be added back if needed

async def manage_dispenser():
    global dispenser_motor_stop_time_s
    while True:
        current_time_s = time.time()
        if dispenser_motor_stop_time_s > 0 and current_time_s >= dispenser_motor_stop_time_s:
            GPIO_CONTROLLER.turn_dispenser_off()
            dispenser_motor_stop_time_s = 0
            logger.info("Dispenser motor turned OFF by manager.")
        await asyncio.sleep(0.1)

def reset_game_completely():
    global current_game_state, is_initial_happy_streak, initial_happy_streak_start_time_ms
    global leaderboard_snapshot_b64, leaderboard_snapshot_taken_this_session
    global minigame_start_time_overall_ms, minigame_current_score_ms, current_target_emotion
    global current_target_emotion_set_time_ms, current_round_deadline_ms
    global current_hold_duration_ms, current_transition_ms, last_game_over_reason
    global game_over_display_start_time_s, emotion_history, stable_primary_emotion_name
    logger.info("Resetting game state to IDLE, clearing leaderboard snapshot and related flags.")
    current_game_state = GameState.IDLE
    is_initial_happy_streak = False
    initial_happy_streak_start_time_ms = 0
    leaderboard_snapshot_b64 = None
    leaderboard_snapshot_taken_this_session = False
    minigame_start_time_overall_ms = 0
    minigame_current_score_ms = 0
    current_target_emotion = None
    current_target_emotion_set_time_ms = 0
    current_round_deadline_ms = 0
    current_hold_duration_ms = minigame_settings["initial_hold_duration_ms"]
    current_transition_ms = minigame_settings["initial_transition_ms"]
    last_game_over_reason = ""
    game_over_display_start_time_s = 0
    emotion_history.clear()
    stable_primary_emotion_name = None

def start_new_minigame_round():
    global current_game_state, current_target_emotion, current_target_emotion_set_time_ms
    global current_round_deadline_ms, current_transition_ms
    possible_next_emotions = [e for e in minigame_settings["allowed_emotions"] if e != current_target_emotion]
    if not possible_next_emotions: possible_next_emotions = minigame_settings["allowed_emotions"]
    current_target_emotion = random.choice(possible_next_emotions)
    current_target_emotion_set_time_ms = int(time.time() * 1000)
    current_round_deadline_ms = current_target_emotion_set_time_ms + current_transition_ms
    current_game_state = GameState.MINIGAME_TRANSITION
    logger.info(f"Minigame Next: {current_target_emotion}. Transition: {current_transition_ms}ms.")

async def video_stream_producer_loop():
    global is_demo_active, models_initialized_successfully, current_game_state
    global is_initial_happy_streak, initial_happy_streak_start_time_ms
    global leaderboard_snapshot_b64, leaderboard_snapshot_taken_this_session
    global candy_eligible_again_time_s, dispenser_motor_stop_time_s
    global minigame_start_time_overall_ms, minigame_current_score_ms
    global current_target_emotion, current_target_emotion_set_time_ms, current_round_deadline_ms
    global current_hold_duration_ms, current_transition_ms, last_game_over_reason, game_over_display_start_time_s
    global emotion_history, stable_primary_emotion_name

    if not models_initialized_successfully:
        logger.error("Models not init. Video stream cannot proceed.")
        await broadcast({"type": "error", "message": "Server models not ready."})
        is_demo_active = False; return
    
    reset_game_completely() 
    current_game_state = GameState.INITIAL_SMILE
    logger.info("video_stream_producer_loop started, game reset to INITIAL_SMILE.")

    cap = None # Camera capture object
    # Camera opening logic (same as your provided version)
    backends_to_try = []
    if platform.system() == "Windows": backends_to_try.append(cv2.CAP_DSHOW)
    elif platform.system() == "Darwin": backends_to_try.append(cv2.CAP_AVFOUNDATION)
    backends_to_try.append(None) 

    for cam_idx in range(5):
        for backend_api in backends_to_try:
            backend_name = "Default" if backend_api is None else \
                           "DSHOW" if backend_api == cv2.CAP_DSHOW else \
                           "AVFOUNDATION" if backend_api == cv2.CAP_AVFOUNDATION else str(backend_api)
            logger.info(f"Attempting Cam {cam_idx} with backend: {backend_name}")
            try:
                cap = cv2.VideoCapture(cam_idx, backend_api) if backend_api is not None else cv2.VideoCapture(cam_idx)
                if cap and cap.isOpened():
                    logger.info(f"Cam {cam_idx} ({backend_name}) opened. Setting properties...")
                    if platform.system() != "Darwin": cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    logger.info(f"Cam {cam_idx} actual settings: {actual_w}x{actual_h} @ {actual_fps:.2f} FPS")
                    ret_read, _ = cap.read()
                    if ret_read: logger.info(f"Cam {cam_idx} OK (backend: {backend_name})."); break 
                    if cap: cap.release(); cap=None; logger.warning(f"Opened Cam {cam_idx} ({backend_name}), but failed read.")
                elif cap: cap.release(); cap=None
            except Exception as e: logger.error(f"Ex opening Cam {cam_idx} ({backend_name}): {e}"); cap.release(); cap=None
        if cap and cap.isOpened(): break
    if not cap or not cap.isOpened(): 
        logger.error("No webcam."); await broadcast({"type":"error","message":"Webcam error."}); is_demo_active=False; return
    
    face_model_h, face_model_w = face_det_input_shape_nchw['h'], face_det_input_shape_nchw['w']
    first_frame_sent_this_session = False

    try:
        while is_demo_active:
            current_time_ms = int(time.time() * 1000)
            current_time_s = time.time()
            
            snapshot_generated_this_frame = False 
            candy_dispensed_this_server_cycle = False # NEW FLAG - reset each server processing cycle

            if current_game_state == GameState.GAME_OVER and game_over_display_start_time_s > 0:
                if (current_time_s - game_over_display_start_time_s) >= GAME_OVER_RESET_DELAY_S:
                    logger.info("Auto-resetting demo after GAME_OVER.")
                    reset_game_completely()
                    current_game_state = GameState.INITIAL_SMILE
            
            ret, original_frame_hwc = cap.read()
            if not ret or original_frame_hwc is None:
                logger.warning("No frame or empty frame from webcam.")
                await asyncio.sleep(0.05); continue

            frame_for_face_det_nchw, frame_resized_for_face_det_hwc = preprocess_frame_face_detection(original_frame_hwc)
            detected_faces = []
            if frame_for_face_det_nchw is not None:
                detected_faces = detect_faces_from_input_frame(frame_for_face_det_nchw)
            
            raw_detected_emotion = None
            current_face_roi_for_snapshot_hwc_original_scale = None

            if detected_faces:
                primary_face_data = detected_faces[0]
                fx_m, fy_m, fx2_m, fy2_m = primary_face_data['rect_model_input_coords']
                if original_frame_hwc is not None and frame_resized_for_face_det_hwc is not None and \
                   face_model_w > 0 and face_model_h > 0:
                    orig_h, orig_w = original_frame_hwc.shape[:2]
                    scale_x_to_orig = orig_w / face_model_w 
                    scale_y_to_orig = orig_h / face_model_h
                    orig_x1, orig_y1 = max(0, int(fx_m * scale_x_to_orig)), max(0, int(fy_m * scale_y_to_orig))
                    orig_x2, orig_y2 = min(orig_w, int(fx2_m * scale_x_to_orig)), min(orig_h, int(fy2_m * scale_y_to_orig))
                    if orig_x1 < orig_x2 and orig_y1 < orig_y2:
                         current_face_roi_for_snapshot_hwc_original_scale = original_frame_hwc[orig_y1:orig_y2, orig_x1:orig_x2]
                if fx_m < fx2_m and fy_m < fy2_m and frame_resized_for_face_det_hwc is not None:
                    roi_for_emotion_resized = frame_resized_for_face_det_hwc[fy_m:fy2_m, fx_m:fx2_m]
                    raw_detected_emotion, _ = recognize_emotion(roi_for_emotion_resized)

            # Emotion Smoothing (same as your provided version)
            if raw_detected_emotion: emotion_history.append(raw_detected_emotion)
            elif not detected_faces: emotion_history.clear(); stable_primary_emotion_name = None
            if len(emotion_history) == emotion_history.maxlen:
                counts = {em:emotion_history.count(em) for em in set(emotion_history)}
                if counts: 
                    most_frequent = max(counts,key=counts.get)
                    stable_primary_emotion_name = most_frequent if counts[most_frequent]>=SMOOTHING_CONSENSUS_THRESHOLD else stable_primary_emotion_name
                elif not detected_faces: stable_primary_emotion_name = None
            elif raw_detected_emotion and len(emotion_history)<emotion_history.maxlen: stable_primary_emotion_name = raw_detected_emotion
            elif not detected_faces : stable_primary_emotion_name = None
            
            primary_emotion_name_for_logic = stable_primary_emotion_name
            ui_in_candy_cooldown = current_time_s < candy_eligible_again_time_s
            
            # --- Game State Machine ---
            if current_game_state == GameState.INITIAL_SMILE:
                if primary_emotion_name_for_logic == "happy":
                    if not is_initial_happy_streak:
                        is_initial_happy_streak = True; initial_happy_streak_start_time_ms = current_time_ms
                        leaderboard_snapshot_taken_this_session = False
                        logger.info("INITIAL_SMILE: New happy streak. leaderboard_snapshot_taken_this_session=False.")
                    smile_duration_ms = current_time_ms - initial_happy_streak_start_time_ms
                    if not ui_in_candy_cooldown and smile_duration_ms >= app_settings["happy_time_ms"]:
                        if not leaderboard_snapshot_taken_this_session:
                            logger.info("INITIAL_SMILE: Smile threshold met for candy & snapshot.")
                            if current_face_roi_for_snapshot_hwc_original_scale is not None and \
                               current_face_roi_for_snapshot_hwc_original_scale.size > 0:
                                temp_b64 = frame_to_base64(current_face_roi_for_snapshot_hwc_original_scale)
                                if temp_b64:
                                    leaderboard_snapshot_b64 = temp_b64
                                    snapshot_generated_this_frame = True
                                    logger.info(f"INITIAL_SMILE: GLOBAL leaderboard_snapshot_b64 SET. Len: {len(leaderboard_snapshot_b64)}. snapshot_generated_this_frame=True.")
                                else: logger.warning("INITIAL_SMILE: frame_to_base64 failed for snapshot.")
                            else: logger.warning("INITIAL_SMILE: No valid face ROI for snapshot.")
                            
                            GPIO_CONTROLLER.turn_dispenser_on()
                            candy_dispensed_this_server_cycle = True # SET THE NEW FLAG
                            dispenser_motor_stop_time_s = current_time_s + (app_settings["motor_on_ms"] / 1000.0)
                            candy_eligible_again_time_s = dispenser_motor_stop_time_s + app_settings["cooldown_s"]
                            leaderboard_snapshot_taken_this_session = True
                            
                            current_game_state = GameState.MINIGAME_BUFFER_SMILE # State changes here
                            current_target_emotion = "happy"
                            minigame_start_time_overall_ms = current_time_ms
                            current_target_emotion_set_time_ms = current_time_ms
                            current_round_deadline_ms = current_time_ms + minigame_settings["initial_smile_buffer_ms"]
                            logger.info(f"INITIAL_SMILE: To MINIGAME_BUFFER_SMILE. Hold HAPPY for {minigame_settings['initial_smile_buffer_ms']}ms. candy_dispensed_this_server_cycle=True.")
                else: 
                    if is_initial_happy_streak:
                        is_initial_happy_streak = False; logger.info("INITIAL_SMILE: Happy streak broken.")
            
            elif current_game_state == GameState.MINIGAME_BUFFER_SMILE:
                if primary_emotion_name_for_logic == "happy":
                    if current_time_ms >= current_round_deadline_ms:
                        time_in_buffer = current_time_ms - current_target_emotion_set_time_ms
                        minigame_current_score_ms += time_in_buffer
                        logger.info(f"MINIGAME_BUFFER_SMILE: OK (+{time_in_buffer}ms). Start round.")
                        start_new_minigame_round()
                else:
                    last_game_over_reason = "Smile lost during buffer!"
                    logger.info(f"GAME OVER: {last_game_over_reason}"); current_game_state = GameState.GAME_OVER
                    game_over_display_start_time_s = current_time_s 
            
            elif current_game_state == GameState.MINIGAME_TRANSITION:
                if current_time_ms >= current_round_deadline_ms:
                    current_game_state = GameState.MINIGAME_PLAYING
                    current_target_emotion_set_time_ms = current_time_ms
                    current_round_deadline_ms = current_time_ms + current_hold_duration_ms
                    logger.info(f"MINIGAME_TRANSITION: To MINIGAME_PLAYING. Target: {current_target_emotion}. Hold: {current_hold_duration_ms}ms.")
            
            elif current_game_state == GameState.MINIGAME_PLAYING:
                if minigame_start_time_overall_ms > 0 and \
                   (current_time_ms - minigame_start_time_overall_ms) >= (minigame_settings["max_total_game_duration_s"] * 1000):
                    last_game_over_reason = "Max game time!"
                    logger.info(f"GAME OVER: {last_game_over_reason}"); current_game_state = GameState.GAME_OVER
                    game_over_display_start_time_s = current_time_s
                elif primary_emotion_name_for_logic == current_target_emotion:
                    if current_time_ms >= current_round_deadline_ms:
                        time_held = current_time_ms - current_target_emotion_set_time_ms
                        minigame_current_score_ms += time_held
                        logger.info(f"MINIGAME_PLAYING: Held {current_target_emotion} (+{time_held}ms). Bonus: {minigame_current_score_ms}ms.")
                        current_hold_duration_ms = max(minigame_settings["min_hold_duration_ms"], int(current_hold_duration_ms * minigame_settings["duration_decay_factor"]))
                        current_transition_ms = max(minigame_settings["min_transition_ms"], int(current_transition_ms * minigame_settings["duration_decay_factor"]))
                        start_new_minigame_round()
                else: 
                    if current_time_ms > (current_target_emotion_set_time_ms + 500): 
                        last_game_over_reason = f"Needed {current_target_emotion}, got {primary_emotion_name_for_logic or 'None'}!"
                        logger.info(f"GAME OVER: {last_game_over_reason}")
                        current_game_state = GameState.GAME_OVER
                        game_over_display_start_time_s = current_time_s
            
            output_frame_hwc = original_frame_hwc.copy()
            if detected_faces and face_model_w > 0 and face_model_h > 0:
                output_h, output_w = output_frame_hwc.shape[:2]
                scale_x_draw, scale_y_draw = output_w / face_model_w, output_h / face_model_h
                fx_m, fy_m, fx2_m, fy2_m = detected_faces[0]['rect_model_input_coords']
                dx1, dy1 = int(fx_m * scale_x_draw), int(fy_m * scale_y_draw)
                dx2, dy2 = int(fx2_m * scale_x_draw), int(fy2_m * scale_y_draw)
                cv2.rectangle(output_frame_hwc, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                if stable_primary_emotion_name:
                    cv2.putText(output_frame_hwc, stable_primary_emotion_name.capitalize(), (dx1, dy1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame_b64_for_client = frame_to_base64(output_frame_hwc)
            ready_signal_for_this_frame = False 
            if not first_frame_sent_this_session and frame_b64_for_client: 
                ready_signal_for_this_frame = True; first_frame_sent_this_session = True
            
            status_payload = {
                "type": "status_update", "current_detected_emotion": stable_primary_emotion_name,
                "is_happy_for_initial_smile": primary_emotion_name_for_logic=="happy" and current_game_state==GameState.INITIAL_SMILE,
                "in_candy_cooldown": ui_in_candy_cooldown,
                "candy_eligible_again_time": candy_eligible_again_time_s * 1000,
                "current_time_ms": current_time_ms,
                "initial_smile_streak_start_time_ms": initial_happy_streak_start_time_ms if is_initial_happy_streak and current_game_state==GameState.INITIAL_SMILE else 0,
                "leaderboard_snapshot_b64": leaderboard_snapshot_b64,
                "new_snapshot_taken": snapshot_generated_this_frame,
                "candy_was_dispensed_this_cycle": candy_dispensed_this_server_cycle, # ADDED FLAG
                "game_state": current_game_state.name, "target_emotion": current_target_emotion,
                "round_deadline_ms": current_round_deadline_ms, "minigame_score_ms": minigame_current_score_ms,
                "game_over_reason": last_game_over_reason if current_game_state==GameState.GAME_OVER else "",
                "game_over_reset_timer_s": max(0, GAME_OVER_RESET_DELAY_S-(current_time_s-game_over_display_start_time_s)) if current_game_state==GameState.GAME_OVER and game_over_display_start_time_s>0 else -1,
                "ready": ready_signal_for_this_frame
            }
            await broadcast(status_payload)
            
            if frame_b64_for_client: 
                await broadcast({"type": "video_frame", "data": frame_b64_for_client})
            else: logger.warning("Skipping video_frame broadcast (None/empty).")
            
            await asyncio.sleep(0.015) 
            
    except asyncio.CancelledError: logger.info("Video stream producer loop was cancelled.")
    except Exception as e:
        logger.error(f"Unhandled error in video_stream_producer_loop: {e}", exc_info=True)
        # await broadcast({"type": "error", "message": f"Server error: {str(e)}"}) # Be cautious with awaits in broad exceptions
    finally:
        if cap: cap.release(); logger.info("Webcam released in video_stream_producer_loop finally.")
        reset_game_completely()
        # is_demo_active = False # This is managed by the calling handler (websocket_handler)
        logger.info("Video stream producer loop finished its finally block cleanup.")
        # DO NOT broadcast demo_stopped from here. It's handled by websocket_handler.

async def websocket_handler(request):
    global is_demo_active, video_producer_task, active_websockets
    ws = web.WebSocketResponse(); await ws.prepare(request); active_websockets.add(ws)
    logger.info(f"WS client connected: {request.remote}")
    await ws.send_json({"type": "config_update", "settings": app_settings})
    if models_initialized_successfully:
        await ws.send_json({"type": "status_update", "ready": True, "game_state": current_game_state.name, "message": "Server ready."})
        logger.info("Sent initial explicit ready signal to newly connected client.")

    try:
        async for msg in ws: 
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data); cmd = data.get("command")
                    logger.info(f"WS recv: {cmd} with data: {data}")
                    if cmd == "start_demo":
                        if not is_demo_active:
                            if not models_initialized_successfully:
                                await ws.send_json({"type":"error","message":"Server models not ready."}); continue
                            is_demo_active = True
                            logger.info("start_demo: is_demo_active=True.")
                            if video_producer_task and not video_producer_task.done():
                                logger.info("start_demo: Cancelling existing video_producer_task.")
                                video_producer_task.cancel()
                                try: await video_producer_task
                                except asyncio.CancelledError: logger.info("start_demo: Previous video_producer_task cancelled.")
                            video_producer_task = asyncio.create_task(video_stream_producer_loop())
                            logger.info("Demo started by WS command (new task created).")
                        else: logger.info("Demo already running."); await ws.send_json({"type": "info", "message": "Demo already active."})
                    
                    elif cmd == "stop_demo":
                        if is_demo_active:
                            logger.info("stop_demo: Demo active. Setting is_demo_active=False, cancelling task.")
                            is_demo_active = False 
                            
                            task_to_cancel = video_producer_task # Store current task
                            video_producer_task = None # Prevent re-use before it's fully done

                            if task_to_cancel and not task_to_cancel.done():
                                task_to_cancel.cancel()
                                try: 
                                    await task_to_cancel # Wait for its finally block to complete
                                    logger.info("stop_demo: video_producer_task finished after cancellation.")
                                except asyncio.CancelledError: 
                                    logger.info("stop_demo: video_producer_task caught CancelledError during await.")
                            
                            logger.info("stop_demo: Broadcasting demo_stopped to all clients.")
                            await broadcast({"type": "demo_stopped", "message": "Demo stopped by user."}) # Send after task is confirmed done
                            GPIO_CONTROLLER.turn_dispenser_off() 
                            dispenser_motor_stop_time_s = 0
                        else: 
                            logger.info("Stop demo cmd, but demo not running."); await ws.send_json({"type": "info", "message": "Demo not active."})
                    
                    elif cmd == "update_settings":
                        app_settings.update(data.get("settings",{})); logger.info(f"Settings updated: {app_settings}")
                        await broadcast({"type":"config_update","settings":app_settings})
                
                except json.JSONDecodeError: 
                    logger.error(f"WS received invalid JSON: {msg.data}"); await ws.send_json({"type":"error","message":"Invalid JSON."})
                except Exception as e: 
                    logger.error(f"WS msg processing error: {e}",exc_info=True); await ws.send_json({"type":"error","message":str(e)})
            elif msg.type == web.WSMsgType.ERROR: logger.error(f"WS connection error {ws.exception()}")
    except Exception as e: logger.error(f"WS handler error for {request.remote}: {e}", exc_info=True)
    finally:
        active_websockets.discard(ws)
        logger.info(f"WS client disconnected: {request.remote}. Remaining: {len(active_websockets)}")
    return ws

async def handle_index_page(request):
    return web.FileResponse(STATIC_FILES_PATH / 'index.html')

async def on_aiohttp_startup(app_obj):
    logger.info("Application server starting up...")
    if not initialize_openvino_models():
        logger.error("OpenVINO init FAILED during startup.")
    app_obj['dispenser_manager_task'] = asyncio.create_task(manage_dispenser())
    logger.info("Dispenser manager task created.")

async def on_aiohttp_shutdown(app_obj):
    global is_demo_active, video_producer_task
    logger.info("Application server shutting down...")
    is_demo_active = False
    
    task_to_await = video_producer_task
    video_producer_task = None

    if task_to_await and not task_to_await.done():
        logger.info("Cancelling video producer task during app shutdown.")
        task_to_await.cancel()
        try: await task_to_await; logger.info("Video producer task finished during app shutdown.")
        except asyncio.CancelledError: logger.info("Video producer task caught CancelledError during app shutdown.")
            
    logger.info(f"Closing {len(active_websockets)} active WebSocket connections.")
    for ws_client in list(active_websockets):
        if not ws_client.closed:
            await ws_client.close(code=WSCloseCode.GOING_AWAY, message='Server shutting down.')
    
    if 'dispenser_manager_task' in app_obj and not app_obj['dispenser_manager_task'].done():
        app_obj['dispenser_manager_task'].cancel()
        try: await app_obj['dispenser_manager_task']
        except asyncio.CancelledError: logger.info("Dispenser manager task cancelled during shutdown.")
    GPIO_CONTROLLER.turn_dispenser_off()
    logger.info("GPIO dispenser off. App cleanup complete.")

def main():
    if not is_el300_present(): logger.info("EL300 HW not detected. Dummy GPIO.")
    else: logger.info("EL300 HW detected.")

    app = web.Application()
    app.on_startup.append(on_aiohttp_startup)
    app.on_shutdown.append(on_aiohttp_shutdown)
    app.router.add_get('/', handle_index_page)
    app.router.add_get(WEBSOCKET_PATH, websocket_handler)
    app.router.add_static('/static/', path=STATIC_FILES_PATH, name='static')
    logger.info(f"Starting server on http://{HTTP_HOST}:{HTTP_PORT}")
    web.run_app(app, host=HTTP_HOST, port=HTTP_PORT)

if __name__ == '__main__':
    main()