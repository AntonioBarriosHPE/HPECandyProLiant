# app.py
# -*- coding: utf-8 -*-
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

import openvino as ov
from aiohttp import web, WSCloseCode
import cv2
from PIL import Image
import io

# Import the new consolidated hardware controllers
from hardware import MotorControllerInterface, ArduinoMotorController, DummyMotorController

# --- Configuration & Globals ---
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 9003
WEBSOCKET_PATH = "/ws"
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_IR_BASE_PATH = SCRIPT_DIR / "openvino_models" / "ir" / "public"
STATIC_FILES_PATH = SCRIPT_DIR / "static"

FACE_DETECTION_MODEL_XML = MODEL_IR_BASE_PATH / "face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
EMOTION_RECOGNITION_MODEL_XML = MODEL_IR_BASE_PATH / "emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml"

# Arduino Configuration - SET YOUR PORT HERE or use None for Dummy
ARDUINO_SERIAL_PORT = "COM5"#ArduinoMotorController.get_auto_detect_com_port()
# ARDUINO_SERIAL_PORT = None # To force dummy controller
ARDUINO_BAUD_RATE = 9600

DETECTION_THRESHOLD = 0.5  # Confidence threshold for face detection
EMOTIONS_CLASSES = ['neutral', 'happy', 'sad', 'surprise', 'angry']

# Default Application Settings (can be overridden by frontend)
app_settings = {
    "happy_time_ms": 5000,  # Milliseconds to hold initial smile for candy
    "motor_on_ms": 3000,  # Milliseconds the motor runs to dispense candy
    "spin_pwm_value": 150,  # Motor PWM value (0-255) for speed control
    "cooldown_s": 5,  # Seconds before candy can be dispensed again
    "min_face_area_threshold": 5000,  # Min face area (pixels w*h) to be considered valid (distance filter)
    "emotion_consensus_window_size": 30, # Number of frames in emotion smoothing window
    "emotion_consensus_required_count": 10 # Number of matching frames required in window for consensus
}

class GameState(Enum):
    IDLE = 0
    INITIAL_SMILE = 1
    MINIGAME_BUFFER_SMILE = 2
    MINIGAME_TRANSITION = 3
    MINIGAME_PLAYING = 4
    GAME_OVER = 5

minigame_settings = {
    "initial_smile_buffer_ms": 3000, # Time to hold smile before minigame starts
    "initial_hold_duration_ms": 5000, # Initial duration for minigame emotion hold
    "initial_transition_ms": 2500, # Initial duration for minigame transition
    "duration_decay_factor": 0.95, # Factor to reduce hold/transition times each round
    "min_hold_duration_ms": 800, # Minimum hold duration
    "min_transition_ms": 700, # Minimum transition duration
    "max_total_game_duration_s": 300, # Max overall duration of a single minigame session
    "allowed_emotions": ['sad', 'happy', 'angry', 'neutral', 'surprise'] # Emotions for minigame rounds
}
GAME_OVER_RESET_DELAY_S = 5.0 # Time to display GAME OVER before auto-resetting

# --- Global State Variables ---
is_demo_active = False
video_producer_task = None
active_websockets = set()
start_stop_lock = asyncio.Lock() # NEW: Lock to prevent race conditions on start/stop commands
current_game_state = GameState.IDLE
is_initial_happy_streak = False
initial_happy_streak_start_time_ms = 0
leaderboard_snapshot_b64 = None # Base64 encoded image for the leaderboard
leaderboard_snapshot_taken_this_session = False # Flag if snapshot was taken for current attempt
candy_eligible_again_time_s = 0 # Timestamp when candy can be dispensed again
dispenser_motor_should_stop_time_s = 0 # Timestamp for motor cycle completion calculation
minigame_start_time_overall_ms = 0 # Start time of the entire minigame session
minigame_current_score_ms = 0 # Accumulated score in the current minigame
current_target_emotion = None # Target emotion for the current minigame round
current_target_emotion_set_time_ms = 0 # Timestamp when current target emotion was set
current_round_deadline_ms = 0 # Deadline for the current minigame action (hold/transition)
current_hold_duration_ms = minigame_settings["initial_hold_duration_ms"]
current_transition_ms = minigame_settings["initial_transition_ms"]
last_game_over_reason = "" # Reason for the last game over
game_over_display_start_time_s = 0 # Timestamp when GAME_OVER state started for display timer
emotion_history = deque(maxlen=app_settings["emotion_consensus_window_size"]) # Initialize with default
stable_primary_emotion_name = None # Smoothed emotion after consensus

# Global motor controller instance
MOTOR_CONTROLLER: MotorControllerInterface = None

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger("hpe_candy_proliant")
# Set log levels for submodules if needed (useful for debugging hardware.py)
logging.getLogger("hpe_candy_proliant.dummy_motor").setLevel(logging.INFO)
logging.getLogger("hpe_candy_proliant.arduino_motor").setLevel(logging.INFO)

# OpenVINO Globals
ie_core = None
face_detection_net = None
face_detection_exec_net = None
emotion_net = None
emotion_exec_net = None
face_det_input_shape_nchw = {} # Stores N, C, H, W for face detection model
emotion_input_shape_nchw = {} # Stores N, C, H, W for emotion recognition model
models_initialized_successfully = False

def initialize_openvino_models():
    global ie_core, face_detection_net, face_detection_exec_net, emotion_net, emotion_exec_net
    global face_det_input_shape_nchw, emotion_input_shape_nchw, models_initialized_successfully

    models_initialized_successfully = False # Reset flag at start of initialization
    logger.info("Initializing OpenVINO Core and models...")
    try:
        ie_core = ov.Core()
        logger.info(f"Available OpenVINO devices: {ie_core.available_devices}")

        if not FACE_DETECTION_MODEL_XML.exists():
            logger.error(f"Face Detection model XML file not found: {FACE_DETECTION_MODEL_XML}")
            return False
        if not EMOTION_RECOGNITION_MODEL_XML.exists():
            logger.error(f"Emotion Recognition model XML file not found: {EMOTION_RECOGNITION_MODEL_XML}")
            return False

        logger.info(f"Loading Face Detection model from: {FACE_DETECTION_MODEL_XML}")
        face_detection_net = ie_core.read_model(model=str(FACE_DETECTION_MODEL_XML))
        fd_input_layer = face_detection_net.input(0)
        n, c, h, w = map(int, fd_input_layer.shape)
        face_det_input_shape_nchw = {'n': n, 'c': c, 'h': h, 'w': w}
        # Compile model for AUTO device selection (CPU, iGPU, etc.)
        face_detection_exec_net = ie_core.compile_model(model=face_detection_net, device_name="AUTO")
        logger.info(f"Face Detection model compiled. Requested device: AUTO. Input Shape (NCHW): {face_det_input_shape_nchw}")

        logger.info(f"Loading Emotion Recognition model from: {EMOTION_RECOGNITION_MODEL_XML}")
        emotion_net = ie_core.read_model(model=str(EMOTION_RECOGNITION_MODEL_XML))
        em_input_layer = emotion_net.input(0)
        n_em, c_em, h_em, w_em = map(int, em_input_layer.shape)
        emotion_input_shape_nchw = {'n': n_em, 'c': c_em, 'h': h_em, 'w': w_em}
        emotion_exec_net = ie_core.compile_model(model=emotion_net, device_name="AUTO")
        logger.info(f"Emotion Recognition model compiled. Requested device: AUTO. Input Shape (NCHW): {emotion_input_shape_nchw}")

        models_initialized_successfully = True
        logger.info("OpenVINO models initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during OpenVINO model initialization: {e}", exc_info=True)
        models_initialized_successfully = False
        return False

def preprocess_frame_face_detection(frame_hwc):
    if not models_initialized_successfully or not face_det_input_shape_nchw:
        logger.warning("Face detection preprocessing skipped: models not ready or input shape unknown.")
        return None, None

    # Get target H, W for the face detection model
    h = face_det_input_shape_nchw['h']
    w = face_det_input_shape_nchw['w']

    resized_frame = cv2.resize(frame_hwc, (w, h))
    # Change data layout from HWC to CHW
    transposed_frame = resized_frame.transpose(2, 0, 1)
    # Add batch dimension (N)
    input_tensor_nchw = np.expand_dims(transposed_frame, 0)
    return input_tensor_nchw, resized_frame # Return both model input and the resized frame used for it

def detect_faces_from_input_frame(frame_for_face_det_nchw):
    if not models_initialized_successfully or frame_for_face_det_nchw is None or \
       not face_detection_net or not face_detection_exec_net:
        logger.debug("Face detection skipped: models/input not ready.")
        return []

    # Perform inference
    results = face_detection_exec_net.infer_new_request({face_detection_net.input(0).get_any_name(): frame_for_face_det_nchw})
    detections = results[face_detection_net.output(0).get_any_name()] # Output shape: [1, 1, N, 7]

    faces = []
    model_input_h = face_det_input_shape_nchw['h']
    model_input_w = face_det_input_shape_nchw['w']

    # Iterate over detections
    for detection in detections[0][0]: # detection = [image_id, label, conf, x_min, y_min, x_max, y_max]
        confidence = float(detection[2])
        if confidence > DETECTION_THRESHOLD:
            # Scale box coordinates to model input dimensions
            x1 = int(detection[3] * model_input_w)
            y1 = int(detection[4] * model_input_h)
            x2 = int(detection[5] * model_input_w)
            y2 = int(detection[6] * model_input_h)

            # Calculate face area based on model input coordinates
            face_width_model = x2 - x1
            face_height_model = y2 - y1
            area_model_coords = face_width_model * face_height_model

            # Apply min_face_area_threshold filter
            if area_model_coords < app_settings['min_face_area_threshold']:
                # logger.debug(f"Face area {area_model_coords} is below threshold {app_settings['min_face_area_threshold']}, ignoring.")
                continue # Skip this face

            faces.append({
                'rect_model_input_coords': (x1, y1, x2, y2), # Coords relative to resized model input frame
                'confidence': confidence,
                'area': area_model_coords # Area relative to resized model input frame
            })

    # Sort faces by area (largest first) - primary face is faces[0] if list is not empty
    faces.sort(key=lambda x: x['area'], reverse=True)
    return faces

def preprocess_face_emotion_recognition(face_roi_hwc):
    if not models_initialized_successfully or not emotion_input_shape_nchw or 'h' not in emotion_input_shape_nchw:
        logger.warning("Emotion recognition preprocessing skipped: models not ready or input shape unknown.")
        return None

    h = emotion_input_shape_nchw['h']
    w = emotion_input_shape_nchw['w']

    resized_face = cv2.resize(face_roi_hwc, (w, h))
    transposed_face = resized_face.transpose(2, 0, 1) # HWC to CHW
    input_tensor_nchw = np.expand_dims(transposed_face, 0) # Add batch dimension
    return input_tensor_nchw

def recognize_emotion(face_roi_for_emotion_hwc):
    if not models_initialized_successfully or not emotion_input_shape_nchw or \
       'h' not in emotion_input_shape_nchw or not emotion_net or not emotion_exec_net:
        logger.debug("Emotion recognition skipped: models/input not ready.")
        return None, 0.0

    if face_roi_for_emotion_hwc is None or face_roi_for_emotion_hwc.size == 0:
        logger.debug("Emotion recognition skipped: empty face ROI.")
        return None, 0.0

    try:
        input_for_emotion_model = preprocess_face_emotion_recognition(face_roi_for_emotion_hwc)
        if input_for_emotion_model is None:
            return None, 0.0

        results = emotion_exec_net.infer_new_request({emotion_net.input(0).get_any_name(): input_for_emotion_model})
        # Output shape is [1, num_classes, 1, 1], need to squeeze
        probabilities = results[emotion_net.output(0).get_any_name()][0].flatten()

        emotion_index = np.argmax(probabilities)
        detected_emotion_label = EMOTIONS_CLASSES[emotion_index]
        confidence = float(probabilities[emotion_index])
        return detected_emotion_label, confidence
    except Exception as e:
        logger.warning(f"Exception during emotion recognition: {e}", exc_info=False)
        return None, 0.0

def frame_to_base64(frame, quality=75):
    """Encodes a frame to Base64. Only used for leaderboard snapshots now."""
    if frame is None:
        logger.warning("frame_to_base64 received a None frame.")
        return None
    try:
        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and buffer is not None and buffer.size > 0:
            return base64.b64encode(buffer).decode('utf-8')
        else:
            logger.warning('cv2.imencode failed or produced empty buffer. Using Pillow fallback.')
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            with io.BytesIO() as byte_stream:
                pil_image.save(byte_stream, format='JPEG', quality=quality)
                return base64.b64encode(byte_stream.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Exception in frame_to_base64: {e}", exc_info=True)
        return None

async def broadcast(message_dict):
    """Broadcasts a JSON message to all connected clients."""
    if active_websockets:
        message_str = json.dumps(message_dict)
        send_tasks = [ws.send_str(message_str) for ws in list(active_websockets) if not ws.closed]
        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)

async def broadcast_frame(raw_jpeg_mv: memoryview):
    jpeg_bytes = raw_jpeg_mv.tobytes()
    for ws in list(active_websockets):
        if ws.closed:
            active_websockets.discard(ws)
            continue

        # --- FIX: find the transport on the underlying request ---
        transport = getattr(ws._req, "transport", None)  # works on aiohttp 3.x/4.x
        if transport and transport.get_write_buffer_size() > 262_144:
            logger.debug("Back‑pressure: dropping frame for a slow client")
            continue
        # ---------------------------------------------------------

        try:
            await ws.send_bytes(jpeg_bytes)
        except (ConnectionResetError, asyncio.CancelledError):
            active_websockets.discard(ws)
        except Exception as e:
            logger.error("Error sending frame: %s", e)
            active_websockets.discard(ws)

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
    leaderboard_snapshot_b64 = None # Clear any existing snapshot
    leaderboard_snapshot_taken_this_session = False
    minigame_start_time_overall_ms = 0
    minigame_current_score_ms = 0
    current_target_emotion = None
    current_target_emotion_set_time_ms = 0
    current_round_deadline_ms = 0
    # Reset durations to initial minigame settings
    current_hold_duration_ms = minigame_settings["initial_hold_duration_ms"]
    current_transition_ms = minigame_settings["initial_transition_ms"]
    last_game_over_reason = ""
    game_over_display_start_time_s = 0
    # Re-initialize emotion_history with potentially new maxlen from app_settings
    emotion_history = deque(maxlen=app_settings["emotion_consensus_window_size"])
    stable_primary_emotion_name = None

def start_new_minigame_round():
    global current_game_state, current_target_emotion, current_target_emotion_set_time_ms
    global current_round_deadline_ms, current_transition_ms

    # Select a new target emotion, try not to repeat the last one
    possible_next_emotions = [e for e in minigame_settings["allowed_emotions"] if e != current_target_emotion]
    if not possible_next_emotions: # If all emotions were somehow same as last, or list is tiny
        possible_next_emotions = minigame_settings["allowed_emotions"]

    current_target_emotion = random.choice(possible_next_emotions)
    current_target_emotion_set_time_ms = int(time.time() * 1000)
    # Deadline for the transition phase
    current_round_deadline_ms = current_target_emotion_set_time_ms + current_transition_ms
    current_game_state = GameState.MINIGAME_TRANSITION
    logger.info(f"Minigame Next Round: Target Emotion '{current_target_emotion}'. Transition Phase Duration: {current_transition_ms}ms.")

async def video_stream_producer_loop():
    global is_demo_active, models_initialized_successfully, current_game_state
    global is_initial_happy_streak, initial_happy_streak_start_time_ms
    global leaderboard_snapshot_b64, leaderboard_snapshot_taken_this_session
    global candy_eligible_again_time_s, dispenser_motor_should_stop_time_s
    global minigame_start_time_overall_ms, minigame_current_score_ms
    global current_target_emotion, current_target_emotion_set_time_ms, current_round_deadline_ms
    global current_hold_duration_ms, current_transition_ms, last_game_over_reason, game_over_display_start_time_s
    global emotion_history, stable_primary_emotion_name, MOTOR_CONTROLLER

    cap = None # Initialize cap to None
    try:
        if not models_initialized_successfully:
            logger.error("OpenVINO Models are not initialized. Video stream producer cannot start.")
            await broadcast({"type": "error", "message": "Server-side AI models are not ready."})
            is_demo_active = False # Ensure demo is marked as inactive
            return # Exit the loop

        reset_game_completely() # Reset game state at the beginning of a new demo session
        current_game_state = GameState.INITIAL_SMILE # Start in INITIAL_SMILE state
        logger.info("Video stream producer loop started. Game reset, current state: INITIAL_SMILE.")

        # Define preferred camera backends based on OS
        backends_to_try = []
        if platform.system() == "Windows": backends_to_try.append(cv2.CAP_DSHOW)
        elif platform.system() == "Darwin": backends_to_try.append(cv2.CAP_AVFOUNDATION)
        backends_to_try.append(None) # Try default backend last

        # Attempt to open webcam
        for cam_idx in range(5): # Try first 5 camera indices
            for backend_api in backends_to_try:
                backend_name = "Default" if backend_api is None else "DSHOW" if backend_api == cv2.CAP_DSHOW else "AVFOUNDATION"
                logger.info(f"Attempting to open Camera Index {cam_idx} with backend: {backend_name}")
                try:
                    cap_instance = cv2.VideoCapture(cam_idx, backend_api) if backend_api is not None else cv2.VideoCapture(cam_idx)
                    if cap_instance and cap_instance.isOpened():
                        logger.info(f"Camera Index {cam_idx} ({backend_name}) opened successfully.")
                        # Set desired properties, check if they were applied
                        if platform.system() != "Darwin": cap_instance.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        cap_instance.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap_instance.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap_instance.set(cv2.CAP_PROP_FPS, 30)

                        ret_read, _ = cap_instance.read() # Test read a frame
                        if ret_read:
                            logger.info(f"Successfully read a test frame from Camera Index {cam_idx} (backend: {backend_name}). Using this camera.")
                            cap = cap_instance
                            break
                        else:
                            logger.warning(f"Opened Camera Index {cam_idx} ({backend_name}), but failed to read a test frame.")
                            if cap_instance: cap_instance.release()
                except Exception as e:
                    logger.error(f"Exception trying to open Camera Index {cam_idx} ({backend_name}): {e}")
            if cap and cap.isOpened(): break

        if not cap or not cap.isOpened():
            logger.error("Failed to open any webcam. Video stream producer cannot continue.")
            await broadcast({"type": "error", "message": "Webcam error on server. Cannot start demo."})
            is_demo_active = False
            return

        face_model_h = face_det_input_shape_nchw['h']
        face_model_w = face_det_input_shape_nchw['w']
        first_frame_sent_this_session = False

        while is_demo_active:
            current_time_ms = int(time.time() * 1000)
            current_time_s = time.time()

            snapshot_generated_this_frame = False
            candy_dispensed_this_server_cycle = False

            # Handle GAME_OVER state timeout for auto-reset
            if current_game_state == GameState.GAME_OVER and game_over_display_start_time_s > 0:
                if (current_time_s - game_over_display_start_time_s) >= GAME_OVER_RESET_DELAY_S:
                    logger.info("Automatic reset after GAME_OVER display period.")
                    reset_game_completely()
                    current_game_state = GameState.INITIAL_SMILE

            ret, original_frame_hwc = cap.read()
            if not ret or original_frame_hwc is None:
                logger.warning("Failed to grab frame from webcam.")
                await asyncio.sleep(0.05)
                continue

            # --- AI Processing ---
            frame_for_face_det_nchw, frame_resized_for_face_det_hwc = preprocess_frame_face_detection(original_frame_hwc)
            detected_faces = detect_faces_from_input_frame(frame_for_face_det_nchw) if frame_for_face_det_nchw is not None else []

            raw_detected_emotion_this_frame = None
            current_face_roi_for_snapshot_hwc_original_scale = None

            if detected_faces:
                primary_face_data = detected_faces[0]
                fx_m, fy_m, fx2_m, fy2_m = primary_face_data['rect_model_input_coords']
                if original_frame_hwc is not None and frame_resized_for_face_det_hwc is not None:
                    orig_h, orig_w = original_frame_hwc.shape[:2]
                    scale_x_to_orig, scale_y_to_orig = orig_w / face_model_w, orig_h / face_model_h
                    orig_x1, orig_y1 = max(0, int(fx_m * scale_x_to_orig)), max(0, int(fy_m * scale_y_to_orig))
                    orig_x2, orig_y2 = min(orig_w, int(fx2_m * scale_x_to_orig)), min(orig_h, int(fy2_m * scale_y_to_orig))
                    if orig_x1 < orig_x2 and orig_y1 < orig_y2:
                         current_face_roi_for_snapshot_hwc_original_scale = original_frame_hwc[orig_y1:orig_y2, orig_x1:orig_x2]

                if fx_m < fx2_m and fy_m < fy2_m and frame_resized_for_face_det_hwc is not None:
                    roi_for_emotion_resized = frame_resized_for_face_det_hwc[fy_m:fy2_m, fx_m:fx2_m]
                    raw_detected_emotion_this_frame, _ = recognize_emotion(roi_for_emotion_resized)

            # --- Emotion Smoothing Logic ---
            emotion_history.append(raw_detected_emotion_this_frame) if raw_detected_emotion_this_frame else emotion_history.clear() if not detected_faces else None

            if len(emotion_history) == app_settings["emotion_consensus_window_size"]:
                counts = {em: emotion_history.count(em) for em in set(emotion_history) if em is not None}
                if counts and counts[max(counts, key=counts.get)] >= app_settings["emotion_consensus_required_count"]:
                    stable_primary_emotion_name = max(counts, key=counts.get)
            elif raw_detected_emotion_this_frame: stable_primary_emotion_name = raw_detected_emotion_this_frame
            elif not detected_faces: stable_primary_emotion_name = None

            primary_emotion_name_for_logic = stable_primary_emotion_name
            ui_in_candy_cooldown = current_time_s < candy_eligible_again_time_s

            # --- Game State Machine Logic ---
            if current_game_state == GameState.INITIAL_SMILE:
                if primary_emotion_name_for_logic == "happy":
                    if not is_initial_happy_streak:
                        is_initial_happy_streak = True
                        initial_happy_streak_start_time_ms = current_time_ms
                        leaderboard_snapshot_taken_this_session = False

                    smile_duration_ms = current_time_ms - initial_happy_streak_start_time_ms
                    if not ui_in_candy_cooldown and smile_duration_ms >= app_settings["happy_time_ms"]:
                        if not leaderboard_snapshot_taken_this_session:
                            if current_face_roi_for_snapshot_hwc_original_scale is not None:
                                temp_b64 = frame_to_base64(current_face_roi_for_snapshot_hwc_original_scale)
                                if temp_b64:
                                    leaderboard_snapshot_b64 = temp_b64
                                    snapshot_generated_this_frame = True

                            motor_activated_successfully = False
                            if MOTOR_CONTROLLER and MOTOR_CONTROLLER.is_connected(): #and not MOTOR_CONTROLLER.is_busy(): # Note: Crashing here , why did you guys add another conditionall here?
                                logger.info("Motor is available. Creating fire-and-forget task to activate.")
                                asyncio.create_task(MOTOR_CONTROLLER.activate_motor())
                                motor_activated_successfully = True

                            if motor_activated_successfully:
                                candy_dispensed_this_server_cycle = True
                                dispenser_motor_should_stop_time_s = current_time_s + (app_settings["motor_on_ms"] / 1000.0)
                                candy_eligible_again_time_s = dispenser_motor_should_stop_time_s + app_settings["cooldown_s"]
                                leaderboard_snapshot_taken_this_session = True
                                current_game_state = GameState.MINIGAME_BUFFER_SMILE
                                current_target_emotion = "happy"
                                minigame_start_time_overall_ms = current_time_ms
                                current_target_emotion_set_time_ms = current_time_ms
                                current_round_deadline_ms = current_time_ms + minigame_settings["initial_smile_buffer_ms"]
                else:
                    if is_initial_happy_streak: is_initial_happy_streak = False

            elif current_game_state == GameState.MINIGAME_BUFFER_SMILE:
                if primary_emotion_name_for_logic == "happy":
                    if current_time_ms >= current_round_deadline_ms:
                        minigame_current_score_ms += (current_time_ms - current_target_emotion_set_time_ms)
                        start_new_minigame_round()
                else:
                    last_game_over_reason = "Smile was lost during minigame start!"
                    current_game_state = GameState.GAME_OVER
                    game_over_display_start_time_s = current_time_s

            elif current_game_state == GameState.MINIGAME_TRANSITION:
                if current_time_ms >= current_round_deadline_ms:
                    current_game_state = GameState.MINIGAME_PLAYING
                    current_target_emotion_set_time_ms = current_time_ms
                    current_round_deadline_ms = current_time_ms + current_hold_duration_ms

            elif current_game_state == GameState.MINIGAME_PLAYING:
                if (current_time_ms - minigame_start_time_overall_ms) >= (minigame_settings["max_total_game_duration_s"] * 1000):
                    last_game_over_reason = "Max game time reached!"
                    current_game_state = GameState.GAME_OVER
                    game_over_display_start_time_s = current_time_s
                elif primary_emotion_name_for_logic == current_target_emotion:
                    if current_time_ms >= current_round_deadline_ms:
                        minigame_current_score_ms += (current_time_ms - current_target_emotion_set_time_ms)
                        current_hold_duration_ms = max(minigame_settings["min_hold_duration_ms"], int(current_hold_duration_ms * minigame_settings["duration_decay_factor"]))
                        current_transition_ms = max(minigame_settings["min_transition_ms"], int(current_transition_ms * minigame_settings["duration_decay_factor"]))
                        start_new_minigame_round()
                else:
                    if current_time_ms > (current_target_emotion_set_time_ms + 500):
                        last_game_over_reason = f"Needed '{current_target_emotion}', got '{primary_emotion_name_for_logic or 'None'}'."
                        current_game_state = GameState.GAME_OVER
                        game_over_display_start_time_s = current_time_s

            # --- Frame Annotation and Broadcasting ---
            output_frame_hwc = original_frame_hwc.copy()
            if detected_faces:
                output_h, output_w = output_frame_hwc.shape[:2]
                scale_x_draw, scale_y_draw = output_w / face_model_w, output_h / face_model_h
                fx_m, fy_m, fx2_m, fy2_m = detected_faces[0]['rect_model_input_coords']
                dx1, dy1, dx2, dy2 = int(fx_m * scale_x_draw), int(fy_m * scale_y_draw), int(fx2_m * scale_x_draw), int(fy2_m * scale_y_draw)
                cv2.rectangle(output_frame_hwc, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                if stable_primary_emotion_name:
                    cv2.putText(output_frame_hwc, stable_primary_emotion_name.capitalize(), (dx1, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Broadcast status and video frame ---
            ready_signal_for_this_frame = not first_frame_sent_this_session
            if ready_signal_for_this_frame: first_frame_sent_this_session = True

            status_payload = {
                "type": "status_update", "current_detected_emotion": stable_primary_emotion_name,
                "is_happy_for_initial_smile": (primary_emotion_name_for_logic == "happy" and current_game_state == GameState.INITIAL_SMILE),
                "in_candy_cooldown": ui_in_candy_cooldown, "candy_eligible_again_time": candy_eligible_again_time_s * 1000,
                "current_time_ms": current_time_ms, "initial_smile_streak_start_time_ms": (initial_happy_streak_start_time_ms if is_initial_happy_streak else 0),
                "leaderboard_snapshot_b64": leaderboard_snapshot_b64, "new_snapshot_taken": snapshot_generated_this_frame,
                "candy_was_dispensed_this_cycle": candy_dispensed_this_server_cycle, "game_state": current_game_state.name,
                "target_emotion": current_target_emotion, "round_deadline_ms": current_round_deadline_ms,
                "minigame_score_ms": minigame_current_score_ms, "game_over_reason": last_game_over_reason if current_game_state == GameState.GAME_OVER else "",
                "game_over_reset_timer_s": (max(0, GAME_OVER_RESET_DELAY_S - (current_time_s - game_over_display_start_time_s)) if current_game_state == GameState.GAME_OVER else -1),
                "ready": ready_signal_for_this_frame
            }
            await broadcast(status_payload)

            # NEW: Encode and broadcast frame as binary data
            ok, buf = cv2.imencode('.jpg', output_frame_hwc, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                await broadcast_frame(memoryview(buf))

            await asyncio.sleep(0.015)

    except asyncio.CancelledError:
        logger.info("Video stream producer loop was cancelled (expected).")
    except Exception as e:
        logger.error(f"Unhandled exception in video_stream_producer_loop: {e}", exc_info=True)
        await broadcast({"type": "error", "message": "A server-side error occurred in video processing."})
    finally:
        # NEW: Centralized, cancellation-safe cleanup
        with suppress(asyncio.CancelledError):
            if cap:
                cap.release()
                logger.info("Webcam released in producer's finally block.")

            reset_game_completely()

            logger.info("Broadcasting final 'demo_stopped' from producer loop cleanup.")
            await broadcast({"type": "demo_stopped", "message": "Demo loop finished."})

            logger.info("Video stream producer loop finished its execution and cleanup.")


async def websocket_handler(request):
    global is_demo_active, video_producer_task, active_websockets, app_settings, emotion_history

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    active_websockets.add(ws)
    logger.info(f"WebSocket client connected: {request.remote}. Total clients: {len(active_websockets)}")

    await ws.send_json({"type": "config_update", "settings": app_settings})

    if models_initialized_successfully:
        motor_status_msg = f"Connected ({type(MOTOR_CONTROLLER).__name__})" if MOTOR_CONTROLLER and MOTOR_CONTROLLER.is_connected() else "Not Connected."
        await ws.send_json({"type": "status_update", "ready": True, "game_state": current_game_state.name, "message": f"Server ready. Motor: {motor_status_msg}"})
    else:
        await ws.send_json({"type": "error", "message": "Server AI models are not initialized."})

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    cmd = data.get("command")
                    logger.info(f"WebSocket received command: '{cmd}'")

                    if cmd == "start_demo":
                        async with start_stop_lock:
                            if not is_demo_active:
                                if not models_initialized_successfully:
                                    await ws.send_json({"type": "error", "message": "Server AI models not ready."})
                                    continue

                                is_demo_active = True
                                if video_producer_task and not video_producer_task.done():
                                    video_producer_task.cancel()
                                    try: await video_producer_task
                                    except asyncio.CancelledError: pass

                                video_producer_task = asyncio.create_task(video_stream_producer_loop())
                                logger.info("Demo started by WebSocket command.")
                            else:
                                await ws.send_json({"type": "info", "message": "Demo is already active."})

                    elif cmd == "stop_demo":
                        async with start_stop_lock:
                            if is_demo_active:
                                logger.info("Command 'stop_demo': Setting is_demo_active=False and cancelling task.")
                                is_demo_active = False # Signal loop to stop
                                task_to_cancel = video_producer_task
                                video_producer_task = None
                                if task_to_cancel and not task_to_cancel.done():
                                    task_to_cancel.cancel()
                                    try: await task_to_cancel
                                    except asyncio.CancelledError: logger.info("Awaited task caught CancelledError as expected.")
                                # Cleanup and 'demo_stopped' broadcast is now handled by the producer's finally block
                            else:
                                await ws.send_json({"type": "info", "message": "Demo is not currently active."})

                    elif cmd == "update_settings":
                        new_settings_payload = data.get("settings", {})
                        if new_settings_payload:
                            app_settings.update(new_settings_payload)
                            logger.info(f"Settings updated: {app_settings}")
                            if 'emotion_consensus_window_size' in new_settings_payload:
                                emotion_history = deque(list(emotion_history), maxlen=app_settings["emotion_consensus_window_size"])

                            if MOTOR_CONTROLLER and MOTOR_CONTROLLER.is_connected():
                                await MOTOR_CONTROLLER.configure_spin_pwm(int(app_settings["spin_pwm_value"]))
                                await MOTOR_CONTROLLER.configure_run_duration(int(app_settings["motor_on_ms"]))

                            await broadcast({"type": "config_update", "settings": app_settings})

                except json.JSONDecodeError: logger.error(f"Invalid JSON from {request.remote}: {msg.data}")
                except Exception as e: logger.error(f"Error processing message: {e}", exc_info=True)

            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"WebSocket connection error for {request.remote}: {ws.exception()}")

    finally:
        active_websockets.discard(ws)
        logger.info(f"WebSocket client disconnected: {request.remote}. Total clients: {len(active_websockets)}")
    return ws

async def handle_index_page(request):
    return web.FileResponse(STATIC_FILES_PATH / 'index.html')

async def on_aiohttp_startup(app_obj):
    global MOTOR_CONTROLLER, app_settings, emotion_history
    logger.info("Application server starting up...")

    if not initialize_openvino_models():
        logger.error("CRITICAL: OpenVINO model initialization FAILED.")

    emotion_history = deque(maxlen=app_settings["emotion_consensus_window_size"])

    logger.info("Initializing Motor Controller...")
    MOTOR_CONTROLLER = ArduinoMotorController(port=ARDUINO_SERIAL_PORT, baud_rate=ARDUINO_BAUD_RATE) if ARDUINO_SERIAL_PORT else DummyMotorController()

    if not await MOTOR_CONTROLLER.connect():
        logger.error(f"Failed to connect {type(MOTOR_CONTROLLER).__name__}. Falling back to DummyMotorController.")
        if not isinstance(MOTOR_CONTROLLER, DummyMotorController):
            MOTOR_CONTROLLER = DummyMotorController()
            await MOTOR_CONTROLLER.connect()

    await MOTOR_CONTROLLER.configure_spin_pwm(app_settings.get("spin_pwm_value"))
    await MOTOR_CONTROLLER.configure_run_duration(app_settings.get("motor_on_ms"))

async def on_aiohttp_shutdown(app_obj):
    global is_demo_active, video_producer_task, MOTOR_CONTROLLER
    logger.info("Application server shutting down...")
    is_demo_active = False

    if video_producer_task and not video_producer_task.done():
        video_producer_task.cancel()
        try: await video_producer_task
        except asyncio.CancelledError: pass

    for ws in list(active_websockets):
        await ws.close(code=WSCloseCode.GOING_AWAY, message=b'Server shutdown.')

    if MOTOR_CONTROLLER: MOTOR_CONTROLLER.disconnect()
    logger.info("Shutdown cleanup complete.")

def main():
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