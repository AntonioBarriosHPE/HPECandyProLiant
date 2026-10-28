# app.py
# -*- coding: utf-8 -*-
import asyncio
import numpy as np
import base64
import json
import logging
import time
import os
import threading
from pathlib import Path
import platform
import sys
import random
import uuid
from enum import Enum
from collections import deque

import openvino as ov
from aiohttp import web, WSCloseCode
import cv2
from PIL import Image
import io
from contextlib import suppress


# Import the new consolidated hardware controllers
from hardware import MotorControllerInterface, ArduinoMotorController, DummyMotorController

# --- Configuration & Globals ---
HTTP_HOST = os.getenv("CANDY_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("CANDY_HTTP_PORT", "9000"))
WEBSOCKET_PATH = "/ws"
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_IR_BASE_PATH = SCRIPT_DIR / "openvino_models" / "ir" / "public"
STATIC_FILES_PATH = SCRIPT_DIR / "static"

# Performance tuning. Precision/device are auto-selected per host at startup; env vars override.
MODEL_PRECISION_OVERRIDE = os.getenv("CANDY_MODEL_PRECISION")
OPENVINO_DEVICE_OVERRIDE = os.getenv("CANDY_OV_DEVICE")
MODEL_PRECISION = None
OPENVINO_DEVICE = None
FACE_DETECTION_MODEL_XML = None
EMOTION_RECOGNITION_MODEL_XML = None
CAMERA_WIDTH = int(os.getenv("CANDY_CAM_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CANDY_CAM_HEIGHT", "720"))
FRAME_SLEEP_S = float(os.getenv("CANDY_FRAME_SLEEP_S", "0"))

def resolve_model_xml(model_dir_name, file_name):
    candidate = MODEL_IR_BASE_PATH / model_dir_name / MODEL_PRECISION / file_name
    return candidate if candidate.exists() else MODEL_IR_BASE_PATH / model_dir_name / "FP32" / file_name

# Set CANDY_ARDUINO_PORT to a specific port, or leave unset to auto-detect at startup.
ARDUINO_SERIAL_PORT = os.getenv("CANDY_ARDUINO_PORT") or None

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
SETTING_BOUNDS = {
    "happy_time_ms": (0, 300000),
    "motor_on_ms": (10, 30000),
    "spin_pwm_value": (0, 255),
    "cooldown_s": (0, 3600),
    "min_face_area_threshold": (0, 258048),
    "emotion_consensus_window_size": (1, 120),
    "emotion_consensus_required_count": (1, 120),
}
INTEGER_SETTINGS = set(SETTING_BOUNDS) - {"happy_time_ms"}

def validate_settings_update(settings_update):
    """Return a bounded, whitelisted settings update or a user-facing error."""
    if not isinstance(settings_update, dict):
        return None, "Settings must be an object."

    validated = {}
    for key, value in settings_update.items():
        if key not in SETTING_BOUNDS:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            return None, f"Invalid value for {key}."
        minimum, maximum = SETTING_BOUNDS[key]
        if value < minimum or value > maximum:
            return None, f"{key} must be between {minimum} and {maximum}."
        if key in INTEGER_SETTINGS and not float(value).is_integer():
            return None, f"{key} must be a whole number."
        validated[key] = int(value) if key in INTEGER_SETTINGS else float(value)

    window_size = validated.get("emotion_consensus_window_size", app_settings["emotion_consensus_window_size"])
    required_count = validated.get("emotion_consensus_required_count", app_settings["emotion_consensus_required_count"])
    if required_count > window_size:
        return None, "Emotion consensus count cannot exceed its window size."
    return validated, None

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
demo_owner = None
video_producer_task = None
active_websockets = set()
start_stop_lock = asyncio.Lock() # NEW: Lock to prevent race conditions on start/stop commands
dispense_lock = asyncio.Lock()
current_game_state = GameState.IDLE
is_initial_happy_streak = False
initial_happy_streak_start_time_ms = 0
leaderboard_snapshot_b64 = None # Base64 encoded image for the leaderboard
leaderboard_snapshot_taken_this_session = False # Flag if snapshot was taken for current attempt
LEADERBOARD_IMAGE_RETENTION_S = 12 * 60 * 60
leaderboard_snapshot_expires_at_ms = 0
leaderboard_snapshot_expiry_handle = None
leaderboard_attempt_id = None
initial_smile_score_ms = 0
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

# --- Pipeline performance tracking ---
PERF_LOG_INTERVAL_S = 5.0
PERF_STAGES = ("capture", "detect", "emotion", "encode", "broadcast", "total")
perf_samples = {stage: deque(maxlen=300) for stage in PERF_STAGES}

def log_perf_summary():
    if not perf_samples["total"]:
        return

    def avg_p95_ms(values):
        ordered = sorted(values)
        avg_ms = sum(ordered) / len(ordered) * 1000
        p95_ms = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000
        return avg_ms, p95_ms

    avg_total, p95_total = avg_p95_ms(perf_samples["total"])
    stage_parts = []
    for stage in PERF_STAGES[:-1]:
        if perf_samples[stage]:
            avg_stage, p95_stage = avg_p95_ms(perf_samples[stage])
            stage_parts.append(f"{stage} {avg_stage:.1f}/{p95_stage:.1f}ms")

    fps = 1000 / avg_total if avg_total else 0
    logger.info(
        f"PERF fps={fps:.1f} frame={avg_total:.1f}/{p95_total:.1f}ms (avg/p95) | "
        + " | ".join(stage_parts)
    )

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
    global MODEL_PRECISION, OPENVINO_DEVICE, FACE_DETECTION_MODEL_XML, EMOTION_RECOGNITION_MODEL_XML

    models_initialized_successfully = False # Reset flag at start of initialization
    logger.info("Initializing OpenVINO Core and models...")
    try:
        ie_core = ov.Core()
        logger.info(f"Available OpenVINO devices: {ie_core.available_devices}; logical CPUs: {os.cpu_count()}")

        # FP16 only pays off on GPU; CPU-only hosts (e.g. Xeon servers) need INT8 instead.
        has_gpu = "GPU" in ie_core.available_devices
        MODEL_PRECISION = MODEL_PRECISION_OVERRIDE or ("FP16" if has_gpu else "FP16-INT8")
        OPENVINO_DEVICE = OPENVINO_DEVICE_OVERRIDE or ("AUTO:GPU,CPU" if has_gpu else "CPU")
        FACE_DETECTION_MODEL_XML = resolve_model_xml("face-detection-adas-0001", "face-detection-adas-0001.xml")
        EMOTION_RECOGNITION_MODEL_XML = resolve_model_xml("emotions-recognition-retail-0003", "emotions-recognition-retail-0003.xml")
        logger.info(f"Selected precision {MODEL_PRECISION} on device {OPENVINO_DEVICE}.")

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
        # Real-time single-stream workload: LATENCY beats the AUTO plugin's throughput default.
        face_detection_exec_net = ie_core.compile_model(model=face_detection_net, device_name=OPENVINO_DEVICE, config={"PERFORMANCE_HINT": "LATENCY"})
        logger.info(f"Face Detection model compiled. Requested device: {OPENVINO_DEVICE}. Input Shape (NCHW): {face_det_input_shape_nchw}")

        logger.info(f"Loading Emotion Recognition model from: {EMOTION_RECOGNITION_MODEL_XML}")
        emotion_net = ie_core.read_model(model=str(EMOTION_RECOGNITION_MODEL_XML))
        em_input_layer = emotion_net.input(0)
        n_em, c_em, h_em, w_em = map(int, em_input_layer.shape)
        emotion_input_shape_nchw = {'n': n_em, 'c': c_em, 'h': h_em, 'w': w_em}
        emotion_exec_net = ie_core.compile_model(model=emotion_net, device_name=OPENVINO_DEVICE, config={"PERFORMANCE_HINT": "LATENCY"})
        logger.info(f"Emotion Recognition model compiled. Requested device: {OPENVINO_DEVICE}. Input Shape (NCHW): {emotion_input_shape_nchw}")

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

def process_frame_ai(original_frame_hwc):
    """Run CPU-heavy image preparation and OpenVINO inference off the event loop."""
    detect_start_s = time.perf_counter()
    frame_for_face_det_nchw, frame_resized_for_face_det_hwc = preprocess_frame_face_detection(original_frame_hwc)
    detected_faces = detect_faces_from_input_frame(frame_for_face_det_nchw) if frame_for_face_det_nchw is not None else []
    detect_elapsed_s = time.perf_counter() - detect_start_s

    raw_emotion = None
    face_roi_original = None
    emotion_elapsed_s = 0.0
    if detected_faces:
        fx_m, fy_m, fx2_m, fy2_m = detected_faces[0]['rect_model_input_coords']
        if frame_resized_for_face_det_hwc is not None:
            orig_h, orig_w = original_frame_hwc.shape[:2]
            scale_x = orig_w / face_det_input_shape_nchw['w']
            scale_y = orig_h / face_det_input_shape_nchw['h']
            orig_x1, orig_y1 = max(0, int(fx_m * scale_x)), max(0, int(fy_m * scale_y))
            orig_x2, orig_y2 = min(orig_w, int(fx2_m * scale_x)), min(orig_h, int(fy2_m * scale_y))
            if orig_x1 < orig_x2 and orig_y1 < orig_y2:
                face_roi_original = original_frame_hwc[orig_y1:orig_y2, orig_x1:orig_x2]

            if fx_m < fx2_m and fy_m < fy2_m:
                emotion_start_s = time.perf_counter()
                roi_for_emotion = frame_resized_for_face_det_hwc[fy_m:fy2_m, fx_m:fx2_m]
                raw_emotion, _ = recognize_emotion(roi_for_emotion)
                emotion_elapsed_s = time.perf_counter() - emotion_start_s

    return detected_faces, raw_emotion, face_roi_original, detect_elapsed_s, emotion_elapsed_s

async def run_blocking_to_completion(function, *args, cancel_cleanup=None):
    """Run blocking work off-loop, but drain it before propagating cancellation."""
    worker_task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker_task)
    except asyncio.CancelledError:
        try:
            result = await asyncio.shield(worker_task)
            if cancel_cleanup is not None:
                await asyncio.shield(asyncio.to_thread(cancel_cleanup, result))
        except Exception:
            logger.exception("Worker failed while its caller was being cancelled.")
        raise

def open_camera_device():
    """Probe and initialize a webcam without blocking the asyncio event loop."""
    backends_to_try = []
    if platform.system() == "Windows":
        backends_to_try.append(cv2.CAP_DSHOW)
    elif platform.system() == "Darwin":
        backends_to_try.append(cv2.CAP_AVFOUNDATION)
    backends_to_try.append(None)

    for cam_idx in range(5):
        for backend_api in backends_to_try:
            backend_name = "Default" if backend_api is None else "DSHOW" if backend_api == cv2.CAP_DSHOW else "AVFOUNDATION"
            logger.info("Attempting to open Camera Index %s with backend: %s", cam_idx, backend_name)
            cap_instance = None
            try:
                cap_instance = cv2.VideoCapture(cam_idx, backend_api) if backend_api is not None else cv2.VideoCapture(cam_idx)
                if not cap_instance or not cap_instance.isOpened():
                    if cap_instance is not None:
                        cap_instance.release()
                    continue

                logger.info("Camera Index %s (%s) opened successfully.", cam_idx, backend_name)
                if platform.system() != "Darwin":
                    cap_instance.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap_instance.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap_instance.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap_instance.set(cv2.CAP_PROP_FPS, 30)
                cap_instance.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ret_read, _ = cap_instance.read()
                if not ret_read:
                    logger.warning("Camera Index %s (%s) opened, but failed to read a test frame.", cam_idx, backend_name)
                    cap_instance.release()
                    cap_instance = None
                    continue

                # Allow DirectShow auto-exposure to settle before returning the camera.
                for _ in range(10):
                    cap_instance.read()
                    time.sleep(0.05)
                logger.info("Successfully initialized Camera Index %s (backend: %s).", cam_idx, backend_name)
                return cap_instance
            except Exception:
                logger.exception("Exception trying to open Camera Index %s (%s).", cam_idx, backend_name)
            if cap_instance is not None:
                try:
                    cap_instance.release()
                except Exception:
                    logger.debug("Failed to release unsuccessful camera probe.", exc_info=True)

    return None

class LatestFrameCapture:
    """Own camera open/read on one thread and keep only the freshest frame."""
    def __init__(self, capture_factory):
        self.capture_factory = capture_factory
        self.capture = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._release_lock = threading.Lock()
        self._released = False
        self._latest = None
        self._error = None
        self._sequence = 0
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            # DirectShow/COM cameras can fail if opened on one thread and read on another.
            self.capture = self.capture_factory()
        except Exception:
            logger.exception("Camera initialization failed in capture thread.")
            with self._lock:
                self._error = "Camera initialization failed."
            self._ready_event.set()
            return
        self._ready_event.set()
        if self.capture is None:
            with self._lock:
                self._error = "No usable camera could be opened."
            return

        consecutive_failures = 0
        while not self._stop_event.is_set():
            started_s = time.perf_counter()
            try:
                ok, frame = self.capture.read()
            except Exception:
                logger.exception("Camera capture worker failed while reading a frame.")
                with self._lock:
                    self._error = "Camera read raised an exception."
                self._stop_event.set()
                break
            elapsed_s = time.perf_counter() - started_s
            if ok and frame is not None:
                consecutive_failures = 0
                with self._lock:
                    self._sequence += 1
                    self._latest = (self._sequence, frame, elapsed_s)
            else:
                consecutive_failures += 1
                if consecutive_failures >= 50:
                    with self._lock:
                        self._error = "Camera stopped returning frames."
                    self._stop_event.set()
                    break
                self._stop_event.wait(0.01)

    def wait_until_ready(self, timeout_s):
        return self._ready_event.wait(timeout_s)

    def read_latest(self, after_sequence):
        with self._lock:
            if self._latest is None or self._latest[0] == after_sequence:
                return None
            return self._latest

    @property
    def error(self):
        with self._lock:
            return self._error

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=0.25)
        if self._thread.is_alive():
            # Releasing the device may unblock drivers stuck in a blocking read.
            self._release_capture_once()
            self._thread.join(timeout=1.0)
        else:
            self._release_capture_once()
        if self._thread.is_alive():
            logger.error("Camera capture thread did not stop after releasing the camera.")

    def _release_capture_once(self):
        with self._release_lock:
            if not self._released and self.capture is not None:
                self.capture.release()
                self._released = True

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

def clear_leaderboard_snapshot():
    global leaderboard_snapshot_b64, leaderboard_snapshot_expires_at_ms, leaderboard_snapshot_expiry_handle
    leaderboard_snapshot_b64 = None
    leaderboard_snapshot_expires_at_ms = 0
    if leaderboard_snapshot_expiry_handle is not None:
        leaderboard_snapshot_expiry_handle.cancel()
        leaderboard_snapshot_expiry_handle = None


def set_leaderboard_snapshot(snapshot_b64):
    global leaderboard_snapshot_b64, leaderboard_snapshot_expires_at_ms, leaderboard_snapshot_expiry_handle
    clear_leaderboard_snapshot()
    if snapshot_b64:
        leaderboard_snapshot_b64 = snapshot_b64
        leaderboard_snapshot_expires_at_ms = (time.time() + LEADERBOARD_IMAGE_RETENTION_S) * 1000
        leaderboard_snapshot_expiry_handle = asyncio.get_running_loop().call_later(
            LEADERBOARD_IMAGE_RETENTION_S, clear_leaderboard_snapshot
        )


def reset_game_completely():
    global current_game_state, is_initial_happy_streak, initial_happy_streak_start_time_ms
    global leaderboard_snapshot_b64, leaderboard_snapshot_taken_this_session
    global minigame_start_time_overall_ms, minigame_current_score_ms, current_target_emotion
    global current_target_emotion_set_time_ms, current_round_deadline_ms
    global current_hold_duration_ms, current_transition_ms, last_game_over_reason
    global game_over_display_start_time_s, emotion_history, stable_primary_emotion_name
    global leaderboard_attempt_id, initial_smile_score_ms

    logger.info("Resetting game state to IDLE, clearing leaderboard snapshot and related flags.")
    current_game_state = GameState.IDLE
    is_initial_happy_streak = False
    initial_happy_streak_start_time_ms = 0
    clear_leaderboard_snapshot()
    leaderboard_snapshot_taken_this_session = False
    leaderboard_attempt_id = None
    initial_smile_score_ms = 0
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
    global leaderboard_attempt_id, initial_smile_score_ms

    cap = None # Initialize cap to None
    capture_worker = None
    try:
        if not models_initialized_successfully:
            logger.error("OpenVINO Models are not initialized. Video stream producer cannot start.")
            await broadcast({"type": "error", "message": "Server-side AI models are not ready."})
            is_demo_active = False # Ensure demo is marked as inactive
            return # Exit the loop

        reset_game_completely() # Reset game state at the beginning of a new demo session
        current_game_state = GameState.INITIAL_SMILE # Start in INITIAL_SMILE state
        logger.info("Video stream producer loop started. Game reset, current state: INITIAL_SMILE.")

        # The capture thread owns the VideoCapture from open through release.
        capture_worker = LatestFrameCapture(open_camera_device)
        capture_worker.start()
        camera_ready = await run_blocking_to_completion(capture_worker.wait_until_ready, 60.0)

        if not camera_ready or capture_worker.error or capture_worker.capture is None:
            logger.error("Failed to open any webcam. Video stream producer cannot continue.")
            await broadcast({"type": "error", "message": "Webcam error on server. Cannot start demo."})
            is_demo_active = False
            return

        cap = capture_worker.capture
        # Keep only the newest frame: the inference pipeline should never process a
        # growing backlog when camera FPS exceeds the model's processing rate.
        last_frame_sequence = 0

        face_model_h = face_det_input_shape_nchw['h']
        face_model_w = face_det_input_shape_nchw['w']
        first_frame_sent_this_session = False
        last_perf_log_s = time.perf_counter()
        frame_start_s = time.perf_counter()

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

            frame_sample = capture_worker.read_latest(last_frame_sequence)
            if frame_sample is None:
                if capture_worker.error:
                    raise RuntimeError(capture_worker.error)
                await asyncio.sleep(0.005)
                continue
            last_frame_sequence, original_frame_hwc, capture_elapsed_s = frame_sample
            perf_samples["capture"].append(capture_elapsed_s)

            # --- AI Processing ---
            (
                detected_faces,
                raw_detected_emotion_this_frame,
                current_face_roi_for_snapshot_hwc_original_scale,
                detect_elapsed_s,
                emotion_elapsed_s,
            ) = await run_blocking_to_completion(process_frame_ai, original_frame_hwc)
            perf_samples["detect"].append(detect_elapsed_s)
            if emotion_elapsed_s:
                perf_samples["emotion"].append(emotion_elapsed_s)

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
                            qualified_smile_score_ms = app_settings["happy_time_ms"]
                            motor_activated_successfully = False
                            motor_activated_successfully = await activate_motor_for_active_game()

                            if motor_activated_successfully:
                                leaderboard_attempt_id = uuid.uuid4().hex
                                initial_smile_score_ms = qualified_smile_score_ms
                                if current_face_roi_for_snapshot_hwc_original_scale is not None:
                                    set_leaderboard_snapshot(frame_to_base64(current_face_roi_for_snapshot_hwc_original_scale))
                                    snapshot_generated_this_frame = bool(leaderboard_snapshot_b64)
                                candy_dispensed_this_server_cycle = True
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
                "type": "status_update", "demo_active": is_demo_active,
                "current_detected_emotion": stable_primary_emotion_name,
                "is_happy_for_initial_smile": (primary_emotion_name_for_logic == "happy" and current_game_state == GameState.INITIAL_SMILE),
                "in_candy_cooldown": ui_in_candy_cooldown, "candy_eligible_again_time": candy_eligible_again_time_s * 1000,
                "current_time_ms": current_time_ms, "initial_smile_streak_start_time_ms": (initial_happy_streak_start_time_ms if is_initial_happy_streak else 0),
                "leaderboard_snapshot_b64": leaderboard_snapshot_b64, "new_snapshot_taken": snapshot_generated_this_frame,
                "leaderboard_snapshot_expires_at_ms": leaderboard_snapshot_expires_at_ms,
                "leaderboard_attempt_id": leaderboard_attempt_id, "initial_smile_score_ms": initial_smile_score_ms,
                "candy_was_dispensed_this_cycle": candy_dispensed_this_server_cycle, "game_state": current_game_state.name,
                "target_emotion": current_target_emotion, "round_deadline_ms": current_round_deadline_ms,
                "minigame_score_ms": minigame_current_score_ms, "game_over_reason": last_game_over_reason if current_game_state == GameState.GAME_OVER else "",
                "game_over_reset_timer_s": (max(0, GAME_OVER_RESET_DELAY_S - (current_time_s - game_over_display_start_time_s)) if current_game_state == GameState.GAME_OVER else -1),
                "ready": ready_signal_for_this_frame
            }
            broadcast_start_s = time.perf_counter()
            await broadcast(status_payload)
            broadcast_elapsed_s = time.perf_counter() - broadcast_start_s

            # NEW: Encode and broadcast frame as binary data
            encode_start_s = time.perf_counter()
            ok, buf = await run_blocking_to_completion(
                cv2.imencode, '.jpg', output_frame_hwc, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            perf_samples["encode"].append(time.perf_counter() - encode_start_s)
            if ok:
                frame_broadcast_start_s = time.perf_counter()
                await broadcast_frame(memoryview(buf))
                broadcast_elapsed_s += time.perf_counter() - frame_broadcast_start_s

            perf_samples["broadcast"].append(broadcast_elapsed_s)
            perf_samples["total"].append(time.perf_counter() - frame_start_s)
            frame_start_s = time.perf_counter()

            if (time.perf_counter() - last_perf_log_s) >= PERF_LOG_INTERVAL_S:
                log_perf_summary()
                last_perf_log_s = time.perf_counter()

            # sleep(0) still yields to the event loop when no clients are connected.
            await asyncio.sleep(FRAME_SLEEP_S)

    except asyncio.CancelledError:
        logger.info("Video stream producer loop was cancelled (expected).")
    except Exception as e:
        logger.error(f"Unhandled exception in video_stream_producer_loop: {e}", exc_info=True)
        await broadcast({"type": "error", "message": "A server-side error occurred in video processing."})
    finally:
        # Reset state before any I/O so a cleanup failure cannot strand the server as active.
        is_demo_active = False
        disarm_demo()
        reset_game_completely()

        with suppress(asyncio.CancelledError):
            try:
                if capture_worker:
                    await run_blocking_to_completion(capture_worker.stop)
                elif cap:
                    await run_blocking_to_completion(cap.release)
                if cap:
                    # Let Windows finish releasing DirectShow before a restart reopens it.
                    await asyncio.sleep(0.4)
                    logger.info("Webcam released in producer's finally block.")
            except Exception:
                logger.exception("Error while releasing camera resources.")
            finally:
                logger.info("Broadcasting final 'demo_stopped' from producer loop cleanup.")
                try:
                    await broadcast({"type": "demo_stopped", "message": "Demo loop finished."})
                except Exception:
                    logger.exception("Failed to broadcast demo-stopped status.")
                logger.info("Video stream producer loop finished its execution and cleanup.")


def disarm_demo():
    global demo_owner
    demo_owner = None


async def activate_motor_for_active_game():
    global candy_eligible_again_time_s, dispenser_motor_should_stop_time_s
    async with dispense_lock:
        if not is_demo_active or demo_owner not in active_websockets or demo_owner.closed:
            return False
        if time.time() < candy_eligible_again_time_s:
            return False
        if not MOTOR_CONTROLLER or not MOTOR_CONTROLLER.is_connected():
            return False
        try:
            if not await MOTOR_CONTROLLER.activate_motor():
                return False
            dispenser_motor_should_stop_time_s = time.time() + (app_settings["motor_on_ms"] / 1000.0)
            candy_eligible_again_time_s = dispenser_motor_should_stop_time_s + app_settings["cooldown_s"]
            return True
        except Exception:
            logger.exception("Motor activation failed.")
            return False


async def stop_active_demo(only_if_no_clients=False):
    """Serialize demo shutdown so a lost UI cannot leave camera capture running."""
    global is_demo_active, video_producer_task
    async with start_stop_lock:
        if only_if_no_clients and active_websockets:
            return False
        disarm_demo()
        if not is_demo_active:
            return False

        logger.info("Stopping active demo%s.", " after the last client disconnected" if only_if_no_clients else "")
        is_demo_active = False
        task_to_cancel = video_producer_task
        video_producer_task = None
        if task_to_cancel and task_to_cancel is not asyncio.current_task() and not task_to_cancel.done():
            task_to_cancel.cancel()
            try:
                await task_to_cancel
            except asyncio.CancelledError:
                logger.info("Awaited demo producer cancellation as expected.")
            except Exception:
                logger.exception("Demo producer failed during shutdown.")
        return True


async def websocket_handler(request):
    global is_demo_active, video_producer_task, active_websockets, app_settings, emotion_history
    global demo_owner

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    active_websockets.add(ws)
    logger.info(f"WebSocket client connected: {request.remote}. Total clients: {len(active_websockets)}")

    await ws.send_json({"type": "config_update", "settings": app_settings})

    if models_initialized_successfully:
        motor_status_msg = f"Connected ({type(MOTOR_CONTROLLER).__name__})" if MOTOR_CONTROLLER and MOTOR_CONTROLLER.is_connected() else "Not Connected."
        await ws.send_json({"type": "status_update", "ready": True, "demo_active": is_demo_active, "game_state": current_game_state.name, "message": f"Server ready. Motor: {motor_status_msg}"})
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
                        if data.get("play_requested") is not True:
                            await ws.send_json({"type": "error", "message": "Press Play before starting the camera."})
                            continue
                        async with start_stop_lock:
                            if not is_demo_active:
                                if not models_initialized_successfully:
                                    await ws.send_json({"type": "error", "message": "Server AI models not ready."})
                                    continue

                                if video_producer_task and not video_producer_task.done():
                                    video_producer_task.cancel()
                                    try: await video_producer_task
                                    except asyncio.CancelledError: pass

                                # Old producer cleanup sets this false; set true only after it drains.
                                demo_owner = ws
                                is_demo_active = True
                                video_producer_task = asyncio.create_task(video_stream_producer_loop())
                                logger.info("Demo started by WebSocket command.")
                                await ws.send_json({
                                    "type": "demo_starting",
                                    "demo_active": True,
                                    "message": "Opening the camera.",
                                })
                            else:
                                await ws.send_json({"type": "info", "message": "Demo is already active."})

                    elif cmd == "stop_demo":
                        stopped = await stop_active_demo()
                        if not stopped:
                            await ws.send_json({"type": "info", "message": "Demo is not currently active."})

                    elif cmd == "dispense_candy":
                        success = False
                        if ws is demo_owner:
                            success = await activate_motor_for_active_game()
                        await ws.send_json({
                            "type": "dispense_result",
                            "success": success,
                            "current_time_ms": time.time() * 1000,
                            "candy_eligible_again_time": candy_eligible_again_time_s * 1000,
                            "message": "Candy dispensing!" if success else "Cannot dispense. Check that your demo is running, the motor is connected, and cooldown has finished.",
                        })

                    elif cmd == "update_settings":
                        new_settings_payload = data.get("settings", {})
                        if new_settings_payload:
                            validated_settings, validation_error = validate_settings_update(new_settings_payload)
                            if validation_error:
                                await ws.send_json({"type": "error", "message": validation_error})
                                continue
                            app_settings.update(validated_settings)
                            logger.info(f"Settings updated: {app_settings}")
                            if 'emotion_consensus_window_size' in validated_settings:
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
        if demo_owner is ws:
            disarm_demo()
            await stop_active_demo()
        elif not active_websockets:
            await stop_active_demo(only_if_no_clients=True)
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
    serial_connection = await asyncio.to_thread(
        ArduinoMotorController.get_auto_detect_connection, port=ARDUINO_SERIAL_PORT
    )
    MOTOR_CONTROLLER = ArduinoMotorController(
        port=serial_connection[0], baud_rate=serial_connection[1]
    ) if serial_connection else DummyMotorController()

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
    disarm_demo()
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
