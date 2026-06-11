"""
FallGuard AI - Multi-Stage Fall Detection Engine
Compatible with MediaPipe 0.10.x (Tasks API - no mp.solutions)
"""

import cv2
import numpy as np
import time
import logging
import os
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)

# ── MediaPipe Tasks API (0.10.x) ─────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components import containers as mp_containers

# Pose landmark indices (MediaPipe Pose)
NOSE        = 0
L_SHOULDER  = 11; R_SHOULDER = 12
L_HIP       = 23; R_HIP      = 24
L_KNEE      = 25; R_KNEE     = 26
L_ANKLE     = 27; R_ANKLE    = 28

HISTORY_LEN = 30

# Pose connections for drawing
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32)
]

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")


def _download_model():
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading MediaPipe pose model (~5MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info("Model downloaded.")
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return False
    return True


@dataclass
class PersonState:
    center_history:    deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    height_history:    deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    angle_history:     deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    inactivity_start:  float = 0.0
    is_inactive:       bool  = False
    prev_center:       Optional[float] = None
    prev_height:       Optional[float] = None


@dataclass
class DetectionResult:
    activity:        str   = 'unknown'
    is_fall:         bool  = False
    confidence:      float = 0.0
    conf_posture:    float = 0.0
    conf_velocity:   float = 0.0
    conf_height:     float = 0.0
    conf_inactivity: float = 0.0
    conf_context:    float = 0.0
    body_angle:      float = 90.0
    velocity:        float = 0.0
    detected_objects: List[str] = field(default_factory=list)
    context_surface: str   = 'floor'
    annotated_frame: Optional[np.ndarray] = None


class FallDetector:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.confidence_threshold    = self.config.get('FALL_CONFIDENCE_THRESHOLD',    0.85)
        self.inactivity_threshold    = self.config.get('INACTIVITY_THRESHOLD_SECONDS', 5)
        self.velocity_threshold      = self.config.get('VELOCITY_THRESHOLD',           0.15)
        self.overlap_threshold       = self.config.get('OVERLAP_THRESHOLD',            0.5)
        self.w_posture    = self.config.get('WEIGHT_POSTURE',        0.30)
        self.w_velocity   = self.config.get('WEIGHT_VELOCITY',       0.25)
        self.w_height     = self.config.get('WEIGHT_HEIGHT_CHANGE',  0.20)
        self.w_inactivity = self.config.get('WEIGHT_INACTIVITY',     0.15)
        self.w_context    = self.config.get('WEIGHT_OBJECT_CONTEXT', 0.10)

        self.pose_landmarker = None
        self._init_pose()

        # YOLO optional
        self.yolo = None
        self._init_yolo()

        # Fallback motion detector
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False)

        self.person_state = PersonState()
        self.fps_tracker  = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.frame_count = 0
        logger.info("FallDetector initialised (MediaPipe Tasks API)")

    # ── Init ─────────────────────────────────────────────────

    def _init_pose(self):
        """Initialise MediaPipe PoseLandmarker (Tasks API)."""
        try:
            if not _download_model():
                raise RuntimeError("Model file unavailable")
            base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
            logger.info("PoseLandmarker ready")
        except Exception as e:
            logger.warning(f"PoseLandmarker init failed: {e}. Angle-only mode.")
            self.pose_landmarker = None

    def _init_yolo(self):
        try:
            from ultralytics import YOLO
            self.yolo = YOLO('yolov8n.pt')
            logger.info("YOLOv8n loaded")
        except Exception:
            logger.info("YOLO unavailable — pose-only mode")

    # ── Public API ────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps_tracker.append(1.0 / dt)

        h, w = frame.shape[:2]
        result = DetectionResult()

        # Stage 1 — object detection
        context_boxes: Dict[str, List] = {}
        person_box = None
        if self.yolo:
            person_box, context_boxes = self._run_yolo(frame)
        else:
            person_box = self._detect_person_fallback(frame)

        # Stage 2 — pose estimation
        kp = self._run_pose(frame, w, h)
        if kp is None:
            result.activity = 'no_person'
            result.annotated_frame = frame.copy()
            return result

        # Stage 3 — fall analysis
        body_angle      = self._compute_body_angle(kp)
        center_y_norm   = kp['center'][1] / h
        body_height_norm= self._compute_body_height(kp, h)

        ps = self.person_state
        velocity = self._compute_velocity(center_y_norm, ps)

        ps.center_history.append(center_y_norm)
        ps.height_history.append(body_height_norm)
        ps.angle_history.append(body_angle)
        ps.prev_center = center_y_norm
        ps.prev_height = body_height_norm

        c_posture    = self._score_posture(body_angle)
        c_velocity   = self._score_velocity(velocity)
        c_height     = self._score_height_change(ps)
        c_inactivity = self._score_inactivity(velocity, now, ps)

        # Stage 4 — context
        if person_box is None:
            xs = [v[0] for v in kp.values()]
            ys = [v[1] for v in kp.values()]
            person_box = (min(xs), min(ys), max(xs), max(ys))

        surface, c_context = self._score_context(person_box, context_boxes)
        result.context_surface  = surface
        result.detected_objects = list(context_boxes.keys())

        conf = (self.w_posture    * c_posture +
                self.w_velocity   * c_velocity +
                self.w_height     * c_height +
                self.w_inactivity * c_inactivity +
                self.w_context    * c_context)
        conf = max(0.0, min(1.0, conf))

        result.body_angle      = round(body_angle, 1)
        result.velocity        = round(velocity, 4)
        result.conf_posture    = round(c_posture, 3)
        result.conf_velocity   = round(c_velocity, 3)
        result.conf_height     = round(c_height, 3)
        result.conf_inactivity = round(c_inactivity, 3)
        result.conf_context    = round(c_context, 3)
        result.confidence      = round(conf, 3)

        result.activity, result.is_fall = self._classify_activity(
            body_angle, velocity, surface, conf, c_posture)

        result.annotated_frame = self._annotate(
            frame.copy(), kp, result, person_box, context_boxes, w, h)
        return result

    @property
    def fps(self) -> float:
        if not self.fps_tracker:
            return 0.0
        return round(sum(self.fps_tracker) / len(self.fps_tracker), 1)

    def release(self):
        if self.pose_landmarker:
            self.pose_landmarker.close()

    # ── Internal ──────────────────────────────────────────────

    def _run_pose(self, frame: np.ndarray, w: int, h: int) -> Optional[dict]:
        """Run PoseLandmarker and return keypoints dict, or None."""
        if self.pose_landmarker is None:
            return self._fallback_keypoints(frame, w, h)
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = self.pose_landmarker.detect(mp_image)
            if not detection.pose_landmarks or len(detection.pose_landmarks) == 0:
                return None
            lms = detection.pose_landmarks[0]  # first person

            def px(idx):
                lm = lms[idx]
                return (int(lm.x * w), int(lm.y * h))

            ls = px(L_SHOULDER); rs = px(R_SHOULDER)
            lh = px(L_HIP);      rh = px(R_HIP)
            la = px(L_ANKLE);    ra = px(R_ANKLE)

            return {
                'l_shoulder': ls, 'r_shoulder': rs,
                'l_hip': lh, 'r_hip': rh,
                'l_ankle': la, 'r_ankle': ra,
                'shoulder_mid': ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2),
                'hip_mid':      ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2),
                'ankle_mid':    ((la[0]+ra[0])//2, (la[1]+ra[1])//2),
                'center':       ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2),
                'nose':         px(NOSE),
                '_landmarks':   lms
            }
        except Exception as e:
            logger.debug(f"Pose error: {e}")
            return None

    def _fallback_keypoints(self, frame, w, h):
        """When pose model unavailable — return dummy centred kp so pipeline continues."""
        mid = (w//2, h//2)
        top = (w//2, h//4)
        bot = (w//2, h*3//4)
        return {
            'l_shoulder': top, 'r_shoulder': top,
            'l_hip': mid, 'r_hip': mid,
            'l_ankle': bot, 'r_ankle': bot,
            'shoulder_mid': top, 'hip_mid': mid,
            'ankle_mid': bot, 'center': mid, 'nose': top,
            '_landmarks': None
        }

    def _run_yolo(self, frame):
        try:
            res = self.yolo(frame, verbose=False, conf=0.35)[0]
            COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
                    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
                    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
                    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
                    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
                    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
                    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
                    'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
                    'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock','vase','scissors',
                    'teddy bear','hair drier','toothbrush']
            person_box = None; best_conf = 0.0; context_boxes = {}
            for box in res.boxes:
                cls = int(box.cls[0]); cf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                if cls == 0 and cf > best_conf:
                    best_conf = cf; person_box = (x1,y1,x2,y2)
                label = COCO[cls] if cls < len(COCO) else str(cls)
                if label in ('bed','couch','chair'):
                    context_boxes.setdefault(label, []).append((x1,y1,x2,y2))
            return person_box, context_boxes
        except Exception as e:
            logger.debug(f"YOLO error: {e}")
            return None, {}

    def _detect_person_fallback(self, frame):
        fg = self.bg_sub.apply(frame)
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 800:
                x,y,bw,bh = cv2.boundingRect(c)
                return (x,y,x+bw,y+bh)
        return None

    def _compute_body_angle(self, kp):
        sm, am = kp['shoulder_mid'], kp['ankle_mid']
        dx = am[0]-sm[0]; dy = am[1]-sm[1]
        return abs(np.degrees(np.arctan2(abs(dy), abs(dx)+1e-6)))

    def _compute_body_height(self, kp, h):
        return abs(kp['ankle_mid'][1] - kp['shoulder_mid'][1]) / max(h,1)

    def _compute_velocity(self, cy, ps):
        if ps.prev_center is None: return 0.0
        return max(0.0, cy - ps.prev_center)

    def _score_posture(self, angle):
        if angle < 20: return 1.0
        if angle < 35: return 0.85
        if angle < 50: return 0.5
        if angle < 65: return 0.2
        return 0.0

    def _score_velocity(self, v):
        if v > 0.12: return 1.0
        if v > 0.07: return 0.7
        if v > 0.03: return 0.3
        return 0.0

    def _score_height_change(self, ps):
        if len(ps.height_history) < 10: return 0.0
        recent   = list(ps.height_history)
        baseline = np.mean(recent[:10])
        current  = np.mean(recent[-3:]) if len(recent)>=3 else recent[-1]
        if baseline < 0.01: return 0.0
        drop = (baseline - current) / baseline
        if drop > 0.40: return 1.0
        if drop > 0.25: return 0.6
        if drop > 0.10: return 0.2
        return 0.0

    def _score_inactivity(self, velocity, now, ps):
        if velocity < 0.01:
            if not ps.is_inactive:
                ps.inactivity_start = now
                ps.is_inactive = True
            elapsed = now - ps.inactivity_start
            return min(1.0, elapsed / max(self.inactivity_threshold, 1))
        else:
            ps.is_inactive = False
            ps.inactivity_start = 0.0
            return 0.0

    def _score_context(self, person_box, context_boxes):
        if not context_boxes:
            return 'floor', 0.5
        px1,py1,px2,py2 = person_box
        p_area = max(1,(px2-px1)*(py2-py1))
        for surface in ('bed','couch'):
            for (ox1,oy1,ox2,oy2) in context_boxes.get(surface,[]):
                ix1=max(px1,ox1);iy1=max(py1,oy1)
                ix2=min(px2,ox2);iy2=min(py2,oy2)
                if ix2>ix1 and iy2>iy1:
                    overlap = (ix2-ix1)*(iy2-iy1)/p_area
                    if overlap >= self.overlap_threshold:
                        return surface, 0.0
        return 'floor', 0.8

    def _classify_activity(self, angle, velocity, surface, conf, c_posture):
        if surface == 'bed':   return 'sleeping',   False
        if surface == 'couch': return 'lying_down',  False
        if angle > 65:
            return ('walking' if velocity > 0.04 else 'standing'), False
        if angle > 45: return 'bending', False
        if angle > 30: return 'sitting', False
        # horizontal
        if conf >= self.confidence_threshold: return 'fallen', True
        if c_posture > 0.5 and velocity < 0.02: return 'lying', False
        return ('falling' if conf > 0.6 else 'unknown'), conf > 0.6

    def _annotate(self, frame, kp, result, person_box, context_boxes, w, h):
        # Draw skeleton
        lms = kp.get('_landmarks')
        if lms:
            for a,b in POSE_CONNECTIONS:
                try:
                    pa = (int(lms[a].x*w), int(lms[a].y*h))
                    pb = (int(lms[b].x*w), int(lms[b].y*h))
                    cv2.line(frame, pa, pb, (0,200,255), 2)
                except: pass
            for lm in lms:
                try:
                    cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),4,(255,255,255),-1)
                except: pass

        # Person box
        if person_box:
            color = (0,0,255) if result.is_fall else (0,255,100)
            cv2.rectangle(frame, person_box[:2], person_box[2:], color, 2)

        # Context boxes
        for label, boxes in context_boxes.items():
            for box in boxes:
                cv2.rectangle(frame, box[:2], box[2:], (255,200,0), 1)
                cv2.putText(frame, label, (box[0], box[1]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,0), 1)

        # HUD
        overlay = frame.copy()
        cv2.rectangle(overlay,(0,0),(260,180),(0,0,0),-1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        activity_color = (0,0,255) if result.is_fall else (0,255,150)
        lines = [
            (f"Activity: {result.activity.upper()}", activity_color),
            (f"Confidence: {result.confidence:.0%}", (200,200,200)),
            (f"Angle: {result.body_angle:.1f} deg",  (200,200,200)),
            (f"Velocity: {result.velocity:.3f}",     (200,200,200)),
            (f"Surface: {result.context_surface}",   (200,200,200)),
            (f"FPS: {self.fps:.1f}",                 (150,150,150)),
        ]
        for i,(text,color) in enumerate(lines):
            cv2.putText(frame, text, (8, 22+i*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        if result.is_fall:
            cv2.rectangle(frame,(0,0),(w,h),(0,0,255),4)
            cv2.putText(frame,"!! FALL DETECTED !!",(w//2-140,h-20),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0,(0,0,255),2,cv2.LINE_AA)
        return frame
