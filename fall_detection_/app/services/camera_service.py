"""
FallGuard AI - Camera Service
Manages webcam stream, frame processing, and SocketIO broadcasting.
"""

import cv2
import threading
import time
import base64
import logging
import os
import numpy as np
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class CameraService:
    """
    Manages webcam capture + fall detection loop.
    Runs in a background thread and emits results via SocketIO.
    """

    def __init__(self, socketio, detector, app_config: dict):
        self.socketio = socketio
        self.detector = detector
        self.config = app_config

        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index = app_config.get('camera_index', 0)
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.alert_cooldown = app_config.get('ALERT_COOLDOWN_SECONDS', 30)
        self.last_alert_time = 0.0
        self.frame_count = 0
        self.start_time = None
        self.event_image_dir = app_config.get('EVENT_IMAGE_DIR', 'static/events')
        os.makedirs(self.event_image_dir, exist_ok=True)

        # Store Flask app reference so background thread can push an app context
        self._flask_app = None

    # ── Public API ────────────────────────────────────────────────

    def start(self, camera_index: int = None):
        if self.is_running:
            return {'success': False, 'message': 'Camera already running'}
        if camera_index is not None:
            self.camera_index = camera_index

        # Grab the current Flask app so the capture thread can use it
        try:
            from flask import current_app
            self._flask_app = current_app._get_current_object()
        except RuntimeError:
            self._flask_app = None
            logger.warning("No Flask app context found — DB saves will not work.")

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return {'success': False, 'message': f'Cannot open camera {self.camera_index}'}

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.is_running = True
        self.start_time = datetime.utcnow()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Camera {self.camera_index} started")
        return {'success': True, 'message': 'Camera started'}

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Camera stopped")
        return {'success': True, 'message': 'Camera stopped'}

    def switch_camera(self, index: int):
        self.stop()
        time.sleep(0.5)
        return self.start(index)

    def get_status(self) -> dict:
        uptime = 0
        if self.start_time:
            uptime = int((datetime.utcnow() - self.start_time).total_seconds())
        return {
            'is_running': self.is_running,
            'camera_index': self.camera_index,
            'fps': self.detector.fps if self.detector else 0,
            'frame_count': self.frame_count,
            'uptime_seconds': uptime
        }

    # ── Capture loop ──────────────────────────────────────────────

    def _capture_loop(self):
        logger.info("Capture loop started")
        while self.is_running:
            with self._lock:
                if self.cap is None or not self.cap.isOpened():
                    break
                ret, frame = self.cap.read()

            if not ret:
                logger.warning("Frame read failed, retrying...")
                time.sleep(0.05)
                continue

            self.frame_count += 1
            try:
                result = self.detector.process_frame(frame)
                self._broadcast_frame(result)
                if result.is_fall:
                    self._handle_fall(result, frame)
            except Exception as e:
                logger.error(f"Frame processing error: {e}", exc_info=True)

        logger.info("Capture loop ended")

    def _broadcast_frame(self, result):
        """Encode annotated frame and emit over SocketIO."""
        if result.annotated_frame is None:
            return
        try:
            _, buf = cv2.imencode('.jpg', result.annotated_frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf).decode('utf-8')

            self.socketio.emit('frame_update', {
                'frame': b64,
                'activity': result.activity,
                'is_fall': result.is_fall,
                'confidence': result.confidence,
                'conf_posture': result.conf_posture,
                'conf_velocity': result.conf_velocity,
                'conf_height': result.conf_height,
                'conf_inactivity': result.conf_inactivity,
                'conf_context': result.conf_context,
                'body_angle': result.body_angle,
                'velocity': result.velocity,
                'context_surface': result.context_surface,
                'detected_objects': result.detected_objects,
                'fps': self.detector.fps
            })
        except Exception as e:
            logger.debug(f"Broadcast error: {e}")

    def _handle_fall(self, result, frame: np.ndarray):
        """Save event image and emit fall alert if cooldown passed."""
        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown:
            return
        self.last_alert_time = now

        # Save snapshot
        ts = datetime.utcnow()
        fname = f"fall_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
        fpath = os.path.join(self.event_image_dir, fname)
        try:
            cv2.imwrite(fpath, frame)
        except Exception as e:
            logger.error(f"Could not save event image: {e}")
            fname = None

        # Persist event to DB
        image_path = f"events/{fname}" if fname else None
        event_id = self._save_event(result, image_path, ts)

        # Emit alert
        self.socketio.emit('fall_alert', {
            'event_id': event_id,
            'timestamp': ts.isoformat(),
            'confidence': result.confidence,
            'activity': result.activity,
            'image_path': image_path,
            'conf_posture': result.conf_posture,
            'conf_velocity': result.conf_velocity,
            'conf_height': result.conf_height,
            'conf_inactivity': result.conf_inactivity,
            'conf_context': result.conf_context,
        })
        logger.warning(f"FALL DETECTED — conf={result.confidence:.2f} event_id={event_id}")

    def _save_event(self, result, image_path: str, ts: datetime) -> str:
        """Persist fall event to database inside a Flask app context."""
        import json

        def _do_save(app):
            # FIX: push an app context so SQLAlchemy works from this background thread
            with app.app_context():
                try:
                    from app import db
                    from app.models.event import Event
                    event = Event(
                        event_type='fall',
                        activity_label=result.activity,
                        is_fall=True,
                        confidence_total=result.confidence,
                        confidence_posture=result.conf_posture,
                        confidence_velocity=result.conf_velocity,
                        confidence_height=result.conf_height,
                        confidence_inactivity=result.conf_inactivity,
                        confidence_context=result.conf_context,
                        body_angle=result.body_angle,
                        velocity=result.velocity,
                        detected_objects=json.dumps(result.detected_objects),
                        image_path=image_path,
                        timestamp=ts
                    )
                    db.session.add(event)
                    db.session.commit()
                    logger.info(f"Event saved to DB: {event.event_id}")
                    return event.event_id
                except Exception as e:
                    logger.error(f"DB save error: {e}", exc_info=True)
                    try:
                        from app import db
                        db.session.rollback()
                    except Exception:
                        pass
                    return 'unknown'

        if self._flask_app is not None:
            return _do_save(self._flask_app)
        else:
            # Fallback: try importing the app directly
            try:
                from app import create_app
                app = create_app()
                return _do_save(app)
            except Exception as e:
                logger.error(f"Could not get Flask app for DB save: {e}")
                return 'unknown'