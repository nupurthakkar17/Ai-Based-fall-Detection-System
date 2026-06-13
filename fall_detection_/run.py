"""
FallGuard AI — Main entry point
Run: python run.py
"""

import os
import sys
import logging

# Ensure project root is on sys.path so `from app import ...` resolves
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Force working directory to project root so SQLite path resolves correctly
os.chdir(ROOT)

from app import create_app, socketio

logger = logging.getLogger(__name__)


def _check_database(app):
    """Verify the database file path and report its status on startup."""
    db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
    print(f"  [DB] URI : {db_uri}")

    if db_uri.startswith('sqlite:///'):
        db_path = db_uri.replace('sqlite:///', '')
        # Relative paths are relative to ROOT
        if not os.path.isabs(db_path):
            db_path = os.path.join(ROOT, db_path)

        if os.path.exists(db_path):
            size_kb = os.path.getsize(db_path) / 1024
            print(f"  [DB] File : {db_path}  ✓  ({size_kb:.1f} KB)")
        else:
            print(f"  [DB] File : {db_path}  ← will be created fresh (no existing data)")
    else:
        print(f"  [DB] Using external DB (PostgreSQL / other)")


def main():
    env = os.getenv('FLASK_ENV', 'development')
    app = create_app(env)

    # ── DB diagnostic ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  FallGuard AI  [{env}]")
    _check_database(app)
    print(f"{'='*50}")
    # ──────────────────────────────────────────────────────────────────────────

    with app.app_context():
        # Initialize AI detector
        from app.services.fall_detector import FallDetector
        from app.services.camera_service import CameraService

        det_config = {
            'FALL_CONFIDENCE_THRESHOLD':    app.config.get('FALL_CONFIDENCE_THRESHOLD',    0.85),
            'INACTIVITY_THRESHOLD_SECONDS': app.config.get('INACTIVITY_THRESHOLD_SECONDS', 5),
            'VELOCITY_THRESHOLD':           app.config.get('VELOCITY_THRESHOLD',           0.15),
            'OVERLAP_THRESHOLD':            app.config.get('OVERLAP_THRESHOLD',            0.5),
            'WEIGHT_POSTURE':               app.config.get('WEIGHT_POSTURE',               0.30),
            'WEIGHT_VELOCITY':              app.config.get('WEIGHT_VELOCITY',              0.25),
            'WEIGHT_HEIGHT_CHANGE':         app.config.get('WEIGHT_HEIGHT_CHANGE',         0.20),
            'WEIGHT_INACTIVITY':            app.config.get('WEIGHT_INACTIVITY',            0.15),
            'WEIGHT_OBJECT_CONTEXT':        app.config.get('WEIGHT_OBJECT_CONTEXT',        0.10),
        }

        detector = FallDetector(config=det_config)

        cam_config = {
            'ALERT_COOLDOWN_SECONDS': app.config.get('ALERT_COOLDOWN_SECONDS', 30),
            'EVENT_IMAGE_DIR':        app.config.get('EVENT_IMAGE_DIR', os.path.join(ROOT, 'static', 'events')),
            'camera_index':           0
        }
        camera_service = CameraService(socketio, detector, cam_config)
        app.config['_camera_service'] = camera_service

    host  = os.getenv('HOST',  '0.0.0.0')
    port  = int(os.getenv('PORT', 5000))
    debug = (env == 'development')

    print(f"\n  Running at  →  http://localhost:{port}")
    print(f"  Login       →  admin / admin123")
    print(f"{'='*50}\n")

    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()