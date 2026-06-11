"""
FallGuard AI — Root-level shim.
The real factory lives in app/__init__.py.
This file exists only so that `python app.py` still works.
"""
from app import create_app, socketio

if __name__ == '__main__':
    import os
    app = create_app(os.getenv('FLASK_ENV', 'development'))
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
