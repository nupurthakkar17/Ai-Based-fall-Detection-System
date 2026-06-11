"""REST API endpoints for camera control, events, export."""
from flask import Blueprint, jsonify, request, current_app, send_file
from flask_login import login_required, current_user
from app.models.event import Event
import io

api_bp = Blueprint('api', __name__)

def get_camera():
    return current_app.config.get('_camera_service')

@api_bp.route('/camera/start', methods=['POST'])
@login_required
def camera_start():
    cam = get_camera()
    if cam is None: return jsonify({'success': False, 'message': 'Camera service unavailable'}), 503
    idx = request.json.get('camera_index', 0) if request.is_json else 0
    return jsonify(cam.start(idx))

@api_bp.route('/camera/stop', methods=['POST'])
@login_required
def camera_stop():
    cam = get_camera()
    if cam is None: return jsonify({'success': False}), 503
    return jsonify(cam.stop())

@api_bp.route('/camera/status')
@login_required
def camera_status():
    cam = get_camera()
    if cam is None: return jsonify({'is_running': False})
    return jsonify(cam.get_status())

@api_bp.route('/camera/switch', methods=['POST'])
@login_required
def camera_switch():
    cam = get_camera()
    idx = (request.json or {}).get('index', 0)
    return jsonify(cam.switch_camera(idx))

@api_bp.route('/events')
@login_required
def events():
    page  = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    only_falls = request.args.get('falls_only', 'false').lower() == 'true'
    q = Event.query
    if only_falls:
        q = q.filter_by(is_fall=True)
    events = q.order_by(Event.timestamp.desc()).paginate(page=page, per_page=limit, error_out=False)
    return jsonify({'events': [e.to_dict() for e in events.items], 'total': events.total, 'pages': events.pages})

@api_bp.route('/export/csv')
@login_required
def export_csv():
    from app.services.report_service import export_events_csv
    events = Event.query.order_by(Event.timestamp.desc()).all()
    csv_bytes = export_events_csv([e.to_dict() for e in events])
    return send_file(io.BytesIO(csv_bytes), mimetype='text/csv',
                     as_attachment=True, download_name='fallguard_events.csv')

@api_bp.route('/export/pdf')
@login_required
def export_pdf():
    from app.services.report_service import generate_pdf_report
    events = Event.query.order_by(Event.timestamp.desc()).limit(200).all()
    pdf = generate_pdf_report([e.to_dict() for e in events], current_user.to_dict())
    mime = 'application/pdf' if pdf[:4] == b'%PDF' else 'text/html'
    return send_file(io.BytesIO(pdf), mimetype=mime,
                     as_attachment=True, download_name='fallguard_report.pdf')
