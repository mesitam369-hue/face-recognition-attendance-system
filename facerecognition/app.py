from flask import Flask, render_template, redirect, url_for, flash, request, send_file, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import os
import io
import json
import time
from datetime import datetime
import pandas as pd
from sqlalchemy.exc import IntegrityError
import base64
import cv2

from models import db, User, FaceEncoding, Attendance, Subject
from camera_utils import VideoCamera, get_face_encoding, match_face, get_face_liveness_metrics, draw_face_box
import numpy as np
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_for_this_mini_project'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

camera = None

def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera()
    return camera

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_defaults():
    if not User.query.filter_by(role='teacher').first():
        hashed_pw = bcrypt.generate_password_hash('admin').decode('utf-8')
        default_teacher = User(username='admin', name='Admin Teacher', password_hash=hashed_pw, role='teacher')
        db.session.add(default_teacher)
        db.session.commit()
    
    if not User.query.filter_by(username='student').first():
        hashed_pw = bcrypt.generate_password_hash('student').decode('utf-8')
        default_student = User(username='student', name='Test Student', password_hash=hashed_pw, role='student')
        db.session.add(default_student)
        db.session.commit()
    # Initialize Subjects matching the screenshot
    default_subjects = [
        {"name": "Engineering Graphics", "icon": "fa-pencil-ruler", "hours": 24},
        {"name": "Engineering Physics", "icon": "fa-atom", "hours": 30},
        {"name": "Mathematics", "icon": "fa-square-root-alt", "hours": 40},
        {"name": "Programming in Python", "icon": "fa-code", "hours": 36},
        {"name": "Life Skills", "icon": "fa-user-graduate", "hours": 20},
        {"name": "General", "icon": "fa-globe", "hours": 10}
    ]
    for subj in default_subjects:
        existing_subj = Subject.query.filter_by(name=subj['name']).first()
        if not existing_subj:
            new_subj = Subject(name=subj['name'], icon=subj['icon'], total_hours=subj['hours'])
            db.session.add(new_subj)
        else:
            # Update existing with default if needed, though usually skip
            pass
    db.session.commit()

with app.app_context():
    db.create_all()
    init_defaults()

@app.route('/', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.role == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
            
    if request.method == 'POST':
        # Now handles both Teacher and Student logic from potentially the same form inputs
        role_type = request.form.get('role_type', 'all')
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            if role_type != 'all' and user.role != role_type:
                 flash(f'Account exists, but not for the selected {role_type} role.', 'danger')
                 return redirect(url_for('login', type=role_type))
            
            login_user(user)
            if user.role == 'teacher':
                return redirect(url_for('teacher_dashboard'))
            else:
                return redirect(url_for('student_dashboard'))
        else:
            flash('Login Unsuccessful. Please check credentials', 'danger')
            return redirect(url_for('login', type=role_type))
            
    return render_template('login.html', type=request.args.get('type'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form.get('username')
        recovery_key = request.form.get('recovery_key')
        new_password = request.form.get('new_password')
        
        # Simple recovery logic using the app's secret key as the recovery key
        if recovery_key == app.config['SECRET_KEY']:
            user = User.query.filter_by(username=username).first()
            if user:
                user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
                db.session.commit()
                flash('Password reset successful! You can now login.', 'success')
                return redirect(url_for('login'))
            else:
                flash('User not found.', 'danger')
        else:
            flash('Invalid recovery key.', 'danger')
            
    return render_template('forgot_password.html')

@app.route('/logout')
def logout():
    global camera
    if camera:
        camera.release()
        camera = None
    logout_user()
    return redirect(url_for('login'))

@app.route('/manage_students')
@login_required
def manage_students():
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))
    
    students = User.query.filter_by(role='student').all()
    return render_template('manage_students.html', students=students)

@app.route('/delete_student/<int:user_id>', methods=['POST'])
@login_required
def delete_student(user_id):
    if current_user.role != 'teacher':
        return jsonify({"success": False, "error": "Unauthorized"}), 401
        
    student = User.query.get_or_404(user_id)
    if student.role != 'student':
        return jsonify({"success": False, "error": "Cannot delete teachers"}), 400
        
    # Delete related records
    FaceEncoding.query.filter_by(user_id=user_id).delete()
    Attendance.query.filter_by(user_id=user_id).delete()
    
    db.session.delete(student)
    db.session.commit()
    
    flash(f'Student {student.name} and all related records have been deleted.', 'success')
    return redirect(url_for('manage_students'))

@app.route('/dashboard/teacher')
@login_required
def teacher_dashboard():
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))
    
    subjects = Subject.query.all()
    total_students = User.query.filter_by(role='student').count()
    today = datetime.utcnow().date()
    todays_attendance = Attendance.query.filter_by(date=today).count()
    
    from sqlalchemy import func
    
    # Efficiently get attendance counts for ALL students in one query
    attendance_counts = db.session.query(
        Attendance.user_id, func.count(Attendance.id)
    ).group_by(Attendance.user_id).all()
    attendance_map = {uid: count for uid, count in attendance_counts}
    
    # Calculate sum of all subject hours for overall student percentage
    total_possible_hours = db.session.query(func.sum(Subject.total_hours)).scalar() or 1
    
    students = User.query.filter_by(role='student').all()
    student_stats = []
    for student in students:
        att_count = attendance_map.get(student.id, 0)
        # Fix: Percentage relative to the sum of all subject hours
        pct = (att_count / float(total_possible_hours)) * 100 
        student_stats.append({
            'id': student.id,
            'name': student.name,
            'username': student.username,
            'percentage': min(100.0, pct)
        })
        
    # Efficiently get attendance counts for ALL subjects in one query
    subject_counts = db.session.query(
        Attendance.subject_id, func.count(Attendance.id)
    ).group_by(Attendance.subject_id).all()
    subject_map = {sid: count for sid, count in subject_counts}
    
    subject_stats = []
    for sbj in subjects:
        logs_count = subject_map.get(sbj.id, 0)
        # Fix: Subject Coverage % = (logs / (total_students * total_hours)) * 100
        possible_logs = total_students * sbj.total_hours
        percent = (logs_count / float(possible_logs)) * 100 if possible_logs > 0 else 0
        subject_stats.append({
            'subject': sbj,
            'logs': logs_count,
            'percentage': min(100.0, percent)
        })
        

    return render_template('teacher_dashboard.html', 
                            subjects=subject_stats, 
                            student_stats=student_stats,
                            total_students=total_students,
                            todays_attendance=todays_attendance)

@app.route('/add_subject', methods=['POST'])
@login_required
def add_subject():
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))
    
    name = request.form.get('name')
    icon = request.form.get('icon', 'fa-book')
    hours = request.form.get('hours', 20)
    
    if Subject.query.filter_by(name=name).first():
        flash(f'Subject {name} already exists.', 'danger')
    else:
        new_subj = Subject(name=name, icon=icon, total_hours=int(hours))
        db.session.add(new_subj)
        db.session.commit()
        flash(f'Subject {name} added successfully.', 'success')
        
    return redirect(url_for('teacher_dashboard'))

@app.route('/delete_subject/<int:subject_id>', methods=['POST'])
@login_required
def delete_subject(subject_id):
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))
        
    subject = Subject.query.get_or_404(subject_id)
    # Delete related attendance records
    Attendance.query.filter_by(subject_id=subject_id).delete()
    
    db.session.delete(subject)
    db.session.commit()
    flash(f'Subject {subject.name} and its records deleted.', 'success')
    return redirect(url_for('teacher_dashboard'))

@app.route('/edit_subject_hours/<int:subject_id>', methods=['POST'])
@login_required
def edit_subject_hours(subject_id):
    if current_user.role != 'teacher':
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    new_hours = request.form.get('total_hours')
    if not new_hours:
        return jsonify({"success": False, "error": "No hours provided"}), 400
    
    try:
        new_hours = int(new_hours)
        if new_hours <= 0:
            raise ValueError
    except ValueError:
        return jsonify({"success": False, "error": "Invalid hours value"}), 400
        
    subject = Subject.query.get_or_404(subject_id)
    subject.total_hours = new_hours
    db.session.commit()
    
    flash(f'Total hours for {subject.name} updated to {new_hours}.', 'success')
    return redirect(url_for('teacher_dashboard'))

@app.route('/dashboard/student')
@login_required
def student_dashboard():
    if current_user.role != 'student':
        return redirect(url_for('teacher_dashboard'))
        
    attendances = Attendance.query.filter_by(user_id=current_user.id).order_by(Attendance.date.desc()).all()
    total_attended = len(attendances)
    
    # Calculate subject-wise attendance for student
    subjects = Subject.query.all()
    subject_attendance = []
    from sqlalchemy import func
    counts = db.session.query(
        Attendance.subject_id, func.count(Attendance.id)
    ).filter(Attendance.user_id == current_user.id).group_by(Attendance.subject_id).all()
    count_map = {sid: cnt for sid, cnt in counts}
    
    total_possible_hours = sum([s.total_hours for s in subjects]) or 1
    overall_percentage = (total_attended / float(total_possible_hours)) * 100
    
    for s in subjects:
        count = count_map.get(s.id, 0)
        # Subject specific percentage: (attended / total_subject_hours) * 100
        pct = (count / float(s.total_hours)) * 100 if s.total_hours > 0 else 0
        subject_attendance.append({
            'subject': s,
            'count': count,
            'percentage': min(100.0, pct)
        })
    
    return render_template('student_dashboard.html', 
                          attendances=attendances, 
                          total_attended=total_attended, 
                          overall_percentage=min(100.0, overall_percentage),
                          subject_attendance=subject_attendance)

@app.route('/register_student', methods=['GET', 'POST'])
@login_required
def register_student():
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))
        
    step = request.args.get('step')
    
    if request.method == 'POST' and not step:
        name = request.form.get('name')
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash(f'Student ID {username} already exists.', 'danger')
            return redirect(url_for('register_student'))
            
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        new_student = User(username=username, name=name, password_hash=hashed_pw, role='student')
        
        db.session.add(new_student)
        db.session.commit()
        
        return redirect(url_for('register_student', step='capture', user_id=new_student.id, name=name, username=username))
        
    return render_template('register_student.html')

@app.route('/capture_face/<int:user_id>', methods=['POST'])
@login_required
def capture_face(user_id):
    if current_user.role != 'teacher':
        return jsonify({"success": False}), 401
        
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "No image data provided"}), 400
        
    try:
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"success": False, "error": "Invalid image format."}), 400
    
    if image is not None:
        encoding = get_face_encoding(image)
        if encoding is not None:
            face_enc = FaceEncoding(user_id=user_id)
            face_enc.set_encoding(encoding.tolist())
            db.session.add(face_enc)
            db.session.commit()
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "No face detected or poor quality. Please reposition."})
            
    return jsonify({"success": False, "error": "Could not decode image."})

@app.route('/finish_registration', methods=['POST'])
@login_required
def finish_registration():
    if current_user.role != 'teacher':
         return redirect(url_for('student_dashboard'))
    global camera
    if camera:
        camera.release()
        camera = None
    flash('Student Registration complete!', 'success')
    return redirect(url_for('teacher_dashboard'))

@app.route('/take_attendance/<int:subject_id>')
@login_required
def take_attendance(subject_id):
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))
        
    subject = Subject.query.get_or_404(subject_id)
    return render_template('take_attendance.html', subject=subject)

attendance_sessions = {}

@app.route('/stop_attendance/<int:subject_id>')
@login_required
def stop_attendance(subject_id):
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))
    global camera
    if camera:
        camera.release()
        camera = None
    if subject_id in attendance_sessions:
        del attendance_sessions[subject_id]
    flash('Session ended successfully. Reviewing results...', 'info')
    return redirect(url_for('session_report', subject_id=subject_id))

@app.route('/session_report/<int:subject_id>')
@login_required
def session_report(subject_id):
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))
    
    subject = Subject.query.get_or_404(subject_id)
    today = datetime.utcnow().date()
    attendances = Attendance.query.filter_by(subject_id=subject_id, date=today).all()
    
    return render_template('session_report.html', subject=subject, attendances=attendances, count=len(attendances))

@app.route('/api/process_frame/<int:subject_id>', methods=['POST'])
@login_required
def process_frame(subject_id):
    if current_user.role != 'teacher':
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image"}), 400

    image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    try:
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    if subject_id not in attendance_sessions:
        with app.app_context():
            all_students = User.query.filter_by(role='student').all()
            known_encodings = []
            known_ids = []
            known_names = []
            for student in all_students:
                encs = FaceEncoding.query.filter_by(user_id=student.id).all()
                for enc in encs:
                    known_encodings.append(np.array(enc.get_encoding()))
                    known_ids.append(student.id)
                    known_names.append(student.name)
        
        attendance_sessions[subject_id] = {
            'known_encodings': known_encodings,
            'known_ids': known_ids,
            'known_names': known_names,
            'liveness_status': {}
        }

    session = attendance_sessions[subject_id]
    liveness_status = session['liveness_status']
    
    name_to_display = "Scanning..."
    matched_user_id = None
    matched_name = "Scanning..."
    
    current_encoding = get_face_encoding(image)
    if current_encoding is not None:
        if len(session['known_encodings']) > 0:
            match_idx = match_face(current_encoding, session['known_encodings'], tolerance=0.48)
            if match_idx != -1:
                matched_user_id = session['known_ids'][match_idx]
                matched_name = session['known_names'][match_idx]
            else:
                matched_name = "Unknown"
        else:
            matched_name = "Unknown"

    CHALLENGE_TYPES = ["blink", "left", "right", "up", "down"]
    CHALLENGE_LABELS = {
        "blink": "Blink",
        "left": "Turn Head Left",
        "right": "Turn Head Right",
        "up": "Look Up",
        "down": "Look Down"
    }
    EYE_AR_THRESH = 0.20
    YAW_THRESH = 0.25
    PITCH_THRESH = 0.18

    liveness_data = None
    if matched_user_id:
        liveness_data = get_face_liveness_metrics(image)
        
        if matched_user_id not in liveness_status:
            selected = random.sample(CHALLENGE_TYPES, 2)
            liveness_status[matched_user_id] = {
                "challenges": selected,
                "idx": 0,
                "verified": False,
                "last_action_time": time.time()
            }
            
        status = liveness_status[matched_user_id]
        
        if not status["verified"]:
            current_challenge = status["challenges"][status["idx"]]
            is_meeting_challenge = False
            
            if liveness_data:
                ear = liveness_data['ear']
                yaw = liveness_data['yaw']
                pitch = liveness_data['pitch']
                
                if current_challenge == "blink" and ear < EYE_AR_THRESH:
                    is_meeting_challenge = True
                elif current_challenge == "left" and yaw < -YAW_THRESH:
                    is_meeting_challenge = True
                elif current_challenge == "right" and yaw > YAW_THRESH:
                    is_meeting_challenge = True
                elif current_challenge == "up" and pitch < -PITCH_THRESH:
                    is_meeting_challenge = True
                elif current_challenge == "down" and pitch > PITCH_THRESH:
                    is_meeting_challenge = True
            
            if is_meeting_challenge:
                status["idx"] += 1
                status["last_action_time"] = time.time()
                if status["idx"] >= len(status["challenges"]):
                    status["verified"] = True
            
            if not status["verified"]:
                action_text = CHALLENGE_LABELS[current_challenge]
                name_to_display = f"{matched_name} - Step {status['idx']+1}/{len(status['challenges'])}: {action_text}"
            else:
                name_to_display = f"{matched_name} (Verified)"
        else:
            name_to_display = f"{matched_name} (Verified)"

        if status["verified"]:
            with app.app_context():
                today = datetime.utcnow().date()
                existing = Attendance.query.filter_by(user_id=matched_user_id, subject_id=subject_id, date=today).first()
                if not existing:
                    new_att = Attendance(user_id=matched_user_id, subject_id=subject_id)
                    db.session.add(new_att)
                    try:
                        db.session.commit()
                    except Exception:
                        db.session.rollback()
        
        if time.time() - status.get("last_action_time", 0) > 10 and not status["verified"]:
            del liveness_status[matched_user_id]
    else:
        name_to_display = matched_name

    processed_b64 = draw_face_box(image, name_to_display)
    return jsonify({"success": True, "image": "data:image/jpeg;base64," + processed_b64})

@app.route('/api/recent_attendance/<int:subject_id>')
@login_required
def recent_attendance(subject_id):
    if current_user.role != 'teacher':
        return jsonify({"attendances": []})
        
    today = datetime.utcnow().date()
    recent = Attendance.query.filter_by(date=today, subject_id=subject_id).order_by(Attendance.time.desc()).limit(10).all()
    data = []
    for att in recent:
        data.append({
            "name": att.user.name,
            "time": att.time.strftime('%H:%M:%S')
        })
    return jsonify({"attendances": data})

@app.route('/export_attendance')
@login_required
def export_attendance():
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))
        
    attendances = Attendance.query.join(Subject).join(User).all()
    
    data = []
    for att in attendances:
        data.append({
            'Date': att.date.strftime('%Y-%m-%d'),
            'Time': att.time.strftime('%H:%M:%S'),
            'Student ID': att.user.username,
            'Name': att.user.name,
            'Subject': att.subject.name,
            'Status': att.status
        })
        
    df = pd.DataFrame(data)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Attendance')
    output.seek(0)
    
    return send_file(
        output,
        download_name=f"attendance_export_{datetime.utcnow().strftime('%Y%m%d')}.xlsx",
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
