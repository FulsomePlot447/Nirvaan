from flask import Flask, render_template, request, jsonify, Response, stream_template
import random
import requests
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import os
from threading import Thread
import time
import calendar

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# FastAPI backend URL (your existing AI backend)
AI_BACKEND_URL = "http://localhost:8000"

days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
periods_per_day = 6

# ===== Helper functions for timetable =====
def can_allocate(timetable, teacher, teacher_data, day, classroom, slot):
    if not (teacher_data[teacher]["start"] <= slot+1 <= teacher_data[teacher]["end"]):
        return False
    consec = 1
    if slot > 0 and timetable[day][classroom][slot-1] == teacher:
        consec += 1
        if slot > 1 and timetable[day][classroom][slot-2] == teacher:
            consec += 1
    return consec <= 2

def allocate_lab(timetable, branch, day, lab_name):
    for slot in range(periods_per_day-1):
        if timetable[day][branch][slot] == "FREE" and timetable[day][branch][slot+1] == "FREE":
            timetable[day][branch][slot] = lab_name
            timetable[day][branch][slot+1] = lab_name
            return True
    return False

# ===== Original Routes =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/calendar")
def show_calendar():
    import calendar
    from datetime import datetime

    # Get month and year from query params, fallback to current
    month = request.args.get("month", type=int)
    year = request.args.get("year", type=int)

    if not month or not year:
        today = datetime.today()
        month, year = today.month, today.year

    # Generate calendar
    cal = calendar.Calendar(firstweekday=6)  # Sunday start
    month_days = cal.monthdayscalendar(year, month)
    month_name = calendar.month_name[month]

    return render_template(
        "cal.html",
        month=month,
        year=year,
        month_name=month_name,
        month_days=month_days,
        datetime=datetime   # 👈 pass datetime into Jinja
    )


@app.route('/generate', methods=['POST'])
def generate():
    num_classrooms = int(request.form['classrooms'])
    num_faculty = int(request.form['faculty'])
    subjects = request.form['subjects'].split(',')

    # Initialize timetable
    timetable = {d: {f"Class {i+1}": ["FREE"] * periods_per_day for i in range(num_classrooms)} for d in days}

    # Teachers data
    teachers = {f"T{i+1}": {"start": 1, "end": 6} for i in range(num_faculty)}
    teachers["Super"] = {"start": 1, "end": 6}

    # Lab mapping per classroom
    labs = {f"Class {i+1}": f"Lab-{i+1}" for i in range(num_classrooms)}

    # Step 1: Allocate Labs
    for d in days:
        for c in labs:
            allocate_lab(timetable, c, d, labs[c])

    # Step 2: Super Teacher Rotation
    for i, d in enumerate(days):
        for j, c in enumerate(timetable[d]):
            super_teacher = "Super"
            for slot in range(periods_per_day):
                if timetable[d][c][slot] == "FREE" and can_allocate(timetable, super_teacher, teachers, d, c, slot):
                    timetable[d][c][slot] = f"Super Teacher"
                    break

    # Step 3: Allocate normal teachers with constraints
    normal_teachers = list(teachers.keys())
    normal_teachers.remove("Super")

    for d in days:
        for c in timetable[d]:
            for slot in range(periods_per_day):
                if timetable[d][c][slot] == "FREE":
                    teacher = random.choice(normal_teachers)
                    subject = random.choice(subjects)
                    if can_allocate(timetable, teacher, teachers, d, c, slot):
                        timetable[d][c][slot] = f"{subject} ({teacher})"

    return render_template('time-table.html', timetable=timetable, days=days, periods=periods_per_day)

# ===== New AI Routes =====
@app.route('/ai-assistant')
def ai_assistant():
    """Render the AI assistant page"""
    return render_template('ai_assistant.html')

@app.route('/ai/chat', methods=['POST'])
def ai_chat():
    """Proxy to AI backend for chat"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        session_id = data.get('session_id', f'flask_session_{int(time.time())}')
        file_context = data.get('file_context', None)
        
        # Forward request to FastAPI backend
        form_data = {
            'query': query,
            'session_id': session_id
        }
        if file_context:
            form_data['file_context'] = 'true'
            
        response = requests.post(f"{AI_BACKEND_URL}/stream", data=form_data, stream=True)
        
        def generate():
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    yield f"data: {chunk}\n\n"
        
        return Response(generate(), mimetype='text/plain')
        
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "AI backend not available. Make sure the FastAPI server is running on port 8000."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ai/upload', methods=['POST'])
def ai_upload():
    """Proxy to AI backend for file uploads"""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
            
        files = request.files.getlist('files')
        
        # Prepare files for forwarding
        files_data = []
        for file in files:
            if file.filename != '':
                files_data.append(('files', (file.filename, file.stream, file.content_type)))
        
        response = requests.post(f"{AI_BACKEND_URL}/upload", files=files_data)
        return jsonify(response.json())
        
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "AI backend not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ai/clear-context', methods=['POST'])
def ai_clear_context():
    """Proxy to AI backend for clearing context"""
    try:
        session_id = request.form.get('session_id', 'default')
        response = requests.post(f"{AI_BACKEND_URL}/clear-context", data={'session_id': session_id})
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ai/clear-files', methods=['DELETE'])
def ai_clear_files():
    """Proxy to AI backend for clearing files"""
    try:
        response = requests.delete(f"{AI_BACKEND_URL}/files")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ai/generate-pdf', methods=['POST'])
def ai_generate_pdf():
    """Proxy to AI backend for PDF generation"""
    try:
        data = request.get_json()
        response = requests.post(f"{AI_BACKEND_URL}/generate-pdf", json=data)
        
        if response.status_code == 200:
            return Response(
                response.content,
                mimetype='application/pdf',
                headers={'Content-Disposition': 'attachment; filename=chat_conversation.pdf'}
            )
        else:
            return jsonify({"error": "PDF generation failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting integrated Flask server...")
    print("🌐 Main website: http://localhost:5000")
    print("🤖 AI Assistant: http://localhost:5000/ai-assistant")
    print("📝 Make sure FastAPI backend is running on port 8000")
    app.run(debug=True, port=5000)