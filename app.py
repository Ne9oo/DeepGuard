import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg') # Headless backend for server use
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from tensorflow.keras.models import load_model
from fpdf import FPDF

# Import dotenv to keep your passwords secure
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the Flask App
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'deepguard_super_secret_key_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepguard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ==========================================
# EMAIL CONFIGURATION (SMTP)
# ==========================================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
# SECURED: Pulling credentials from the hidden .env file
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = 'DeepGuard Security <noreply@deepguard.com>'

mail = Mail(app)
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Upload & Storage Configuration
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
SPECTROGRAM_FOLDER = os.path.join(app.root_path, 'static', 'spectrograms')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'opus', 'm4a', 'aac', 'flac', 'wma', 'amr', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True) 

# Initialize Extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'alert'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
# LOAD DEEP LEARNING MODEL
# ==========================================
MODEL_PATH = os.path.join(app.root_path, 'deepguard_cnn.h5')
try:
    model = load_model(MODEL_PATH)
    print("[+] DeepGuard CNN Model loaded successfully.")
except Exception as e:
    model = None
    print(f"[-] WARNING: Could not load 'deepguard_cnn.h5'. AI functionality will run in simulation mode. Error: {e}")

# ==========================================
# DATABASE MODELS
# ==========================================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    
    # Core Features
    scans_used = db.Column(db.Integer, default=0)
    is_pro = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # System Settings Preferences
    scan_mode = db.Column(db.String(20), default='standard') # 'fast', 'standard', 'pro'
    auto_delete = db.Column(db.Boolean, default=True)
    email_alerts = db.Column(db.Boolean, default=False)
    
    scans = db.relationship('ScanRecord', backref='user', lazy=True)

class ScanRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    scan_date = db.Column(db.DateTime, default=datetime.utcnow)
    result = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==========================================
# SECURITY DECORATORS & HELPERS
# ==========================================
def requires_verification(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_verified:
            return redirect(url_for('unverified'))
        return f(*args, **kwargs)
    return decorated_function

def send_verification_email(user_email):
    token = s.dumps(user_email, salt='email-verify')
    verify_url = url_for('verify_email', token=token, _external=True)
    
    msg = Message('Verify Your DeepGuard Account', recipients=[user_email])
    
    # --- UPDATED PROFESSIONAL EMAIL TEXT ---
    msg.body = f'''Welcome to DeepGuard!

You are one step away from accessing our advanced audio forensics platform. Please verify your email address to activate your account by clicking the secure link below:

{verify_url}

SECURITY WARNING: If you did not sign up for a DeepGuard account, please ignore and delete this email. Do not click the link above.

Stay secure,
The DeepGuard Team'''
    # ---------------------------------------

    try:
        mail.send(msg)
    except Exception as e:
        print(f"Email sending failed: {e}")

def send_scan_alert_email(user_email, filename, result, risk):
    msg = Message('DeepGuard: Scan Completed', recipients=[user_email])
    msg.body = f'''Hello!
    
Your DeepGuard audio analysis is complete.

File: {filename}
Verdict: {result}
Risk Level: {risk}

Log in to your DeepGuard dashboard to view the spectrogram and download the PDF forensic report.
'''
    try:
        mail.send(msg)
    except Exception as e:
        print(f"Scan alert email failed: {e}")

# ==========================================
# FORENSICS ANALYSIS PIPELINE
# ==========================================
def extract_mfcc(audio, sample_rate, max_pad_len=150):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    if mfccs.shape[1] > max_pad_len:
        mfccs = mfccs[:, :max_pad_len]
    else:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def generate_spectrogram_image(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 3.5))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Acoustic Mel-Spectrogram Profile')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=False, facecolor='#0a0f1c')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def analyze_audio_forensics(filepath, mode='standard'):
    
    # Apply user preference for scan depth
    if mode == 'fast':
        duration = 3.0
    elif mode == 'pro':
        duration = 10.0
    else:
        duration = 5.0
        
    y, sr = librosa.load(filepath, sr=16000, duration=duration)
    spectrogram_b64 = generate_spectrogram_image(y, sr)
    
    if model is None:
        import random
        is_synthetic = random.choice([True, False])
        fake_prob = random.uniform(0.75, 0.99) if is_synthetic else random.uniform(0.01, 0.25)
    else:
        features = extract_mfcc(y, sr)
        features = features.reshape(1, features.shape[0], features.shape[1], 1)
        prediction = model.predict(features)
        fake_prob = float(prediction[0][0])
        
    if fake_prob > 0.5:
        result = 'Deepfake'
        confidence = round(fake_prob * 100, 1)
    else:
        result = 'Authentic'
        confidence = round((1 - fake_prob) * 100, 1)
        
    if fake_prob >= 0.71:
        risk_level = 'High Risk'
    elif fake_prob >= 0.31:
        risk_level = 'Medium Risk'
    else:
        risk_level = 'Low Risk'
        
    return {
        'result': result,
        'confidence': confidence,
        'risk_level': risk_level,
        'spectrogram': spectrogram_b64
    }

# ==========================================
# CORE ROUTES
# ==========================================
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if password != confirm_password:
            flash('Passwords do not match.', 'alert')
            return redirect(url_for('signup'))
        if User.query.filter_by(email=email).first():
            flash('Email is already registered. Please log in.', 'alert')
            return redirect(url_for('signup'))
        if User.query.filter_by(username=username).first():
            flash('Username is already taken. Please choose another.', 'alert')
            return redirect(url_for('signup'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password, is_admin=False, is_verified=False)
        db.session.add(new_user)
        db.session.commit()
        send_verification_email(email)
        flash('Account created! Please check your email to verify your account before logging in.', 'safe')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            if not user.is_verified: return redirect(url_for('unverified'))
            if user.is_admin: return redirect(url_for('admin_dashboard'))
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password.', 'alert')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/unverified')
@login_required
def unverified():
    if current_user.is_verified: return redirect(url_for('dashboard'))
    return render_template('unverified.html')

@app.route('/resend_verification')
@login_required
def resend_verification():
    if current_user.is_verified: return redirect(url_for('dashboard'))
    send_verification_email(current_user.email)
    flash('A new verification email has been sent.', 'safe')
    return redirect(url_for('unverified'))

@app.route('/verify/<token>')
def verify_email(token):
    try:
        email = s.loads(token, salt='email-verify', max_age=3600)
    except SignatureExpired:
        flash('The verification link has expired. Please log in to request a new one.', 'alert')
        return redirect(url_for('login'))
    except BadTimeSignature:
        flash('Invalid verification link.', 'alert')
        return redirect(url_for('login'))
    user = User.query.filter_by(email=email).first_or_404()
    if user.is_verified:
        flash('Account already verified. Please log in.', 'safe')
    else:
        user.is_verified = True
        db.session.commit()
        flash('Your account has been successfully verified!', 'safe')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
@requires_verification
def dashboard():
    return render_template('dashboard.html')

@app.route('/scan', methods=['POST'])
@login_required
@requires_verification
def scan_audio():
    if not current_user.is_pro and current_user.scans_used >= 5:
        return jsonify({'success': False, 'message': 'Daily scan limit reached (5/5). Please upgrade to Pro.'}), 403

    if 'audio_file' not in request.files:
        return jsonify({'success': False, 'message': 'No audio file found.'}), 400

    file = request.files['audio_file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file format.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}")
    file.save(filepath)

    try:
        # Pass the user's preferred scan mode (Fast/Standard/Pro)
        analysis = analyze_audio_forensics(filepath, mode=current_user.scan_mode)
        
        new_scan = ScanRecord(filename=filename, result=analysis['result'], confidence=analysis['confidence'], user_id=current_user.id)
        current_user.scans_used += 1
        db.session.add(new_scan)
        db.session.commit() 
        
        try:
            image_data = base64.b64decode(analysis['spectrogram'])
            img_path = os.path.join(SPECTROGRAM_FOLDER, f"spectrogram_{new_scan.id}.png")
            with open(img_path, 'wb') as img_file:
                img_file.write(image_data)
        except Exception as e:
            print(f"Failed to save spectrogram image: {e}")

        # APPLY SETTING: Auto-Delete File
        if current_user.auto_delete and os.path.exists(filepath):
            os.remove(filepath)
            
        # APPLY SETTING: Email Alerts
        if current_user.email_alerts:
            send_scan_alert_email(current_user.email, filename, analysis['result'], analysis['risk_level'])

        return jsonify({
            'success': True, 
            'filename': filename, 
            'result': analysis['result'], 
            'confidence': analysis['confidence'], 
            'risk_level': analysis['risk_level'], 
            'spectrogram': analysis['spectrogram'],
            'scans_used': current_user.scans_used, 
            'is_pro': current_user.is_pro
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'message': f'Analysis error: {str(e)}'}), 500

@app.route('/history')
@login_required
@requires_verification
def history():
    user_scans = ScanRecord.query.filter_by(user_id=current_user.id).order_by(ScanRecord.scan_date.desc()).all()
    return render_template('history.html', scans=user_scans)

@app.route('/delete_scan/<int:scan_id>', methods=['POST'])
@login_required
@requires_verification
def delete_scan(scan_id):
    scan = ScanRecord.query.get_or_404(scan_id)
    if scan.user_id != current_user.id and not current_user.is_admin:
        flash('Unauthorized to delete this record.', 'alert')
        return redirect(url_for('history'))

    spec_path = os.path.join(SPECTROGRAM_FOLDER, f"spectrogram_{scan.id}.png")
    if os.path.exists(spec_path):
        os.remove(spec_path)

    db.session.delete(scan)
    db.session.commit()
    flash('Scan record deleted successfully.', 'safe')
    return redirect(url_for('history'))

@app.route('/download_report/<int:scan_id>')
@login_required
@requires_verification
def download_report(scan_id):
    scan = ScanRecord.query.get_or_404(scan_id)
    if scan.user_id != current_user.id and not current_user.is_admin:
        flash('Unauthorized access.', 'alert')
        return redirect(url_for('history'))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 22)
    pdf.cell(0, 15, txt="DeepGuard Security Incident Report", ln=True, align='C')
    pdf.line(10, 25, 200, 25)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, txt="Scan ID:", border=0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=f"DG-{scan.id:06d}", border=0, ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, txt="Timestamp:", border=0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=scan.scan_date.strftime("%Y-%m-%d %H:%M:%S UTC"), border=0, ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, txt="Target File:", border=0)
    pdf.set_font("Arial", '', 12)
    display_filename = (scan.filename[:65] + '...') if len(scan.filename) > 65 else scan.filename
    pdf.cell(0, 10, txt=display_filename, border=0, ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, txt="Requested By:", border=0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=f"{current_user.username}", border=0, ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Forensic Verdict", ln=True)
    pdf.line(10, 85, 200, 85)
    pdf.ln(5)
    verdict = "Synthetic Deepfake" if scan.result == 'Deepfake' else "Authentic Human Voice"
    risk = "HIGH RISK" if scan.result == 'Deepfake' else "LOW RISK"
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, txt="Classification:", border=0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=verdict, border=0, ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, txt="Confidence:", border=0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=f"{scan.confidence}%", border=0, ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, txt="Risk Level:", border=0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=risk, border=0, ln=True)
    pdf.ln(10)
    spec_path = os.path.join(app.root_path, 'static', 'spectrograms', f"spectrogram_{scan.id}.png")
    if os.path.exists(spec_path):
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt="Visual Spectrogram Analysis", ln=True)
        y_before_img = pdf.get_y()
        pdf.image(spec_path, x=10, y=y_before_img, w=190)
        pdf.set_y(y_before_img + 75) 
        pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 6, txt="Disclaimer: DeepGuard AI Analysis.")
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return send_file(io.BytesIO(pdf_bytes), as_attachment=True, download_name=f"DeepGuard_{scan.id}.pdf", mimetype='application/pdf')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
@requires_verification
def profile():
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        if form_type == 'update_username':
            new_username = request.form.get('username').strip()
            if new_username and new_username != current_user.username and not User.query.filter_by(username=new_username).first():
                current_user.username = new_username
                db.session.commit()
                flash('Username updated!', 'safe')
            return redirect(url_for('profile'))
        elif form_type == 'update_password':
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            if bcrypt.check_password_hash(current_user.password, current_password) and new_password == request.form.get('confirm_new_password'):
                current_user.password = bcrypt.generate_password_hash(new_password).decode('utf-8')
                db.session.commit()
                flash('Password updated!', 'safe')
            return redirect(url_for('profile'))
    return render_template('profile.html')

@app.route('/settings', methods=['GET', 'POST'])
@login_required
@requires_verification
def settings():
    if request.method == 'POST':
        current_user.scan_mode = request.form.get('scan_mode')
        current_user.auto_delete = 'auto_delete' in request.form
        current_user.email_alerts = 'email_alerts' in request.form
        
        db.session.commit()
        flash('System settings updated and saved successfully!', 'safe')
        return redirect(url_for('settings'))
        
    return render_template('settings.html')

@app.route('/subscription', methods=['GET', 'POST'])
@login_required
@requires_verification
def subscription():
    if request.method == 'POST':
        current_user.is_pro = True
        db.session.commit()
        flash('Upgraded to Pro!', 'safe')
        return redirect(url_for('subscription'))
    return render_template('subscription.html')

@app.route('/admin')
@login_required
@requires_verification
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access Denied.', 'alert')
        return redirect(url_for('dashboard'))
    
    all_users = User.query.all()
    recent_scans = ScanRecord.query.order_by(ScanRecord.scan_date.desc()).limit(10).all()

    # Calculate real scan volume for the last 7 days
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    traffic_labels = []
    traffic_data = []

    for i in range(6, -1, -1):
        target_date = today - timedelta(days=i)
        next_date = target_date + timedelta(days=1)
        
        # Get the day name (e.g., 'Mon', 'Tue')
        traffic_labels.append(target_date.strftime('%a')) 
        
        # Count how many scans happened on this specific day
        count = ScanRecord.query.filter(
            ScanRecord.scan_date >= target_date,
            ScanRecord.scan_date < next_date
        ).count()
        traffic_data.append(count)

    return render_template('admin.html', users=all_users, scans=recent_scans, traffic_labels=traffic_labels, traffic_data=traffic_data)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        admin_email = 'admin@deepguard.com'
        if not User.query.filter_by(email=admin_email).first():
            hashed_pw = bcrypt.generate_password_hash('admin123').decode('utf-8')
            admin_user = User(username='Admin', email=admin_email, password=hashed_pw, is_admin=True, is_verified=True)
            db.session.add(admin_user)
            db.session.commit()
    app.run(debug=True)