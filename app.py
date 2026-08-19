import os
import numpy as np
import librosa
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from tensorflow.keras.models import load_model

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
app.config['MAIL_USERNAME'] = 'afiqdanish@gmail.com'     # <-- Your verified Gmail address
app.config['MAIL_PASSWORD'] = 'abcdefghijklmnop'         # <-- Your 16-character Google App Password
app.config['MAIL_DEFAULT_SENDER'] = 'DeepGuard Security <noreply@deepguard.com>'

mail = Mail(app)
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Upload Configuration
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    
    # Features
    scans_used = db.Column(db.Integer, default=0)
    is_pro = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_verified = db.Column(db.Boolean, default=False) # MFA Email Verification Status
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    scans = db.relationship('ScanRecord', backref='user', lazy=True)


class ScanRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    scan_date = db.Column(db.DateTime, default=datetime.utcnow)
    result = db.Column(db.String(20), nullable=False)  # 'Deepfake' or 'Authentic'
    confidence = db.Column(db.Float, nullable=True)     # e.g., 94.5
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
    msg.body = f'''Welcome to DeepGuard! 

To unlock your account and access the Voice Forensics Engine, please click the link below to verify your email address:
{verify_url}

If you did not make this request, please ignore this email.
'''
    try:
        mail.send(msg)
    except Exception as e:
        print(f"Email sending failed (Check your SMTP settings): {e}")


# ==========================================
# FORENSICS ANALYSIS PIPELINE (CNN & Librosa)
# ==========================================
def extract_mfcc(filepath, max_pad_len=150):
    try:
        audio, sample_rate = librosa.load(filepath, sr=16000, duration=5.0)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return mfccs
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def analyze_audio_forensics(filepath):
    if model is None:
        import random
        is_synthetic = random.choice([True, False])
        fake_prob = random.uniform(0.75, 0.99) if is_synthetic else random.uniform(0.01, 0.25)
    else:
        features = extract_mfcc(filepath)
        if features is None:
            raise ValueError("Could not extract audio features.")
            
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
        'risk_level': risk_level
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
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

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
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            
            if not user.is_verified:
                return redirect(url_for('unverified'))
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
                
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password.', 'alert')
            
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


# ==========================================
# VERIFICATION ROUTES
# ==========================================
@app.route('/unverified')
@login_required
def unverified():
    if current_user.is_verified:
        return redirect(url_for('dashboard'))
    return render_template('unverified.html')


@app.route('/resend_verification')
@login_required
def resend_verification():
    if current_user.is_verified:
        return redirect(url_for('dashboard'))
        
    send_verification_email(current_user.email)
    flash('A new verification email has been sent. Please check your inbox.', 'safe')
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
        flash('Your account has been successfully verified! You can now log in.', 'safe')
        
    return redirect(url_for('login'))


# ==========================================
# PROTECTED WORKSPACE ROUTES
# ==========================================
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
        analysis = analyze_audio_forensics(filepath)
        new_scan = ScanRecord(filename=filename, result=analysis['result'], confidence=analysis['confidence'], user_id=current_user.id)
        current_user.scans_used += 1
        db.session.add(new_scan)
        db.session.commit()

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({'success': True, 'filename': filename, 'result': analysis['result'], 'confidence': analysis['confidence'], 'risk_level': analysis['risk_level'], 'scans_used': current_user.scans_used, 'is_pro': current_user.is_pro})

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


@app.route('/profile', methods=['GET', 'POST'])
@login_required
@requires_verification
def profile():
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        
        # Handle Username Update
        if form_type == 'update_username':
            new_username = request.form.get('username').strip()
            
            if not new_username:
                flash('Username cannot be empty.', 'alert')
                return redirect(url_for('profile'))
                
            if new_username == current_user.username:
                flash('New username is the same as your current username.', 'alert')
                return redirect(url_for('profile'))
                
            existing_user = User.query.filter_by(username=new_username).first()
            if existing_user:
                flash('This username is already taken. Please choose another.', 'alert')
                return redirect(url_for('profile'))
                
            current_user.username = new_username
            db.session.commit()
            flash('Username updated successfully!', 'safe')
            return redirect(url_for('profile'))

        # Handle Password Update
        elif form_type == 'update_password':
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            confirm_new_password = request.form.get('confirm_new_password')
            
            if not bcrypt.check_password_hash(current_user.password, current_password):
                flash('Incorrect current password.', 'alert')
            elif new_password != confirm_new_password:
                flash('New passwords do not match.', 'alert')
            else:
                current_user.password = bcrypt.generate_password_hash(new_password).decode('utf-8')
                db.session.commit()
                flash('Password updated successfully.', 'safe')
            return redirect(url_for('profile'))
            
    return render_template('profile.html')


@app.route('/settings')
@login_required
@requires_verification
def settings():
    return render_template('settings.html')


@app.route('/subscription', methods=['GET', 'POST'])
@login_required
@requires_verification
def subscription():
    if request.method == 'POST':
        current_user.is_pro = True
        db.session.commit()
        flash('Successfully upgraded to DeepGuard Pro!', 'safe')
        return redirect(url_for('subscription'))
    return render_template('subscription.html')


@app.route('/admin')
@login_required
@requires_verification
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access Denied. Administrator privileges required.', 'alert')
        return redirect(url_for('dashboard'))
    
    all_users = User.query.all()
    recent_scans = ScanRecord.query.order_by(ScanRecord.scan_date.desc()).limit(10).all()
    total_users = len(all_users)
    total_scans = ScanRecord.query.count()
    fake_scans = ScanRecord.query.filter_by(result='Deepfake').count()
    detection_rate = round((fake_scans / total_scans * 100), 1) if total_scans > 0 else 0
    
    return render_template('admin.html', users=all_users, scans=recent_scans, total_users=total_users, total_scans=total_scans, detection_rate=detection_rate)


# ==========================================
# INITIALIZATION & SEEDING
# ==========================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        admin_email = 'admin@deepguard.com'
        if not User.query.filter_by(email=admin_email).first():
            hashed_pw = bcrypt.generate_password_hash('admin123').decode('utf-8')
            admin_user = User(username='Admin', email=admin_email, password=hashed_pw, is_admin=True, is_verified=True)
            db.session.add(admin_user)
            db.session.commit()
            print(f"[*] Premade Admin Account Generated! Email: {admin_email} | Password: admin123")
            
    app.run(debug=True)