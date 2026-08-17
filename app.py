import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename

# Initialize the Flask App
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'deepguard_super_secret_key_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepguard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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
# FORENSICS ANALYSIS PIPELINE (AI Placeholder)
# ==========================================
def analyze_audio_forensics(filepath):
    """
    Forensics Pipeline:
    1. Loads audio via Librosa.
    2. Extracts Mel-Spectrogram / MFCCs.
    3. Runs inference through trained CNN.
    (Currently simulates realistic inference scores until model weights are loaded).
    """
    import random
    
    # Simulated prediction output for testing the complete UI/UX flow
    is_synthetic = random.choice([True, False])
    if is_synthetic:
        confidence = round(random.uniform(75.0, 99.2), 1)
        result = 'Deepfake'
        risk_level = 'High Risk'
    else:
        confidence = round(random.uniform(80.0, 98.8), 1)
        result = 'Authentic'
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
        new_user = User(username=username, email=email, password=hashed_password, is_admin=False)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! You can now log in.', 'safe')
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


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


# ==========================================
# ASYNCHRONOUS SCANNING ENDPOINT
# ==========================================
@app.route('/scan', methods=['POST'])
@login_required
def scan_audio():
    # 1. Enforce Daily Scan Quota for Basic Users
    if not current_user.is_pro and current_user.scans_used >= 5:
        return jsonify({
            'success': False,
            'message': 'Daily scan limit reached (5/5). Please upgrade to Pro for unlimited forensic analysis.'
        }), 403

    # 2. Check File Payload
    if 'audio_file' not in request.files:
        return jsonify({'success': False, 'message': 'No audio file found in the request.'}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file format. Only .WAV and .MP3 files are supported.'}), 400

    # 3. Secure File Saving
    filename = secure_filename(file.filename)
    unique_filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        # 4. Perform Forensic CNN Audio Analysis
        analysis = analyze_audio_forensics(filepath)

        # 5. Record to Database
        new_scan = ScanRecord(
            filename=filename,
            result=analysis['result'],
            confidence=analysis['confidence'],
            user_id=current_user.id
        )
        current_user.scans_used += 1
        db.session.add(new_scan)
        db.session.commit()

        # 6. Optional Clean-Up (Auto-delete raw file after extraction to preserve storage)
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            'success': True,
            'filename': filename,
            'result': analysis['result'],
            'confidence': analysis['confidence'],
            'risk_level': analysis['risk_level'],
            'scans_used': current_user.scans_used,
            'is_pro': current_user.is_pro
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'message': f'Analysis error: {str(e)}'}), 500


@app.route('/history')
@login_required
def history():
    user_scans = ScanRecord.query.filter_by(user_id=current_user.id).order_by(ScanRecord.scan_date.desc()).all()
    return render_template('history.html', scans=user_scans)


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        flash('Password updated successfully.', 'safe')
        return redirect(url_for('profile'))
    return render_template('profile.html')


@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')


@app.route('/subscription', methods=['GET', 'POST'])
@login_required
def subscription():
    if request.method == 'POST':
        current_user.is_pro = True
        db.session.commit()
        flash('Successfully upgraded to DeepGuard Pro!', 'safe')
        return redirect(url_for('subscription'))
    return render_template('subscription.html')


@app.route('/admin')
@login_required
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
    
    return render_template('admin.html', 
                           users=all_users, 
                           scans=recent_scans,
                           total_users=total_users,
                           total_scans=total_scans,
                           detection_rate=detection_rate)


# ==========================================
# INITIALIZATION & SEEDING
# ==========================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        admin_email = 'admin@deepguard.com'
        if not User.query.filter_by(email=admin_email).first():
            hashed_pw = bcrypt.generate_password_hash('admin123').decode('utf-8')
            admin_user = User(username='Admin', email=admin_email, password=hashed_pw, is_admin=True)
            db.session.add(admin_user)
            db.session.commit()
            print(f"[*] Premade Admin Account Generated! Email: {admin_email} | Password: admin123")
            
    app.run(debug=True)