import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import stripe
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from fpdf import FPDF
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ==========================================
# ENVIRONMENT & AI MODEL LOADING
# ==========================================
from deepguard_model import load_detector, analyze_file
from forensic_report import (sha256_file, probe_audio, build_evidence, build_report_pdf)
MODEL_PATH = os.path.join(app.root_path, 'best_model.pt')
detector = load_detector(MODEL_PATH)
MODEL_SHA256 = sha256_file(MODEL_PATH)
print("[+] DeepGuard detector loaded.")

# ==========================================
# RATE LIMITER INITIALIZATION
# ==========================================
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# ==========================================
# CONFIGURATION
# ==========================================
_secret = os.environ.get('SECRET_KEY')
if not _secret:
    import secrets
    _secret = secrets.token_hex(32)
    print("[!] WARNING: SECRET_KEY not set in .env - using a temporary random key (logins reset on restart).")
app.config['SECRET_KEY'] = _secret
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepguard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = 'DeepGuard Security <noreply@deepguard.com>'

mail = Mail(app)
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Stripe Configuration
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')

PACKAGES = {
    'single': {'name': 'Single Credit', 'price': 500, 'credits': 1},
    'starter': {'name': 'Starter Bundle', 'price': 2000, 'credits': 5},
    'value': {'name': 'Value Bundle', 'price': 5000, 'credits': 15}
}

# Storage Directories
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
SPECTROGRAM_FOLDER = os.path.join(app.root_path, 'static', 'spectrograms')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'opus', 'm4a', 'aac', 'flac', 'wma', 'amr', 'webm', 'mpeg', 'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)
EVIDENCE_FOLDER = os.path.join(app.root_path, 'evidence')  
os.makedirs(EVIDENCE_FOLDER, exist_ok=True)

# Database & Authentication
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'alert'

# ==========================================
# SYSTEM MAINTENANCE LOCK
# ==========================================
MAINTENANCE_MODE = os.environ.get('MAINTENANCE_MODE') == 'True'

@app.before_request
def check_maintenance_mode():
    if MAINTENANCE_MODE and request.endpoint != 'static':
        return render_template('maintenance.html'), 503

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
    
    # CREDIT SYSTEM 
    credits = db.Column(db.Integer, default=2) 
    scans_used = db.Column(db.Integer, default=0) 
    
    is_admin = db.Column(db.Boolean, default=False)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    scan_mode = db.Column(db.String(20), default='standard') 
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
    msg.body = f'''Welcome to DeepGuard!\nPlease verify your email address to activate your account by clicking the secure link below:\n\n{verify_url}\n\nStay secure,\nThe DeepGuard Team'''
    try:
        mail.send(msg)
    except Exception as e:
        print(f"Email sending failed: {e}")

def send_scan_alert_email(user_email, filename, result, risk):
    msg = Message('DeepGuard: Scan Completed', recipients=[user_email])
    msg.body = f'''Hello!\nYour DeepGuard audio analysis is complete.\nFile: {filename}\nVerdict: {result}\nRisk Level: {risk}\nLog in to your DeepGuard dashboard to view the report.'''
    try:
        mail.send(msg)
    except Exception as e:
        print(f"Scan alert email failed: {e}")

def send_reset_email(user):
    token = s.dumps(user.email, salt='password-reset')
    reset_url = url_for('reset_token', token=token, _external=True)
    msg = Message('DeepGuard: Password Reset Request', recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{reset_url}

If you did not make this request, simply ignore this email and no changes will be made.
'''
    try:
        mail.send(msg)
    except Exception as e:
        print(f"Password reset email failed: {e}")

# ==========================================
# AUDIO FORENSICS ENGINE 
# ==========================================
def generate_spectrogram_image(y, sr):
    """Generates the full visual Mel-spectrogram displayed on the UI dashboard."""
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
    if mode == 'fast':
        duration_limit = 15.0
    elif mode == 'deep' or mode == 'pro':
        duration_limit = 300.0
    else:
        duration_limit = 60.0

    y, sr = librosa.load(filepath, sr=22050, duration=duration_limit)
    spectrogram_b64 = generate_spectrogram_image(y, sr)

    details = analyze_file(detector, filepath, duration_limit)
    fake_prob = details['fake_prob']
    print(f"[debug] {os.path.basename(filepath)} -> P(fake)={fake_prob:.4f}")
        
    result = 'Deepfake' if fake_prob > 0.5 else 'Authentic'
    confidence = round(fake_prob * 100, 1) 
        
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
        'spectrogram': spectrogram_b64,
        'details': details
    }

# ==========================================
# CORE ROUTES
# ==========================================
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/signup', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
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
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            
            # SECURE ADMIN PROMOTION: Auto-grant admin rights to the group email
            if user.email == 'deepguard.my@gmail.com' and not user.is_admin:
                user.is_admin = True
                user.is_verified = True # Auto-verify the admin account
                db.session.commit()
                
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

@app.route('/unverified')
@login_required
def unverified():
    if current_user.is_verified:
        return redirect(url_for('dashboard'))
    return render_template('unverified.html')

@app.route('/resend_verification')
@login_required
@limiter.limit("3 per minute")
def resend_verification():
    if current_user.is_verified:
        return redirect(url_for('dashboard'))
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

@app.route('/reset_password', methods=['GET', 'POST'])
@limiter.limit("3 per minute")
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            send_reset_email(user)
        flash('If an account with that email exists, a password reset link has been sent.', 'safe')
        return redirect(url_for('login'))
    return render_template('reset_request.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    try:
        email = s.loads(token, salt='password-reset', max_age=1800)
    except SignatureExpired:
        flash('The password reset link has expired. Please request a new one.', 'alert')
        return redirect(url_for('reset_request'))
    except BadTimeSignature:
        flash('Invalid password reset link.', 'alert')
        return redirect(url_for('reset_request'))
        
    user = User.query.filter_by(email=email).first_or_404()
    
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if password != confirm_password:
            flash('Passwords do not match.', 'alert')
            return redirect(url_for('reset_token', token=token))
            
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been successfully updated! You can now log in.', 'safe')
        return redirect(url_for('login'))
        
    return render_template('reset_token.html', token=token)

@app.route('/dashboard')
@login_required
@requires_verification
def dashboard():
    return render_template('dashboard.html')

@app.route('/scan', methods=['POST'])
@login_required
@requires_verification
@limiter.limit("3 per minute")
def scan_audio():
    scan_cost = 2 if current_user.scan_mode in ['pro', 'deep'] else 1
    
    # 1. Bypass the insufficient funds check for admins
    if not current_user.is_admin and current_user.credits < scan_cost:
        return jsonify({'success': False, 'message': f'Insufficient credits. This scan requires {scan_cost} credit(s). Please top up your balance.'}), 403

    if 'audio_file' not in request.files:
        return jsonify({'success': False, 'message': 'No audio file found.'}), 400

    file = request.files['audio_file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file format.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}")
    file.save(filepath)
    upload_sha256 = sha256_file(filepath)   
    upload_size = os.path.getsize(filepath)

    try:
        analysis = analyze_audio_forensics(filepath, mode=current_user.scan_mode)
        new_scan = ScanRecord(filename=filename, result=analysis['result'], confidence=analysis['confidence'], user_id=current_user.id)
        
        # 2. Only deduct credits if the user is a regular user
        if not current_user.is_admin:
            current_user.credits -= scan_cost
            
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

        try:
            evidence = build_evidence(
                scan_id=new_scan.id, original_filename=filename, sha256=upload_sha256,
                size_bytes=upload_size, probe=probe_audio(filepath), uploaded_utc=new_scan.scan_date,
                username=current_user.username, scan_mode=current_user.scan_mode,
                file_retained=not current_user.auto_delete, details=analysis['details'],
                result=analysis['result'], confidence=analysis['confidence'],
                risk_level=analysis['risk_level'], model_sha256=MODEL_SHA256)
            with open(os.path.join(EVIDENCE_FOLDER, f"scan_{new_scan.id}.json"), 'w', encoding='utf-8') as ef:
                json.dump(evidence, ef, indent=2)
        except Exception as e:
            print(f"Failed to save evidence record: {e}")

        if current_user.auto_delete and os.path.exists(filepath):
            os.remove(filepath)
            
        if current_user.email_alerts:
            send_scan_alert_email(current_user.email, filename, analysis['result'], analysis['risk_level'])

        return jsonify({
            'success': True, 
            'filename': filename, 
            'result': analysis['result'], 
            'confidence': analysis['confidence'], 
            'risk_level': analysis['risk_level'], 
            'spectrogram': analysis['spectrogram'],
            'credits': current_user.credits
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

    evidence = None
    ev_path = os.path.join(EVIDENCE_FOLDER, f"scan_{scan.id}.json")
    if os.path.exists(ev_path):
        with open(ev_path, encoding='utf-8') as ef:
            evidence = json.load(ef)
    spec_path = os.path.join(SPECTROGRAM_FOLDER, f"spectrogram_{scan.id}.png")
    owner = db.session.get(User, scan.user_id)
    pdf_bytes = build_report_pdf(scan, evidence, owner.username if owner else current_user.username, spec_path)
    return send_file(io.BytesIO(pdf_bytes), as_attachment=True,
                     download_name=f"DeepGuard_Forensic_Report_DG-{scan.id:06d}.pdf", mimetype='application/pdf')

@app.route('/download_evidence/<int:scan_id>')
@login_required
@requires_verification
def download_evidence(scan_id):
    scan = ScanRecord.query.get_or_404(scan_id)
    if scan.user_id != current_user.id and not current_user.is_admin:
        flash('Unauthorized access.', 'alert')
        return redirect(url_for('history'))
    ev_path = os.path.join(EVIDENCE_FOLDER, f"scan_{scan.id}.json")
    if not os.path.exists(ev_path):
        flash('No evidence record exists for this scan.', 'alert')
        return redirect(url_for('history'))
    return send_file(ev_path, as_attachment=True,
                     download_name=f"DeepGuard_Evidence_DG-{scan.id:06d}.json", mimetype='application/json')

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

@app.route('/topup')
@login_required
@requires_verification
def topup():
    return render_template('topup.html')

@app.route('/create-checkout-session', methods=['POST'])
@login_required
@requires_verification
def create_checkout_session():
    package_id = request.form.get('package')
    if package_id not in PACKAGES:
        flash('Invalid package selected.', 'alert')
        return redirect(url_for('topup'))

    package = PACKAGES[package_id]

    try:
        checkout_session = stripe.checkout.Session.create(
            line_items=[{
                'price_data': {
                    'currency': 'myr',
                    'unit_amount': package['price'],
                    'product_data': {
                        'name': f"DeepGuard: {package['name']} ({package['credits']} Credits)",
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('payment_success', package_id=package_id, _external=True) + '&session_id={CHECKOUT_SESSION_ID}',
            cancel_url=url_for('topup', _external=True),
            client_reference_id=str(current_user.id)
        )
        return redirect(checkout_session.url, code=303)
    except Exception as e:
        flash(f"Payment gateway error: {str(e)}", 'alert')
        return redirect(url_for('topup'))

@app.route('/payment-success')
@login_required
@requires_verification
def payment_success():
    session_id = request.args.get('session_id')
    package_id = request.args.get('package_id')

    if not session_id or package_id not in PACKAGES:
        flash("Invalid payment session.", 'alert')
        return redirect(url_for('topup'))

    try:
        session = stripe.checkout.Session.retrieve(session_id)
        
        if session.payment_status == 'paid' and session.client_reference_id == str(current_user.id):
            credits_to_add = PACKAGES[package_id]['credits']
            current_user.credits += credits_to_add
            db.session.commit()
            
            flash(f"Payment successful! {credits_to_add} credits have been added to your account.", 'safe')
        else:
            flash("Payment could not be verified.", 'alert')
            
        return redirect(url_for('topup'))
    except Exception as e:
        flash("An error occurred verifying your payment.", 'alert')
        return redirect(url_for('topup'))

@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    user = current_user
    
    # 1. Find all scans belonging to this user
    user_scans = ScanRecord.query.filter_by(user_id=user.id).all()
    
    # 2. Loop through and delete the physical files from the server
    for scan in user_scans:
        spec_path = os.path.join(SPECTROGRAM_FOLDER, f"spectrogram_{scan.id}.png")
        if os.path.exists(spec_path):
            os.remove(spec_path)
            
        ev_path = os.path.join(EVIDENCE_FOLDER, f"scan_{scan.id}.json")
        if os.path.exists(ev_path):
            os.remove(ev_path)
            
    # 3. Delete the scan records from the database
    ScanRecord.query.filter_by(user_id=user.id).delete()
    
    # 4. Delete the user account itself
    db.session.delete(user)
    db.session.commit()
    
    # 5. Log them out and redirect to home
    logout_user()
    flash('Your account and all associated data have been permanently deleted.', 'safe')
    return redirect(url_for('index'))

@app.route('/admin')
@login_required
@requires_verification
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access Denied.', 'alert')
        return redirect(url_for('dashboard'))
    
    all_users = User.query.all()
    recent_scans = ScanRecord.query.order_by(ScanRecord.scan_date.desc()).limit(10).all()

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    traffic_labels = []
    traffic_data = []

    for i in range(6, -1, -1):
        target_date = today - timedelta(days=i)
        next_date = target_date + timedelta(days=1)
        traffic_labels.append(target_date.strftime('%a')) 
        count = ScanRecord.query.filter(
            ScanRecord.scan_date >= target_date,
            ScanRecord.scan_date < next_date
        ).count()
        traffic_data.append(count)

    return render_template('admin.html', users=all_users, scans=recent_scans, traffic_labels=traffic_labels, traffic_data=traffic_data)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# ==========================================
# DATABASE INITIALIZATION & STARTUP
# ==========================================
with app.app_context():
    db.create_all()
    # The hardcoded admin123 account has been permanently removed for security.

if __name__ == '__main__':
    app.run(debug=True)