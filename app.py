from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from datetime import datetime

# Initialize the Flask App
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'deepguard_super_secret_key_2026' # Change this to a random string in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepguard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'alert' # Custom category for Tailwind CSS styling

# ==========================================
# DATABASE MODELS
# ==========================================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    
    # Features
    scans_used = db.Column(db.Integer, default=0) # Tracks daily limit
    is_pro = db.Column(db.Boolean, default=False) # Tracks subscription status
    created_at = db.Column(db.DateTime, default=datetime.utcnow) # Tracks when user joined
    
    # Relationship: One User can have Many ScanRecords
    scans = db.relationship('ScanRecord', backref='user', lazy=True)

class ScanRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    scan_date = db.Column(db.DateTime, default=datetime.utcnow)
    result = db.Column(db.String(20), nullable=False) # e.g., 'Deepfake', 'Authentic'
    confidence = db.Column(db.Float, nullable=True) # e.g., 98.5
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ==========================================
# ROUTES
# ==========================================

# Landing Page
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


# Sign Up Route
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

        email_exists = User.query.filter_by(email=email).first()
        if email_exists:
            flash('Email is already registered. Please log in.', 'alert')
            return redirect(url_for('signup'))
            
        username_exists = User.query.filter_by(username=username).first()
        if username_exists:
            flash('Username is already taken. Please choose another.', 'alert')
            return redirect(url_for('signup'))
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! You can now log in.', 'safe')
        return redirect(url_for('login'))
    
    return render_template('signup.html')


# Login Route
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
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password.', 'alert')
            
    return render_template('login.html')


# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


# User Dashboard
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


# Scan History Page
@app.route('/history')
@login_required
def history():
    # Fetch all scans for the current user, newest first
    user_scans = ScanRecord.query.filter_by(user_id=current_user.id).order_by(ScanRecord.scan_date.desc()).all()
    return render_template('history.html', scans=user_scans)


# User Profile
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        flash('Password updated successfully.', 'safe')
        return redirect(url_for('profile'))
        
    return render_template('profile.html')

# User Settings
@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

# Subscription Page
@app.route('/subscription', methods=['GET', 'POST'])
@login_required
def subscription():
    if request.method == 'POST':
        # Upgrade the user to Pro
        current_user.is_pro = True
        db.session.commit()
        flash('Successfully upgraded to DeepGuard Pro!', 'safe')
        return redirect(url_for('subscription'))
        
    return render_template('subscription.html')


# ==========================================
# INITIALIZATION
# ==========================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)