from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt

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
    
    # Features requested by supervisor
    scans_used = db.Column(db.Integer, default=0) # Tracks daily limit
    is_pro = db.Column(db.Boolean, default=False) # Tracks subscription status

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ==========================================
# ROUTES
# ==========================================

# Landing Page
@app.route('/')
def index():
    # If the user is already logged in, they probably want to see the dashboard, not the landing page.
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

        # 1. Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match.', 'alert')
            return redirect(url_for('signup'))

        # 2. Check if email or username already exists
        email_exists = User.query.filter_by(email=email).first()
        if email_exists:
            flash('Email is already registered. Please log in.', 'alert')
            return redirect(url_for('signup'))
            
        username_exists = User.query.filter_by(username=username).first()
        if username_exists:
            flash('Username is already taken. Please choose another.', 'alert')
            return redirect(url_for('signup'))
        
        # 3. Hash the password and create the user
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
        
        # Find the user by email
        user = User.query.filter_by(email=email).first()
        
        # Check if user exists and password matches
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


# User Dashboard (Protected Route)
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


# ==========================================
# INITIALIZATION
# ==========================================
if __name__ == '__main__':
    # Creates the deepguard.db file and User table if it doesn't exist yet
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)