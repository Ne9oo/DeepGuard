from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt

app = Flask(__name__)

# --- CONFIGURATION ---
# The secret key is used to digitally sign the session cookies
app.config['SECRET_KEY'] = 'super-secret-development-key'
# This tells Flask exactly where to create and find the SQLite database file
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepguard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- INITIALIZE EXTENSIONS ---
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
# Tells Flask-Login where to send users if they try to access a protected page without logging in
login_manager.login_view = 'home' 

# --- DATABASE MODELS ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False) # Will store the hashed password
    role = db.Column(db.String(20), default='guest') # 'admin' or 'guest'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- ROUTES ---
@app.route('/')
def home():
    # Public guest page
    return render_template('index.html')

@app.route('/admin')
@login_required 
def admin_dashboard():
    # The secure dashboard for your team
    return render_template('admin.html')

@app.route('/settings')
# Anyone can access settings; the HTML template dynamically changes based on auth status
def settings():
    return render_template('settings.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If the user is already logged in, send them straight to the admin dashboard
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Look up the user in the database by their email
        user = User.query.filter_by(email=email).first()
        
        # Check if the user exists and if the hashed password matches
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid email or password. Please try again.', 'error')
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # We will add the database logic to create the user here later
        pass
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user() # Kills the session cookie
    return redirect(url_for('home')) # Sends them back to the public dashboard

# --- INITIALIZATION SCRIPT ---
if __name__ == '__main__':
    with app.app_context():
        # This creates the database tables based on the User class above
        db.create_all()
        
        # Check if the database is empty. If it is, create default admin accounts for the team.
        if not User.query.filter_by(email='afiq@deepguard.com').first():
            # Hash the default password "admin123"
            hashed_pw = bcrypt.generate_password_hash('admin123').decode('utf-8')
            
            # Create the team accounts
            afiq = User(email='afiq@deepguard.com', password=hashed_pw, role='admin')
            haris = User(email='haris@deepguard.com', password=hashed_pw, role='admin')
            megat = User(email='megat@deepguard.com', password=hashed_pw, role='admin')
            
            # Add them to the database and save
            db.session.add_all([afiq, haris, megat])
            db.session.commit()
            print("Database initialized! Default team admin accounts created.")

    app.run(debug=True)