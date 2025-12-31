import numpy as np
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
from datetime import datetime, timedelta
import hashlib
import sqlite3
import os
from faker import Faker
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pytz
try:
    from meteostat import Point, Daily, Hourly
except ModuleNotFoundError:
    Point = Daily = Hourly = None

import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Solaris Power Analytics",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 24px !important;
        color: #1C83E1;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 8px;
    }
    .sidebar .sidebar-content {
        background-color: #F0F2F6;
    }
    .prediction-card {
        background-color: #E6F7FF;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .alert-card {
        background-color: #FFF3CD;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .critical-alert {
        background-color: #F8D7DA;
        border-left: 5px solid #DC3545;
    }
    .warning-alert {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
    }
    .info-alert {
        background-color: #D1ECF1;
        border-left: 5px solid #17A2B8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Faker for generating fake data
fake = Faker()

# Database Setup
def init_db():
    conn = sqlite3.connect('solaris.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY,
                  password TEXT,
                  name TEXT,
                  email TEXT,
                  role TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # User plants junction table
    c.execute('''CREATE TABLE IF NOT EXISTS user_plants
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  plant_name TEXT,
                  FOREIGN KEY(username) REFERENCES users(username))''')
    
    # Plants table
    c.execute('''CREATE TABLE IF NOT EXISTS plants
                 (name TEXT PRIMARY KEY,
                  location TEXT,
                  capacity REAL,
                  commission_date TEXT,
                  panels INTEGER,
                  efficiency REAL,
                  latitude REAL,
                  longitude REAL)''')
    
    # Maintenance table
    c.execute('''CREATE TABLE IF NOT EXISTS maintenance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  plant_name TEXT,
                  type TEXT,
                  scheduled_date TEXT,
                  completed_date TEXT,
                  technician TEXT,
                  priority TEXT,
                  status TEXT,
                  notes TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(plant_name) REFERENCES plants(name))''')
    
    # Alerts table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  plant_name TEXT,
                  type TEXT,
                  severity TEXT,
                  message TEXT,
                  timestamp TEXT,
                  resolved BOOLEAN DEFAULT 0,
                  resolved_at TEXT,
                  resolved_by TEXT,
                  FOREIGN KEY(plant_name) REFERENCES plants(name))''')
    
    # Predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  plant_name TEXT,
                  timestamp TEXT,
                  predicted_power REAL,
                  actual_power REAL,
                  inputs TEXT)''')
    
    # Check if default admin exists
    c.execute("SELECT COUNT(*) FROM users WHERE username='admin'")
    if c.fetchone()[0] == 0:
        hashed_pw = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)", 
                 ("admin", hashed_pw, "Administrator", "admin@solarispower.com", "admin", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    
    # Check if plants exist
    c.execute("SELECT COUNT(*) FROM plants")
    if c.fetchone()[0] == 0:
        default_plants = [
            ("Plant A", "California, USA", 50.0, "2020-05-15", 12500, 22.5, 34.0522, -118.2437),
            ("Plant B", "Arizona, USA", 75.0, "2021-03-22", 18750, 21.8, 33.4484, -112.0740),
            ("Plant C", "Nevada, USA", 100.0, "2022-01-10", 25000, 23.1, 36.1699, -115.1398)
        ]
        c.executemany("INSERT INTO plants VALUES (?, ?, ?, ?, ?, ?, ?, ?)", default_plants)
        
        # Assign all plants to admin
        for plant in default_plants:
            c.execute("INSERT INTO user_plants (username, plant_name) VALUES (?, ?)", 
                     ("admin", plant[0]))
        conn.commit()
    
    conn.close()

# Initialize database
init_db()

# User Authentication System
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def get_user_plants(username):
    conn = sqlite3.connect('solaris.db')
    c = conn.cursor()
    c.execute("SELECT plant_name FROM user_plants WHERE username=?", (username,))
    plants = [row[0] for row in c.fetchall()]
    conn.close()
    return plants

def get_plant_details(plant_name):
    conn = sqlite3.connect('solaris.db')
    c = conn.cursor()
    c.execute("SELECT * FROM plants WHERE name=?", (plant_name,))
    plant = c.fetchone()
    conn.close()
    if plant:
        return {
            "name": plant[0],
            "location": plant[1],
            "capacity": plant[2],
            "commission_date": plant[3],
            "panels": plant[4],
            "efficiency": plant[5],
            "latitude": plant[6],
            "longitude": plant[7]
        }
    return None

def get_all_plants():
    conn = sqlite3.connect('solaris.db')
    c = conn.cursor()
    c.execute("SELECT name FROM plants")
    plants = [row[0] for row in c.fetchall()]
    conn.close()
    return plants

# Solar Plant Data Generation with Real Weather Data
def generate_solar_data(plant_name, days=7):
    plant = get_plant_details(plant_name)
    if not plant:
        return pd.DataFrame()
    
    # Get historical weather data
    location = Point(plant['latitude'], plant['longitude'])
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        weather_data = Hourly(location, start_date, end_date).fetch()
        weather_data = weather_data.reset_index()
    except:
        # Fallback to random data if weather API fails
        weather_data = pd.DataFrame({
            'time': pd.date_range(end=end_date, periods=days*24, freq='H'),
            'temp': np.random.uniform(15, 35, days*24),
            'rhum': np.random.uniform(30, 80, days*24),
            'wspd': np.random.uniform(0.5, 8.5, days*24),
            'pres': np.random.uniform(980, 1040, days*24),
        })
    
    # Generate realistic solar data based on weather
    data = []
    base_power = plant['capacity'] * 0.7  # Typical capacity factor
    
    for idx, row in weather_data.iterrows():
        timestamp = row['time'] if 'time' in row else row['time']
        temp = row['temp'] if 'temp' in row else np.random.uniform(15, 35)
        humidity = row['rhum'] if 'rhum' in row else np.random.uniform(30, 80)
        wind_speed = row['wspd'] if 'wspd' in row else np.random.uniform(0.5, 8.5)
        pressure = row['pres'] if 'pres' in row else np.random.uniform(980, 1040)
        
        # Calculate solar position (simplified)
        hour = timestamp.hour
        solar_noon = 12
        distance_to_noon = abs(hour - solar_noon)
        
        # Base generation with some daily variation
        hour_factor = 0.3 + 0.7 * np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else 0.3
        
        # Temperature effect (optimal around 25°C)
        temp_effect = 1 - 0.005 * abs(temp - 25)
        
        # Humidity effect
        humidity_effect = 1 - 0.002 * (humidity - 50)
        
        # Wind effect (cooling can help)
        wind_effect = 1 + 0.01 * wind_speed if wind_speed < 10 else 1 - 0.02 * (wind_speed - 10)
        
        # Cloud cover estimation (simplified)
        cloud_cover = 0.5 * (1 - humidity/100) + 0.3 * (1 - pressure/1013) + np.random.uniform(-0.1, 0.1)
        cloud_cover = max(0, min(1, cloud_cover))
        
        # Final generation calculation
        generation = base_power * hour_factor * temp_effect * humidity_effect * wind_effect * (1 - 0.3 * cloud_cover)
        generation = max(0, generation * (0.95 + np.random.random() * 0.1))  # Add some randomness
        
        # Efficiency calculation
        base_eff = plant['efficiency']
        eff = base_eff * temp_effect * (1 - 0.2 * cloud_cover) * (0.95 + np.random.random() * 0.1)
        eff = max(5, min(base_eff, eff))
        
        data.append({
            "Plant": plant_name,
            "Timestamp": timestamp,
            "Power (MW)": round(generation, 2),
            "Efficiency (%)": round(eff, 1),
            "Temperature (°C)": round(temp, 1),
            "Humidity (%)": round(humidity, 1),
            "Wind Speed (m/s)": round(wind_speed, 1),
            "Pressure (hPa)": round(pressure, 1),
            "Cloud Cover": round(cloud_cover, 2),
            "Irradiance (W/m²)": round(800 * (1 - cloud_cover) * hour_factor, 1)
        })
    
    return pd.DataFrame(data)

# ML Model - Random Forest for demonstration
def train_model():
    # Generate synthetic training data
    np.random.seed(42)
    X = np.random.rand(1000, 5)  # 5 features
    y = 50 * X[:,0] + 30 * (1 - X[:,1]) + 20 * X[:,2] - 15 * X[:,3] + 10 * X[:,4] + np.random.randn(1000) * 5
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model and scalers
    joblib.dump(model, 'solar_power_model.joblib')
    
    # Create and save scalers (for demonstration)
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    joblib.dump(scaler_X, 'scaler_X.joblib')
    joblib.dump(scaler_y, 'scaler_y.joblib')
    
    return model

@st.cache_resource
def load_model():
    if not os.path.exists('solar_power_model.joblib'):
        return train_model()
    try:
        return joblib.load('solar_power_model.joblib')
    except:
        return train_model()

loaded_model = load_model()

# Prediction function
def predict_power_generation(input_data):
    try:
        # Load scalers
        scaler_X = joblib.load('scaler_X.joblib')
        scaler_y = joblib.load('scaler_y.joblib')
        
        # Standardize input data
        input_array = np.asarray(input_data, dtype=float).reshape(1, -1)
        input_scaled = scaler_X.transform(input_array)
        
        prediction = loaded_model.predict(input_scaled)
        prediction = scaler_y.inverse_transform(prediction.reshape(-1, 1))
        return max(0, prediction[0][0])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Alert System
def create_alert(plant_name, alert_type, severity, message):
    conn = sqlite3.connect('solaris.db')
    c = conn.cursor()
    c.execute("INSERT INTO alerts (plant_name, type, severity, message, timestamp) VALUES (?, ?, ?, ?, ?)",
              (plant_name, alert_type, severity, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_alerts(plant_name=None, resolved=False):
    conn = sqlite3.connect('solaris.db')
    c = conn.cursor()
    
    # Convert Python boolean to SQLite integer (1/0)
    resolved_int = 1 if resolved else 0
    
    if plant_name:
        c.execute("SELECT * FROM alerts WHERE plant_name=? AND resolved=? ORDER BY timestamp DESC", 
                 (plant_name, resolved_int))
    else:
        c.execute("SELECT * FROM alerts WHERE resolved=? ORDER BY timestamp DESC", 
                 (resolved_int,))
    
    alerts = []
    for row in c.fetchall():
        alerts.append({
            "id": row[0],
            "plant_name": row[1],
            "type": row[2],
            "severity": row[3],
            "message": row[4],
            "timestamp": row[5],
            "resolved": bool(row[6]),  # Convert back to Python boolean when reading
            "resolved_at": row[7],
            "resolved_by": row[8]
        })
    
    conn.close()
    return alerts

# Login Page
def login_page():
    st.markdown('<div class="header">Solaris Power Analytics</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login_form"):
            st.markdown('<div class="subheader">Login to Your Account</div>', unsafe_allow_html=True)
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                conn = sqlite3.connect('solaris.db')
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE username=?", (username,))
                user_data = c.fetchone()
                conn.close()
                
                if user_data and check_hashes(password, user_data[1]):
                    st.session_state.logged_in = True
                    st.session_state.user = {
                        "username": user_data[0],
                        "name": user_data[2],
                        "email": user_data[3],
                        "role": user_data[4]
                    }
                    st.session_state.user['plants'] = get_user_plants(username)
                    st.session_state.current_plant = st.session_state.user['plants'][0] if st.session_state.user['plants'] else None
                    
                    # Initialize plant data
                    if 'plant_data' not in st.session_state:
                        st.session_state.plant_data = {}
                        for plant in st.session_state.user['plants']:
                            st.session_state.plant_data[plant] = generate_solar_data(plant, days=30)
                    
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

# Main Application
def main_app():
    # Navigation Menu
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/024/056/716/small/sun-or-sunrise-with-solar-panel-for-renewable-power-energy-logo-design-vector.jpg", width=150)
        st.markdown(f"### Welcome, {st.session_state.user['name']}")
        
        # Check for unresolved alerts
        if st.session_state.user['role'] in ['admin', 'operator']:
            alerts = get_alerts()
            if alerts:
                st.markdown(f"⚠️ **{len(alerts)} active alerts**", help=f"{len([a for a in alerts if a['severity'] == 'critical'])} critical")
        
        # Plant selection for users with multiple plants
        if len(st.session_state.user['plants']) > 1:
            st.session_state.current_plant = st.selectbox(
                "Select Plant", 
                st.session_state.user['plants'],
                index=0
            )
        elif st.session_state.user['plants']:
            st.session_state.current_plant = st.session_state.user['plants'][0]
            st.markdown(f"**Plant:** {st.session_state.current_plant}")
        else:
            st.session_state.current_plant = None
            st.warning("No plants assigned")
        
        menu_options = {
            "Dashboard": "speedometer2",
            "Power Prediction": "lightning-charge",
            "Historical Data": "graph-up",
            "Plant Analytics": "sun",
            "Alerts": "exclamation-triangle",
            "Maintenance": "tools",
            "Reports": "file-earmark-text",
            "User Management": "people",
            "Account Settings": "person-circle"
        }
        
        # Hide admin-only features from non-admin users
        if st.session_state.user['role'] != 'admin':
            del menu_options["User Management"]
            if st.session_state.user['role'] != 'operator':
                del menu_options["Maintenance"]
        
        selected = option_menu(
            menu_title=None,
            options=list(menu_options.keys()),
            icons=list(menu_options.values()),
            default_index=0,
            styles={
                "container": {"padding": "0!important"},
                "nav-link": {"font-size": "16px", "margin": "5px 0"}
            }
        )
        
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.username = None
            st.rerun()
    
    # Dashboard Page
    if selected == "Dashboard":
        st.markdown('<div class="header">Plant Performance Dashboard</div>', unsafe_allow_html=True)
        
        if not st.session_state.current_plant:
            st.warning("No plant selected. Please contact admin to assign a plant.")
            return
        
        plant_info = get_plant_details(st.session_state.current_plant)
        
        # Plant info row
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            st.markdown(f"### {st.session_state.current_plant}")
            st.markdown(f"**Location:** {plant_info.get('location', 'N/A')}")
            st.markdown(f"**Capacity:** {plant_info.get('capacity', 'N/A'):.1f} MW")
            st.markdown(f"**Commission Date:** {plant_info.get('commission_date', 'N/A')}")
        
        # Metrics Row
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        # Get recent data for the current plant
        plant_df = st.session_state.plant_data.get(st.session_state.current_plant, pd.DataFrame())
        if not plant_df.empty:
            latest_data = plant_df[plant_df['Timestamp'] == plant_df['Timestamp'].max()].iloc[0]
            avg_power = plant_df['Power (MW)'].mean()
            avg_eff = plant_df['Efficiency (%)'].mean()
            total_gen = plant_df['Power (MW)'].sum()
            
            # Determine weather condition based on irradiance and cloud cover
            irradiance = latest_data['Irradiance (W/m²)']
            if irradiance > 800:
                weather = "Sunny"
            elif irradiance > 500:
                weather = "Partly Cloudy"
            else:
                weather = "Cloudy"
            
            with col1:
                st.markdown(f'<div class="metric-card">Current Output<br><span style="font-size:28px">{latest_data["Power (MW)"]:.1f} MW</span></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card">Daily Average<br><span style="font-size:28px">{avg_power:.1f} MW</span></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card">System Efficiency<br><span style="font-size:28px">{avg_eff:.1f}%</span></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card">Weather Condition<br><span style="font-size:28px">{weather}</span></div>', unsafe_allow_html=True)
        
        # Charts Row
        st.markdown('<div class="subheader">Performance Overview</div>', unsafe_allow_html=True)
        
        if not plant_df.empty:
            # Daily summary
            plant_df['Date'] = plant_df['Timestamp'].dt.date
            daily_df = plant_df.groupby('Date').agg({
                'Power (MW)': 'sum',
                'Efficiency (%)': 'mean',
                'Temperature (°C)': 'mean',
                'Irradiance (W/m²)': 'mean',
                'Cloud Cover': 'mean'
            }).reset_index()
            
            tab1, tab2, tab3 = st.tabs(["Generation", "Efficiency", "Environmental"])
            
            with tab1:
                fig_daily = px.line(daily_df, x="Date", y="Power (MW)", 
                                  title="Daily Power Generation (MWh)")
                st.plotly_chart(fig_daily, use_container_width=True)
                
                # Hourly generation for the last 3 days
                last_3_days = plant_df[plant_df['Timestamp'] >= plant_df['Timestamp'].max() - timedelta(days=3)]
                fig_hourly = px.line(last_3_days, x="Timestamp", y="Power (MW)", 
                                   color=last_3_days['Timestamp'].dt.date.astype(str),
                                   title="Hourly Power Generation (Last 3 Days)")
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with tab2:
                fig_eff = px.line(daily_df, x="Date", y="Efficiency (%)", 
                                title="Daily Average Efficiency")
                st.plotly_chart(fig_eff, use_container_width=True)
                
                # Efficiency vs. Temperature
                fig_eff_temp = px.scatter(plant_df.sample(n=min(500, len(plant_df))), x="Temperature (°C)", y="Efficiency (%)",
                          trendline="lowess", title="Efficiency vs. Temperature")


                st.plotly_chart(fig_eff_temp, use_container_width=True)
            
            with tab3:
                fig_temp = px.line(daily_df, x="Date", y="Temperature (°C)", 
                                 title="Daily Average Temperature")
                st.plotly_chart(fig_temp, use_container_width=True)
                
                fig_irrad = px.line(daily_df, x="Date", y="Irradiance (W/m²)", 
                                   title="Daily Average Solar Irradiance")
                st.plotly_chart(fig_irrad, use_container_width=True)
    
    # Power Prediction Page
    elif selected == "Power Prediction":
        st.markdown('<div class="header">Power Generation Predictor</div>', unsafe_allow_html=True)

        if not st.session_state.current_plant:
            st.warning("No plant selected. Please contact admin to assign a plant.")
            return

        # Input form
        with st.form("solar_form"):
            col1, col2 = st.columns(2)

            with col1:
                distance_to_solar_noon = st.number_input("Distance to Solar Noon (radians)", min_value=0.0, max_value=3.14, value=0.0)
                temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
                wind_direction = st.number_input("Wind Direction (degrees)", min_value=0.0, max_value=360.0, value=180.0)
                wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=3.0)

            with col2:
                sky_cover = st.number_input("Sky Cover (0–1)", min_value=0.0, max_value=1.0, value=0.0)
                visibility = st.number_input("Visibility (km)", min_value=0.0, max_value=50.0, value=10.0)
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
                average_wind_speed = st.number_input("Average Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=2.0)
                average_pressure = st.number_input("Average Pressure (inHg)", min_value=28.0, max_value=32.0, value=29.9)

            submit = st.form_submit_button("Predict Power Generation")

        # Prediction logic
        if submit:
            try:
                input_data = [
                    distance_to_solar_noon,
                    temperature,
                    wind_direction,
                    wind_speed,
                    sky_cover,
                    visibility,
                    humidity,
                    average_wind_speed,
                    average_pressure
                ]
                
                prediction = predict_power_generation(input_data)
                
                if prediction is not None:
                    # Save prediction to database
                    conn = sqlite3.connect('solaris.db')
                    c = conn.cursor()
                    c.execute("INSERT INTO predictions (plant_name, timestamp, predicted_power, inputs) VALUES (?, ?, ?, ?)",
                              (st.session_state.current_plant, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                               prediction, str(input_data)))
                    conn.commit()
                    conn.close()
                    
                    predicted_mw = prediction / 1000 
                    st.success(f"🔋 Estimated Solar Power Generated: **{predicted_mw:.2f} MW**")
                    
                    # Get plant capacity for context
                    plant_info = get_plant_details(st.session_state.current_plant)
                    if plant_info:
                        plant_capacity_mw = float(plant_info['capacity'])
                        capacity_utilization = (predicted_mw / plant_info['capacity']) * 100
                        st.metric("Capacity Utilization", f"{capacity_utilization:.1f}%")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    # Historical Data Page
    elif selected == "Historical Data":
        st.markdown('<div class="header">Prediction History</div>', unsafe_allow_html=True)
        
        conn = sqlite3.connect('solaris.db')
        c = conn.cursor()
        
        # Get predictions for current user's plants
        if st.session_state.user['role'] == 'admin':
            c.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
        else:
            placeholders = ','.join(['?']*len(st.session_state.user['plants']))
            c.execute(f"SELECT * FROM predictions WHERE plant_name IN ({placeholders}) ORDER BY timestamp DESC", 
                     tuple(st.session_state.user['plants']))
        
        predictions = []
        for row in c.fetchall():
            predictions.append({
                "id": row[0],
                "plant_name": row[1],
                "timestamp": row[2],
                "predicted_power": row[3],
                "actual_power": row[4],
                "inputs": eval(row[5]) if row[5] else None
            })
        
        conn.close()
        
        if not predictions:
            st.info("No predictions recorded yet. Use the Power Prediction tool to get started.")
        else:
            # Convert predictions to DataFrame
            history_data = []
            for pred in predictions:
                if pred['plant_name'] == st.session_state.current_plant or st.session_state.user['role'] == 'admin':
                    record = {
                        "Timestamp": pred["timestamp"],
                        "Plant": pred["plant_name"],
                        "Prediction (MW)": pred["predicted_power"],
                        "Actual (MW)": pred["actual_power"] if pred["actual_power"] else "N/A",
                        "Temperature (°C)": pred["inputs"][1] if pred["inputs"] else "N/A",
                        "Cloud Cover": f"{pred['inputs'][4]*100:.1f}%" if pred["inputs"] else "N/A"
                    }
                    history_data.append(record)
            
            df_history = pd.DataFrame(history_data)
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                time_filter = st.selectbox("Time Range", ["Last 7 days", "Last 30 days", "Last 90 days", "All time"])
            with col2:
                if st.session_state.user['role'] == 'admin':
                    plant_filter = st.selectbox("Plant", ["All"] + get_all_plants())
                else:
                    st.markdown(f"**Plant:** {st.session_state.current_plant}")
                    plant_filter = st.session_state.current_plant
            
            # Apply filters
            if time_filter == "Last 7 days":
                cutoff = datetime.now() - timedelta(days=7)
                df_filtered = df_history[pd.to_datetime(df_history['Timestamp']) >= cutoff]
            elif time_filter == "Last 30 days":
                cutoff = datetime.now() - timedelta(days=30)
                df_filtered = df_history[pd.to_datetime(df_history['Timestamp']) >= cutoff]
            elif time_filter == "Last 90 days":
                cutoff = datetime.now() - timedelta(days=90)
                df_filtered = df_history[pd.to_datetime(df_history['Timestamp']) >= cutoff]
            else:
                df_filtered = df_history.copy()
            
            if plant_filter != "All":
                df_filtered = df_filtered[df_filtered['Plant'] == plant_filter]
            
            # Show data and charts
            st.dataframe(df_filtered, use_container_width=True, hide_index=True)
            
            st.markdown('<div class="subheader">Prediction Trends</div>', unsafe_allow_html=True)
            
            if not df_filtered.empty:
                tab1, tab2 = st.tabs(["Over Time", "Accuracy Analysis"])
                
                with tab1:
                    fig_history = px.line(df_filtered, x="Timestamp", y="Prediction (MW)", 
                                        color="Plant" if plant_filter == "All" else None,
                                        title="Power Generation Predictions Over Time")
                    st.plotly_chart(fig_history, use_container_width=True)
                
                with tab2:
                    # Only show if we have actual values to compare
                    if 'Actual (MW)' in df_filtered.columns and not all(df_filtered['Actual (MW)'] == "N/A"):
                        df_filtered['Actual (MW)'] = pd.to_numeric(df_filtered['Actual (MW)'], errors='coerce')
                        df_filtered = df_filtered.dropna(subset=['Actual (MW)'])
                        
                        if not df_filtered.empty:
                            df_filtered['Error (%)'] = 100 * (df_filtered['Prediction (MW)'] - df_filtered['Actual (MW)']) / df_filtered['Actual (MW)']
                            
                            fig_error = px.scatter(df_filtered, x="Timestamp", y="Error (%)",
                                                 trendline="lowess",
                                                 title="Prediction Error Over Time")
                            st.plotly_chart(fig_error, use_container_width=True)
                            
                            st.metric("Mean Absolute Error", f"{abs(df_filtered['Error (%)']).mean():.1f}%")
                        else:
                            st.warning("No actual power data available for accuracy analysis")
                    else:
                        st.warning("No actual power data available for accuracy analysis")
    
    # Plant Analytics Page
    elif selected == "Plant Analytics":
        st.markdown('<div class="header">Plant Performance Analytics</div>', unsafe_allow_html=True)
        
        if not st.session_state.current_plant:
            st.warning("No plant selected. Please contact admin to assign a plant.")
            return
        
        plant_df = st.session_state.plant_data.get(st.session_state.current_plant, pd.DataFrame())
        
        if plant_df.empty:
            st.error("No data available for the selected plant.")
            return
        
        st.markdown('<div class="subheader">Performance Metrics</div>', unsafe_allow_html=True)
        
        # Time period selection
        col1, col2 = st.columns(2)
        with col1:
            time_period = st.selectbox("Analysis Period", 
                                     ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"])
        with col2:
            if time_period == "Custom":
                date_range = st.date_input("Select Date Range", 
                                         [datetime.now() - timedelta(days=7), datetime.now()],
                                         max_value=datetime.now())
        
        # Filter data based on selection
        if time_period == "Last 7 days":
            cutoff = datetime.now() - timedelta(days=7)
            analysis_df = plant_df[plant_df['Timestamp'] >= cutoff]
        elif time_period == "Last 30 days":
            cutoff = datetime.now() - timedelta(days=30)
            analysis_df = plant_df[plant_df['Timestamp'] >= cutoff]
        elif time_period == "Last 90 days":
            cutoff = datetime.now() - timedelta(days=90)
            analysis_df = plant_df[plant_df['Timestamp'] >= cutoff]
        elif time_period == "Custom" and len(date_range) == 2:
            analysis_df = plant_df[
                (plant_df['Timestamp'].dt.date >= date_range[0]) & 
                (plant_df['Timestamp'].dt.date <= date_range[1])
            ]
        else:
            analysis_df = plant_df.copy()
        
        if analysis_df.empty:
            st.warning("No data available for the selected time period.")
            return
        
        # Summary statistics
        st.markdown("### Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_power = analysis_df['Power (MW)'].mean()
            st.metric("Average Power Output", f"{avg_power:.1f} MW")
        
        with col2:
            max_power = analysis_df['Power (MW)'].max()
            st.metric("Peak Power Output", f"{max_power:.1f} MW")
        
        with col3:
            avg_eff = analysis_df['Efficiency (%)'].mean()
            st.metric("Average Efficiency", f"{avg_eff:.1f}%")
        
        with col4:
            total_gen = analysis_df['Power (MW)'].sum()
            st.metric("Total Generation", f"{total_gen:.0f} MWh")
        
        # Detailed analysis tabs
        tab1, tab2, tab3 = st.tabs(["Performance Trends", "Correlation Analysis", "Anomaly Detection"])
        
        with tab1:
            st.markdown("### Performance Over Time")
            
            # Resample to daily data
            daily_df = analysis_df.set_index('Timestamp').resample('D').agg({
                'Power (MW)': 'mean',
                'Efficiency (%)': 'mean',
                'Temperature (°C)': 'mean',
                'Irradiance (W/m²)': 'mean'
            }).reset_index()
            
            fig_daily = px.line(daily_df, x="Timestamp", y=["Power (MW)", "Efficiency (%)"],
                              title="Daily Performance Metrics",
                              labels={"value": "Metric Value", "variable": "Metric"})
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Hourly pattern
            analysis_df['Hour'] = analysis_df['Timestamp'].dt.hour
            hourly_avg = analysis_df.groupby('Hour').agg({
                'Power (MW)': 'mean',
                'Efficiency (%)': 'mean'
            }).reset_index()
            
            fig_hourly = px.line(hourly_avg, x="Hour", y=["Power (MW)", "Efficiency (%)"],
                               title="Average Hourly Pattern",
                               labels={"value": "Metric Value", "variable": "Metric"})
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with tab2:
            st.markdown("### Parameter Correlations")
            
            param = st.selectbox("Analyze correlation with:",
                               ["Temperature (°C)", "Irradiance (W/m²)", "Wind Speed (m/s)", "Cloud Cover"])
            
            fig_corr = px.scatter(analysis_df.sample(n=min(500, len(analysis_df))), x=param, y="Power (MW)",
                      trendline="lowess", title=f"Power Output vs. {param}")


            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlation matrix
            st.markdown("#### Correlation Matrix")
            corr_df = analysis_df[['Power (MW)', 'Efficiency (%)', 'Temperature (°C)', 
                                 'Irradiance (W/m²)', 'Wind Speed (m/s)', 'Cloud Cover']].corr()
            fig_heatmap = px.imshow(corr_df, text_auto=True, aspect="auto",
                                  title="Parameter Correlation Matrix")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab3:
            st.markdown("### Performance Anomalies")
            
            # Simple anomaly detection (Z-score)
            analysis_df['Power_zscore'] = (analysis_df['Power (MW)'] - analysis_df['Power (MW)'].mean()) / analysis_df['Power (MW)'].std()
            anomalies = analysis_df[abs(analysis_df['Power_zscore']) > 2.5]
            
            if not anomalies.empty:
                st.warning(f"Found {len(anomalies)} potential anomalies (outside 2.5σ)")
                
                fig_anomalies = px.scatter(analysis_df, x="Timestamp", y="Power (MW)",
                                         color=abs(analysis_df['Power_zscore']) > 2.5,
                                         title="Power Output with Anomalies Highlighted",
                                         labels={"color": "Anomaly"})
                st.plotly_chart(fig_anomalies, use_container_width=True)
                
                st.dataframe(anomalies[['Timestamp', 'Power (MW)', 'Temperature (°C)', 
                                      'Irradiance (W/m²)', 'Wind Speed (m/s)', 'Cloud Cover']],
                            use_container_width=True, hide_index=True)
                
                # Option to create alert for anomalies
                if st.button("Create Alert for Anomalies"):
                    for _, row in anomalies.iterrows():
                        message = f"Power output anomaly detected: {row['Power (MW)']:.1f} MW at {row['Timestamp']}"
                        create_alert(st.session_state.current_plant, "Performance", 
                                    "critical" if abs(row['Power_zscore']) > 3 else "warning", 
                                    message)
                    st.success(f"Created {len(anomalies)} alerts for anomalies")
            else:
                st.success("No significant anomalies detected in the selected period.")
    
    # Alerts Page
    elif selected == "Alerts":
        st.markdown('<div class="header">Plant Alerts</div>', unsafe_allow_html=True)
        
        if st.session_state.user['role'] not in ['admin', 'operator']:
            st.warning("You don't have permission to access this page.")
            return
        
        tab1, tab2 = st.tabs(["Active Alerts", "Alert History"])
        
        with tab1:
            st.markdown("### Active Alerts")
            
            if st.session_state.user['role'] == 'admin':
                alerts = get_alerts()
            else:
                alerts = get_alerts(st.session_state.current_plant)
            
            if not alerts:
                st.success("No active alerts")
            else:
                critical_alerts = [a for a in alerts if a['severity'] == 'critical']
                warning_alerts = [a for a in alerts if a['severity'] == 'warning']
                info_alerts = [a for a in alerts if a['severity'] == 'info']
                
                if critical_alerts:
                    st.markdown("#### Critical Alerts")
                    for alert in critical_alerts:
                        with st.container():
                            st.markdown(f'<div class="alert-card critical-alert">', unsafe_allow_html=True)
                            st.markdown(f"**{alert['type']}** - {alert['plant_name']}")
                            st.markdown(f"*{alert['timestamp']}*")
                            st.markdown(alert['message'])
                            
                            if st.button("Resolve", key=f"resolve_{alert['id']}"):
                                conn = sqlite3.connect('solaris.db')
                                c = conn.cursor()
                                c.execute("UPDATE alerts SET resolved=1, resolved_at=?, resolved_by=? WHERE id=?",
                                          (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                           st.session_state.user['name'], 
                                           alert['id']))
                                conn.commit()
                                conn.close()
                                st.success("Alert resolved")
                                st.rerun()
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                
                if warning_alerts:
                    st.markdown("#### Warning Alerts")
                    for alert in warning_alerts:
                        with st.container():
                            st.markdown(f'<div class="alert-card warning-alert">', unsafe_allow_html=True)
                            st.markdown(f"**{alert['type']}** - {alert['plant_name']}")
                            st.markdown(f"*{alert['timestamp']}*")
                            st.markdown(alert['message'])
                            
                            if st.button("Resolve", key=f"resolve_{alert['id']}"):
                                conn = sqlite3.connect('solaris.db')
                                c = conn.cursor()
                                c.execute("UPDATE alerts SET resolved=1, resolved_at=?, resolved_by=? WHERE id=?",
                                          (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                           st.session_state.user['name'], 
                                           alert['id']))
                                conn.commit()
                                conn.close()
                                st.success("Alert resolved")
                                st.rerun()
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                
                if info_alerts:
                    st.markdown("#### Informational Alerts")
                    for alert in info_alerts:
                        with st.container():
                            st.markdown(f'<div class="alert-card info-alert">', unsafe_allow_html=True)
                            st.markdown(f"**{alert['type']}** - {alert['plant_name']}")
                            st.markdown(f"*{alert['timestamp']}*")
                            st.markdown(alert['message'])
                            
                            if st.button("Resolve", key=f"resolve_{alert['id']}"):
                                conn = sqlite3.connect('solaris.db')
                                c = conn.cursor()
                                c.execute("UPDATE alerts SET resolved=1, resolved_at=?, resolved_by=? WHERE id=?",
                                          (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                           st.session_state.user['name'], 
                                           alert['id']))
                                conn.commit()
                                conn.close()
                                st.success("Alert resolved")
                                st.rerun()
                            
                            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Alert History")
            
            if st.session_state.user['role'] == 'admin':
                resolved_alerts = get_alerts(resolved=True)
            else:
                resolved_alerts = get_alerts(st.session_state.current_plant, resolved=True)
            
            if not resolved_alerts:
                st.info("No resolved alerts in history")
            else:
                df_alerts = pd.DataFrame(resolved_alerts)
                st.dataframe(df_alerts[['plant_name', 'type', 'severity', 'message', 'timestamp', 'resolved_at']],
                            use_container_width=True, hide_index=True)
    
    # Maintenance Page (Admin/Operator only)
    elif selected == "Maintenance":
        st.markdown('<div class="header">Plant Maintenance</div>', unsafe_allow_html=True)
        
        if st.session_state.user['role'] not in ['admin', 'operator']:
            st.warning("You don't have permission to access this page.")
            return
        
        tab1, tab2 = st.tabs(["Maintenance Log", "Schedule Maintenance"])
        
        with tab1:
            st.markdown("### Maintenance History")
            
            conn = sqlite3.connect('solaris.db')
            c = conn.cursor()
            
            if st.session_state.user['role'] == 'admin':
                c.execute("SELECT * FROM maintenance ORDER BY scheduled_date DESC")
            else:
                c.execute("SELECT * FROM maintenance WHERE plant_name=? ORDER BY scheduled_date DESC", 
                         (st.session_state.current_plant,))
            
            maintenance_log = []
            for row in c.fetchall():
                maintenance_log.append({
                    "id": row[0],
                    "plant_name": row[1],
                    "type": row[2],
                    "scheduled_date": row[3],
                    "completed_date": row[4],
                    "technician": row[5],
                    "priority": row[6],
                    "status": row[7],
                    "notes": row[8],
                    "created_at": row[9]
                })
            
            conn.close()
            
            # Display log
            df_maintenance = pd.DataFrame(maintenance_log)
            required_columns = ['plant_name', 'type', 'scheduled_date', 'status', 'priority', 'technician']

            if df_maintenance.empty:
                st.warning("No maintenance records found.")
            elif all(col in df_maintenance.columns for col in required_columns):
                st.dataframe(df_maintenance[required_columns], use_container_width=True, hide_index=True)
            else:
                st.error(f"Maintenance data is missing one or more expected columns: {set(required_columns) - set(df_maintenance.columns)}")

        
        with tab2:
            st.markdown("### Schedule New Maintenance")
            
            with st.form("maintenance_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.user['role'] == 'admin':
                        plant = st.selectbox("Plant", get_all_plants())
                    else:
                        plant = st.session_state.current_plant
                        st.markdown(f"**Plant:** {plant}")
                    
                    maintenance_type = st.selectbox("Maintenance Type", 
                                                 ["Panel Cleaning", "Inverter Check", 
                                                  "Transformer Maintenance", "General Inspection"])
                    scheduled_date = st.date_input("Scheduled Date", min_value=datetime.now().date())
                
                with col2:
                    technician = st.text_input("Technician", fake.name())
                    priority = st.select_slider("Priority", ["Low", "Medium", "High"])
                    notes = st.text_area("Notes")
                
                if st.form_submit_button("Schedule Maintenance"):
                    conn = sqlite3.connect('solaris.db')
                    c = conn.cursor()
                    c.execute("INSERT INTO maintenance (plant_name, type, scheduled_date, technician, priority, status, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (plant, maintenance_type, scheduled_date.strftime("%Y-%m-%d"), 
                               technician, priority, "Scheduled", notes))
                    conn.commit()
                    conn.close()
                    
                    # Create info alert
                    create_alert(plant, "Maintenance", "info", 
                               f"{maintenance_type} scheduled for {scheduled_date} with {technician}")
                    
                    st.success("Maintenance scheduled successfully!")
    
    # Reports Page
    elif selected == "Reports":
        st.markdown('<div class="header">Plant Reports</div>', unsafe_allow_html=True)
        
        if not st.session_state.current_plant:
            st.warning("No plant selected. Please contact admin to assign a plant.")
            return
        
        plant_df = st.session_state.plant_data.get(st.session_state.current_plant, pd.DataFrame())
        
        if plant_df.empty:
            st.error("No data available for the selected plant.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox("Report Type", 
                                     ["Daily Performance", "Weekly Summary", "Monthly Analysis"])
        
        with col2:
            if report_type == "Daily Performance":
                report_date = st.date_input("Select Date", datetime.now())
            elif report_type == "Weekly Summary":
                report_week = st.date_input("Select Week", datetime.now())
            else:
                report_month = st.date_input("Select Month", datetime.now())
        
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                if report_type == "Daily Performance":
                    # Filter data for selected date
                    report_df = plant_df[plant_df['Timestamp'].dt.date == report_date]
                    
                    if report_df.empty:
                        st.warning("No data available for selected date")
                    else:
                        st.markdown(f"### Daily Performance Report - {report_date}")
                        
                        # Summary stats
                        total_gen = report_df['Power (MW)'].sum()
                        avg_power = report_df['Power (MW)'].mean()
                        avg_eff = report_df['Efficiency (%)'].mean()
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Generation", f"{total_gen:.1f} MWh")
                        col2.metric("Average Power", f"{avg_power:.1f} MW")
                        col3.metric("Average Efficiency", f"{avg_eff:.1f}%")
                        
                        # Hourly chart
                        fig = px.line(report_df, x="Timestamp", y="Power (MW)", 
                                    title=f"Hourly Power Generation - {report_date}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Environmental factors
                        fig_env = px.line(report_df, x="Timestamp", y=["Temperature (°C)", "Irradiance (W/m²)"],
                                        title="Environmental Conditions",
                                        labels={"value": "Value", "variable": "Metric"})
                        st.plotly_chart(fig_env, use_container_width=True)
                
                elif report_type == "Weekly Summary":
                    # Filter data for selected week
                    start_date = report_week - timedelta(days=report_week.weekday())
                    end_date = start_date + timedelta(days=6)
                    report_df = plant_df[
                        (plant_df['Timestamp'].dt.date >= start_date) & 
                        (plant_df['Timestamp'].dt.date <= end_date)
                    ]
                    
                    if report_df.empty:
                        st.warning("No data available for selected week")
                    else:
                        st.markdown(f"### Weekly Performance Report - Week of {start_date}")
                        
                        # Daily summary
                        daily_df = report_df.groupby(report_df['Timestamp'].dt.date).agg({
                            'Power (MW)': 'sum',
                            'Efficiency (%)': 'mean'
                        }).reset_index()
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Total Weekly Generation", f"{daily_df['Power (MW)'].sum():.1f} MWh")
                        col2.metric("Average Daily Efficiency", f"{daily_df['Efficiency (%)'].mean():.1f}%")
                        
                        # Daily generation chart
                        fig = px.bar(daily_df, x="Timestamp", y="Power (MW)", 
                                    title="Daily Generation Summary")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Efficiency trend
                        fig_eff = px.line(daily_df, x="Timestamp", y="Efficiency (%)",
                                        title="Daily Efficiency Trend")
                        st.plotly_chart(fig_eff, use_container_width=True)
                
                else:  # Monthly Analysis
                    # Filter data for selected month
                    report_df = plant_df[
                        (plant_df['Timestamp'].dt.year == report_month.year) & 
                        (plant_df['Timestamp'].dt.month == report_month.month)
                    ]
                    
                    if report_df.empty:
                        st.warning("No data available for selected month")
                    else:
                        st.markdown(f"### Monthly Performance Report - {report_month.strftime('%B %Y')}")
                        
                        # Weekly summary
                        report_df['Week'] = report_df['Timestamp'].dt.isocalendar().week
                        weekly_df = report_df.groupby('Week').agg({
                            'Power (MW)': 'sum',
                            'Efficiency (%)': 'mean'
                        }).reset_index()
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Total Monthly Generation", f"{weekly_df['Power (MW)'].sum():.1f} MWh")
                        col2.metric("Average Weekly Efficiency", f"{weekly_df['Efficiency (%)'].mean():.1f}%")
                        
                        # Weekly generation chart
                        fig = px.bar(weekly_df, x="Week", y="Power (MW)", 
                                    title="Weekly Generation Summary")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Environmental correlations
                        fig_corr = px.scatter(report_df.sample(n=min(500, len(report_df))), x="Temperature (°C)", y="Efficiency (%)",
                      trendline="lowess", title="Efficiency vs. Temperature")


                        st.plotly_chart(fig_corr, use_container_width=True)
    
    # User Management Page (Admin only)
    elif selected == "User Management":
        st.markdown('<div class="header">User Management</div>', unsafe_allow_html=True)
        
        if st.session_state.user['role'] != 'admin':
            st.warning("You don't have permission to access this page.")
            return
        
        tab1, tab2 = st.tabs(["User List", "Create User"])
        
        with tab1:
            st.markdown("### Current Users")
            
            # Display user table
            conn = sqlite3.connect('solaris.db')
            c = conn.cursor()
            c.execute("SELECT * FROM users")
            
            user_data = []
            for row in c.fetchall():
                plants = get_user_plants(row[0])
                user_data.append({
                    "Username": row[0],
                    "Name": row[2],
                    "Email": row[3],
                    "Role": row[4],
                    "Plants": ", ".join(plants) if plants else "None"
                })
            
            conn.close()
            
            df_users = pd.DataFrame(user_data)
            st.dataframe(df_users, use_container_width=True, hide_index=True)
            
            # User edit/delete
            st.markdown("### Edit User")
            edit_user = st.selectbox("Select User to Edit", df_users['Username'].tolist())
            
            if edit_user:
                conn = sqlite3.connect('solaris.db')
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE username=?", (edit_user,))
                user_info = c.fetchone()
                
                if user_info:
                    user_plants = get_user_plants(edit_user)
                    
                    with st.form("edit_user_form"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            new_name = st.text_input("Name", value=user_info[2])
                            new_email = st.text_input("Email", value=user_info[3])
                        
                        with col2:
                            new_role = st.selectbox("Role", ["admin", "analyst", "operator"], 
                                                  index=["admin", "analyst", "operator"].index(user_info[4]))
                            new_plants = st.multiselect("Assigned Plants", get_all_plants(),
                                                      default=user_plants)
                        
                        if st.form_submit_button("Update User"):
                            # Update user info
                            c.execute("UPDATE users SET name=?, email=?, role=? WHERE username=?",
                                     (new_name, new_email, new_role, edit_user))
                            
                            # Update plants
                            c.execute("DELETE FROM user_plants WHERE username=?", (edit_user,))
                            for plant in new_plants:
                                c.execute("INSERT INTO user_plants (username, plant_name) VALUES (?, ?)",
                                         (edit_user, plant))
                            
                            conn.commit()
                            st.success("User updated successfully!")
                            st.rerun()
                
                # Delete user
                if st.button("Delete User"):
                    if edit_user == st.session_state.user['username']:
                        st.error("You cannot delete your own account!")
                    else:
                        c.execute("DELETE FROM users WHERE username=?", (edit_user,))
                        c.execute("DELETE FROM user_plants WHERE username=?", (edit_user,))
                        conn.commit()
                        conn.close()
                        st.success("User deleted successfully!")
                        st.rerun()
                
                conn.close()
        
        with tab2:
            st.markdown("### Create New User")
            
            with st.form("create_user_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    new_username = st.text_input("Username")
                    new_password = st.text_input("Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                
                with col2:
                    new_name = st.text_input("Full Name")
                    new_email = st.text_input("Email")
                    new_role = st.selectbox("Role", ["admin", "analyst", "operator"])
                    new_plants = st.multiselect("Assigned Plants", get_all_plants())
                
                if st.form_submit_button("Create User"):
                    conn = sqlite3.connect('solaris.db')
                    c = conn.cursor()
                    
                    # Check if username exists
                    c.execute("SELECT COUNT(*) FROM users WHERE username=?", (new_username,))
                    if c.fetchone()[0] > 0:
                        st.error("Username already exists!")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match!")
                    else:
                        # Create user
                        c.execute("INSERT INTO users (username, password, name, email, role) VALUES (?, ?, ?, ?, ?)",
                                 (new_username, make_hashes(new_password), new_name, new_email, new_role))
                        
                        # Assign plants
                        for plant in new_plants:
                            c.execute("INSERT INTO user_plants (username, plant_name) VALUES (?, ?)",
                                     (new_username, plant))
                        
                        conn.commit()
                        conn.close()
                        st.success("User created successfully!")
    
    # Account Settings Page
    elif selected == "Account Settings":
        st.markdown('<div class="header">Account Settings</div>', unsafe_allow_html=True)
        
        with st.form("account_form"):
            st.markdown(f"**Username:** {st.session_state.user['username']}")
            st.markdown(f"**Name:** {st.session_state.user['name']}")
            st.markdown(f"**Email:** {st.session_state.user['email']}")
            st.markdown(f"**Role:** {st.session_state.user['role']}")
            
            # Password change
            st.markdown("### Change Password")
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("Update Account"):
                conn = sqlite3.connect('solaris.db')
                c = conn.cursor()
                
                # Verify current password
                c.execute("SELECT password FROM users WHERE username=?", (st.session_state.user['username'],))
                db_password = c.fetchone()[0]
                
                if not check_hashes(current_password, db_password):
                    st.error("Current password is incorrect!")
                elif new_password and new_password != confirm_password:
                    st.error("New passwords do not match!")
                elif new_password:
                    # Update password
                    new_hashed = make_hashes(new_password)
                    c.execute("UPDATE users SET password=? WHERE username=?",
                             (new_hashed, st.session_state.user['username']))
                    conn.commit()
                    st.success("Password updated successfully!")
                else:
                    st.info("No changes made")
                
                conn.close()

# App Flow Control
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:
    main_app()
