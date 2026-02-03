"""
=============================================================================
DASHBOARD MODULE - Interactive Streamlit Interface
=============================================================================
Professional dashboard integrating LSTM predictions and Fuzzy Logic decisions
with enhanced visualizations and real-time monitoring.

Author: MATOUSSI Tasnim
Lab: AI-P1 | ISET Bizerte
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json

# Import our custom modules
try:
    from tensorflow.keras.models import load_model
    import joblib
except ImportError as e:
    st.error(f"Missing required libraries: {e}")
    st.stop()
    
from decision import EnhancedFuzzyEnergyController


# =============================================================================
# SIMPLE MODEL WRAPPER FOR PRE-TRAINED MODELS
# =============================================================================

class SimpleEnergyModel:
    """Simple wrapper to load and use pre-trained energy model"""
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def load_pretrained(self):
        """Load pre-trained model and scaler"""
        try:
            import os
            if not os.path.exists('models/energy_lstm.h5'):
                st.error("❌ Energy model file not found: models/energy_lstm.h5")
                return False
            if not os.path.exists('models/energy_scaler.pkl'):
                st.error("❌ Energy scaler file not found: models/energy_scaler.pkl")
                return False
                
            self.model = load_model('models/energy_lstm.h5')
            self.scaler = joblib.load('models/energy_scaler.pkl')
            return True
        except Exception as e:
            st.error(f"❌ Failed to load energy model: {str(e)}")
            return False
    
    def predict(self, features):
        """Make prediction from features array"""
        if self.model is None or self.scaler is None:
            return 500  # Default value
        
        try:
            features = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            features_scaled = features_scaled.reshape((1, 1, features_scaled.shape[1]))
            prediction = self.model.predict(features_scaled, verbose=0)
            return max(0, prediction[0, 0])
        except Exception as e:
            st.warning(f"Prediction error: {e}")
            return 500


class SimpleSolarModel:
    """Simple wrapper to load and use pre-trained solar model"""
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def load_pretrained(self):
        """Load pre-trained model and scaler"""
        try:
            import os
            if not os.path.exists('models/solar_lstm.h5'):
                st.error("❌ Solar model file not found: models/solar_lstm.h5")
                return False
            if not os.path.exists('models/solar_scaler.pkl'):
                st.error("❌ Solar scaler file not found: models/solar_scaler.pkl")
                return False
                
            self.model = load_model('models/solar_lstm.h5')
            self.scaler = joblib.load('models/solar_scaler.pkl')
            return True
        except Exception as e:
            st.error(f"❌ Failed to load solar model: {str(e)}")
            return False
    
    def predict(self, features):
        """Make prediction from features array"""
        if self.model is None or self.scaler is None:
            return 0  # Default value
        
        try:
            features = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            features_scaled = features_scaled.reshape((1, 1, features_scaled.shape[1]))
            prediction = self.model.predict(features_scaled, verbose=0)
            return max(0, prediction[0, 0])
        except Exception as e:
            st.warning(f"Prediction error: {e}")
            return 0

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Smart Home Energy System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-size: 400% 400%;
    animation: gradient 20s ease infinite;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stMetric {
    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
    padding: 1.5rem;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 2px solid rgba(255,255,255,0.3);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    transition: transform 0.3s ease;
}

.stMetric:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
}

h1, h2, h3 {
    color: #ffffff;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.stButton button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.5);
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.7);
}

.rule-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #667eea;
    color: white;
}

.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 15px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-on {
    background: #10b981;
    color: white;
}

.status-off {
    background: #ef4444;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_solar_generation(hour, weather, cloud_cover=30):
    """Calculate realistic solar generation"""
    if weather == 'Rainy' or not (6 <= hour <= 20):
        return 0
    
    base = 1000
    solar_angle = np.sin((hour - 6) * np.pi / 14)
    
    if weather == 'Sunny':
        cloud_factor = 1 - (cloud_cover / 100) * 0.2
    elif weather == 'Cloudy':
        cloud_factor = 0.4
    else:
        cloud_factor = 0
    
    return base + solar_angle * 1500 * cloud_factor


def get_total_consumption(appliances):
    """Calculate total power consumption from appliances"""
    return sum(app['power'] if app['status'] else 0 for app in appliances)


def apply_fuzzy_control(fuzzy_output, appliances, solar_gen):
    """Apply fuzzy control decisions to appliances"""
    savings = 0
    
    if fuzzy_output > 85:  # Reduce aggressively
        for app in appliances:
            if app['priority'] in ['low', 'medium'] and app['status']:
                if 'Refrigerator' not in app['name']:
                    app['status'] = False
                    savings += app['power'] * 0.15 / 1000
                    
    elif fuzzy_output > 65:  # Reduce moderately
        for app in appliances:
            if app['priority'] == 'low' and app['status']:
                app['status'] = False
                savings += app['power'] * 0.15 / 1000
                
    elif fuzzy_output < 30:  # Enable all/most
        for app in appliances:
            if solar_gen > 1500 and not app['status']:
                app['status'] = True
    
    return savings


def format_time_period(hour):
    """Convert hour to readable time period"""
    if 0 <= hour < 6:
        return "Late Night"
    elif 6 <= hour < 9:
        return "Early Morning"
    elif 9 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 22:
        return "Evening"
    else:
        return "Night"

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        # Models (will be trained on first run)
        st.session_state.models_trained = False
        st.session_state.energy_model = None
        st.session_state.solar_model = None
        st.session_state.fuzzy_controller = None
        
        # Simulation state
        st.session_state.running = False
        st.session_state.current_hour = 0
        st.session_state.last_update = time.time()
        
        # Data storage
        st.session_state.energy_data = []
        st.session_state.total_savings = 0
        st.session_state.total_cost = 0
        
        # Appliances
        st.session_state.appliances = [
            {'name': '❄️ Air Conditioner', 'power': 2000, 'priority': 'low', 'status': False, 'consumption': 0},
            {'name': '🔥 Water Heater', 'power': 3000, 'priority': 'high', 'status': False, 'consumption': 0},
            {'name': '👗 Washing Machine', 'power': 1500, 'priority': 'medium', 'status': False, 'consumption': 0},
            {'name': '🧊 Refrigerator', 'power': 200, 'priority': 'high', 'status': True, 'consumption': 0},
            {'name': '💡 Lighting', 'power': 150, 'priority': 'medium', 'status': True, 'consumption': 0},
            {'name': '📺 TV', 'power': 180, 'priority': 'low', 'status': False, 'consumption': 0},
            {'name': '🍕 Microwave', 'power': 1200, 'priority': 'medium', 'status': False, 'consumption': 0},
            {'name': '💻 Computer', 'power': 350, 'priority': 'medium', 'status': False, 'consumption': 0},
            {'name': '🎮 Gaming Console', 'power': 200, 'priority': 'low', 'status': False, 'consumption': 0},
            {'name': '☕ Coffee Maker', 'power': 1000, 'priority': 'low', 'status': False, 'consumption': 0},
        ]

# =============================================================================
# MODEL LOADING (NO TRAINING!)
# =============================================================================

@st.cache_resource
def load_pretrained_models():
    """Load pre-trained models (NO TRAINING - instant loading!)"""
    
    import os
    
    # Check if models directory exists
    if not os.path.exists('models'):
        st.error("❌ Models folder not found!")
        st.warning("""
        ### 🚨 Models Not Found!
        
        The pre-trained models are missing. You need to train them first.
        
        **To train models:**
        
        1. Open a terminal/command prompt
        2. Run: `python train_models.py`
        3. Wait for training to complete (~2-3 minutes)
        4. Refresh this dashboard
        
        **Expected files:**
        - `models/energy_lstm.h5`
        - `models/energy_scaler.pkl`
        - `models/solar_lstm.h5`
        - `models/solar_scaler.pkl`
        """)
        return None, None
    
    with st.spinner("📦 Loading pre-trained models..."):
        energy_model = SimpleEnergyModel()
        solar_model = SimpleSolarModel()
        
        energy_loaded = energy_model.load_pretrained()
        solar_loaded = solar_model.load_pretrained()
        
        if energy_loaded and solar_loaded:
            st.success("✅ Pre-trained models loaded successfully!")
            return energy_model, solar_model
        else:
            st.error("❌ Failed to load models!")
            st.info("""
            ### 📝 Instructions:
            
            **Step 1:** Train the models
            ```bash
            python train_models.py
            ```
            
            **Step 2:** Wait for completion (~2-3 minutes)
            
            **Step 3:** Refresh this page
            """)
            return None, None

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function"""
    
    initialize_session_state()
    
    # Header
    st.markdown("""
    <h1 style='text-align: center; font-size: 3rem; margin-bottom: 0;'>
    🏠 AI Smart Home Energy System
    </h1>
    <h3 style='text-align: center; color: rgba(255,255,255,0.9); margin-top: 0;'>
    LSTM Neural Networks
    </h3>
    <p style='text-align: center; color: rgba(255,255,255,0.8);'>
    Lab AI-P1 | ISET Bizerte | Developed by MATOUSSI Tasnim
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Initialize models
    if not st.session_state.models_trained:
        energy_model, solar_model = load_pretrained_models()
        
        if energy_model is None or solar_model is None:
            st.stop()  # Stop if models not loaded
        
        st.session_state.energy_model = energy_model
        st.session_state.solar_model = solar_model
        st.session_state.fuzzy_controller = EnhancedFuzzyEnergyController()
        st.session_state.models_trained = True
    
    # =============================================================================
    # SIDEBAR CONTROLS
    # =============================================================================
    
    with st.sidebar:
        st.markdown("## ⚙️ System Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start" if not st.session_state.running else "⏸️ Pause",
                        use_container_width=True, type="primary"):
                st.session_state.running = not st.session_state.running
                if st.session_state.running:
                    st.session_state.last_update = time.time()
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.current_hour = 0
                st.session_state.energy_data = []
                st.session_state.total_savings = 0
                st.session_state.total_cost = 0
                for app in st.session_state.appliances:
                    app['consumption'] = 0
                    app['status'] = 'Refrigerator' in app['name'] or 'Lighting' in app['name']
                st.rerun()
        
        st.markdown("---")
        st.markdown("## 🌤️ Environment Settings")
        
        weather = st.selectbox(
            "Weather Condition",
            ["☀️ Sunny", "☁️ Cloudy", "🌧️ Rainy"],
            index=0
        )
        weather = weather.split()[1]
        
        temperature = st.slider("Temperature (°C)", 10, 40, 25, 1)
        cloud_cover = st.slider("Cloud Cover (%)", 0, 100, 30, 5)
        occupancy = st.slider("Occupancy (people)", 0, 5, 2, 1)
        
        st.markdown("---")
        st.markdown("## 🏠 Manual Appliance Control")
        
        for idx, app in enumerate(st.session_state.appliances):
            with st.expander(f"{app['name']} ({app['power']}W)"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.caption(f"Priority: {app['priority'].upper()}")
                with col2:
                    new_status = st.checkbox(
                        "ON",
                        value=app['status'],
                        key=f"app_{idx}"
                    )
                    st.session_state.appliances[idx]['status'] = new_status
    
    # =============================================================================
    # MAIN LOGIC - AUTO ADVANCE (NO COUNTDOWN TIMER)
    # =============================================================================
    
    # Auto-advance time if running (every 5 seconds = 1 hour)
    if st.session_state.running:
        current_time = time.time()
        elapsed = current_time - st.session_state.last_update
        
        # Advance hour when 5 seconds elapsed
        if elapsed >= 5:
            st.session_state.current_hour = (st.session_state.current_hour + 1) % 24
            st.session_state.last_update = current_time
            st.rerun()
    
    hour = st.session_state.current_hour
    solar = calculate_solar_generation(hour, weather, cloud_cover)
    consumption = get_total_consumption(st.session_state.appliances)
    
    # LSTM Predictions
    energy_features = np.array([
        temperature, hour, occupancy,
        60,  # humidity
        1 if 5 <= (datetime.now().weekday()) <= 6 else 0,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        1 if 5 <= (datetime.now().weekday()) <= 6 else 0,
        1 if (6 <= hour <= 9 or 17 <= hour <= 22) else 0
    ])
    
    solar_features = np.array([
        temperature, hour, 60, 1013,
        cloud_cover,
        np.sin(2 * np.pi * hour / 24),
        1 if 6 <= hour <= 20 else 0
    ])
    
    predicted_consumption = st.session_state.energy_model.predict(energy_features)
    predicted_solar = st.session_state.solar_model.predict(solar_features)
    
    # Fuzzy Logic Control
    fuzzy_output = st.session_state.fuzzy_controller.compute(
        consumption, solar, temperature, hour
    )
    
    savings = apply_fuzzy_control(fuzzy_output, st.session_state.appliances, solar)
    st.session_state.total_savings += savings
    
    # Update appliance consumption
    for app in st.session_state.appliances:
        if app['status']:
            app['consumption'] += app['power'] / 1000 / 24  # kWh
    
    net_energy = solar - consumption
    cost = max(0, -net_energy) * 0.15 / 1000  # Cost for deficit
    st.session_state.total_cost += cost
    
    # Log data
    st.session_state.energy_data.append({
        'hour': hour,
        'solar': solar,
        'consumption': consumption,
        'predicted_consumption': predicted_consumption,
        'predicted_solar': predicted_solar,
        'net': net_energy,
        'temperature': temperature,
        'fuzzy_output': fuzzy_output,
        'savings': savings,
        'cost': cost
    })
    
    if len(st.session_state.energy_data) > 48:  # Keep last 48 hours
        st.session_state.energy_data.pop(0)
    
    # =============================================================================
    # KEY METRICS DISPLAY
    # =============================================================================
    
    st.markdown("## 📊 Real-Time System Status")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "⏰ Current Time",
            f"{hour:02d}:00",
            delta=format_time_period(hour)
        )
    
    with col2:
        st.metric(
            "☀️ Solar Generation",
            f"{solar:.0f}W",
            delta=f"{(solar - predicted_solar):.0f}W"
        )
    
    with col3:
        st.metric(
            "⚡ Consumption",
            f"{consumption:.0f}W",
            delta=f"{(consumption - predicted_consumption):.0f}W"
        )
    
    with col4:
        st.metric(
            "🔋 Net Energy",
            f"{net_energy:.0f}W",
            delta="Surplus" if net_energy > 0 else "Deficit",
            delta_color="normal" if net_energy > 0 else "inverse"
        )
    
    with col5:
        st.metric(
            "💰 Total Savings",
            f"${st.session_state.total_savings:.2f}",
            delta=f"+${savings:.4f}"
        )
    
    st.markdown("---")
    
    # =============================================================================
    # VISUALIZATIONS
    # =============================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "🧠 LSTM Predictions",
        "🎯 Fuzzy Decisions",
        "🏠 Appliances",
        "📋 Rules"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Energy Flow Sankey Diagram")
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="white", width=2),
                    label=["☀️ Solar", "⚡ Grid", "🧠 AI Controller", "🏠 Appliances", "🔋 Battery"],
                    color=['#fbbf24', '#667eea', '#764ba2', '#10b981', '#3b82f6']
                ),
                link=dict(
                    source=[0, 1, 2, 2, 2],
                    target=[2, 2, 3, 4, 1],
                    value=[
                        max(0, solar),
                        max(0, consumption - solar),
                        min(consumption, solar + max(0, consumption - solar)),
                        max(0, solar - consumption) * 0.3,
                        max(0, consumption - solar) * 0.2
                    ],
                    color=[
                        'rgba(251, 191, 36, 0.4)',
                        'rgba(102, 126, 234, 0.4)',
                        'rgba(118, 75, 162, 0.4)',
                        'rgba(16, 185, 129, 0.4)',
                        'rgba(59, 130, 246, 0.4)'
                    ]
                )
            )])
            
            fig_sankey.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12)
            )
            st.plotly_chart(fig_sankey, use_container_width=True)
        
        with col2:
            st.markdown("### Fuzzy Control Output")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fuzzy_output,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Control Level", 'font': {'color': 'white', 'size': 16}},
                delta={'reference': 50, 'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': 'white'},
                    'bar': {'color': "#764ba2"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d1fae5"},
                        {'range': [30, 60], 'color': "#fed7aa"},
                        {'range': [60, 80], 'color': "#fecaca"},
                        {'range': [80, 100], 'color': "#fca5a5"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Historical data chart
        if len(st.session_state.energy_data) > 1:
            st.markdown("### 24-Hour Energy Profile")
            df = pd.DataFrame(st.session_state.energy_data)
            
            fig_energy = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_energy.add_trace(
                go.Scatter(x=df['hour'], y=df['consumption'],
                          name="Consumption", mode='lines',
                          line=dict(color='#ef4444', width=3)),
                secondary_y=False
            )
            
            fig_energy.add_trace(
                go.Scatter(x=df['hour'], y=df['solar'],
                          name="Solar", mode='lines',
                          fill='tozeroy',
                          line=dict(color='#fbbf24', width=2)),
                secondary_y=False
            )
            
            fig_energy.add_trace(
                go.Scatter(x=df['hour'], y=df['fuzzy_output'],
                          name="Fuzzy Output", mode='lines',
                          line=dict(color='#764ba2', width=2, dash='dash')),
                secondary_y=True
            )
            
            fig_energy.update_xaxes(title_text="Hour", color='white')
            fig_energy.update_yaxes(title_text="Power (W)", secondary_y=False, color='white')
            fig_energy.update_yaxes(title_text="Control (%)", secondary_y=True, color='white')
            
            fig_energy.update_layout(
                height=400,
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.1)',
                font=dict(color='white'),
                legend=dict(
                    bgcolor='rgba(255,255,255,0.1)',
                    bordercolor='white',
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig_energy, use_container_width=True)
    
    # TAB 2: LSTM PREDICTIONS
    with tab2:
        st.markdown("### LSTM Model Performance")
        
        if len(st.session_state.energy_data) > 1:
            df = pd.DataFrame(st.session_state.energy_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cons_mae = np.abs(df['consumption'] - df['predicted_consumption']).mean()
                st.metric("⚡ Consumption MAE", f"{cons_mae:.0f}W")
            
            with col2:
                solar_mae = np.abs(df['solar'] - df['predicted_solar']).mean()
                st.metric("☀️ Solar MAE", f"{solar_mae:.0f}W")
            
            with col3:
                cons_error_pct = (cons_mae / df['consumption'].mean() * 100) if df['consumption'].mean() > 0 else 0
                st.metric("⚡ Error %", f"{cons_error_pct:.1f}%")
            
            with col4:
                solar_error_pct = (solar_mae / (df['solar'].mean() + 1) * 100)
                st.metric("☀️ Error %", f"{solar_error_pct:.1f}%")
            
            # Prediction comparison
            fig_pred = go.Figure()
            
            fig_pred.add_trace(go.Scatter(
                x=df['hour'], y=df['consumption'],
                name='Actual Consumption',
                line=dict(color='#ef4444', width=3)
            ))
            
            fig_pred.add_trace(go.Scatter(
                x=df['hour'], y=df['predicted_consumption'],
                name='Predicted Consumption',
                line=dict(color='#fbbf24', width=2, dash='dash')
            ))
            
            fig_pred.update_layout(
                title="LSTM Consumption Predictions vs Actual",
                xaxis_title="Hour",
                yaxis_title="Power (W)",
                height=400,
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.1)',
                font=dict(color='white'),
                legend=dict(bgcolor='rgba(255,255,255,0.1)')
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
    
    # TAB 3: FUZZY DECISIONS
    with tab3:
        st.markdown("### Current Fuzzy Logic Decision")
        
        explanation = st.session_state.fuzzy_controller.get_decision_explanation(
            consumption, solar, temperature, hour, fuzzy_output
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Determine decision color
            if fuzzy_output < 30:
                decision_color = "#10b981"
                bg_color = "rgba(16, 185, 129, 0.2)"
            elif fuzzy_output < 60:
                decision_color = "#f59e0b"
                bg_color = "rgba(245, 158, 11, 0.2)"
            else:
                decision_color = "#ef4444"
                bg_color = "rgba(239, 68, 68, 0.2)"
            
            st.markdown(f"""
            <div style='background: {bg_color}; padding: 1.5rem; border-radius: 10px; 
                        border-left: 5px solid {decision_color}; backdrop-filter: blur(10px);'>
                <h3 style='margin-top: 0; color: white;'>🎯 Decision: {explanation['decision']}</h3>
                <p style='color: rgba(255,255,255,0.95); font-size: 1.1rem;'>
                    <strong>Control Value:</strong> {fuzzy_output:.1f}/100
                </p>
                <p style='color: rgba(255,255,255,0.95); font-size: 1.05rem;'>
                    <strong>Recommendation:</strong> {explanation['recommendation']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### System Conditions")
            for factor in explanation['primary_factors']:
                st.info(factor)
        
        with col2:
            st.markdown("#### 🔥 Top Active Fuzzy Rules")
            
            # Get fresh active rules
            active_rules = st.session_state.fuzzy_controller.get_active_rules(
                consumption, solar, temperature, hour
            )
            
            if active_rules and len(active_rules) > 0:
                st.success(f"✅ Found {len(active_rules)} active rules")
                
                for i, rule in enumerate(active_rules[:5], 1):
                    # Color code by activation level
                    if rule['activation'] > 0.7:
                        activation_color = "#10b981"
                        bg_gradient = "linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(5, 150, 105, 0.2))"
                    elif rule['activation'] > 0.4:
                        activation_color = "#f59e0b"
                        bg_gradient = "linear-gradient(135deg, rgba(245, 158, 11, 0.3), rgba(217, 119, 6, 0.2))"
                    else:
                        activation_color = "#ef4444"
                        bg_gradient = "linear-gradient(135deg, rgba(239, 68, 68, 0.3), rgba(220, 38, 38, 0.2))"
                    
                    st.markdown(f"""
                    <div style='background: {bg_gradient}; 
                                padding: 1rem; 
                                border-radius: 10px; 
                                margin: 0.75rem 0;
                                border: 2px solid {activation_color};
                                backdrop-filter: blur(10px);'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                            <strong style='color: white; font-size: 1.1rem;'>#{i} - Rule {rule['rule_id']}</strong>
                            <span style='background: {activation_color}; 
                                         padding: 0.35rem 0.85rem; 
                                         border-radius: 20px; 
                                         font-size: 0.95rem; 
                                         font-weight: 700;
                                         color: white;
                                         box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                                {rule['activation']:.3f}
                            </span>
                        </div>
                        <p style='color: white; margin: 0.5rem 0; font-size: 1rem; font-weight: 500;'>
                            {rule['description']}
                        </p>
                        <small style='color: rgba(255,255,255,0.85); font-size: 0.9rem;'>
                            📂 Category: {rule['category']}
                        </small><br>
                        <small style='color: rgba(255,255,255,0.8); font-size: 0.85rem; font-style: italic;'>
                            💡 {rule['rationale']}
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show membership values
                st.markdown("---")
                st.markdown("#### 📊 Fuzzy Membership Values")
                
                # Get current memberships for display
                demand_level = "Very Low" if consumption < 500 else "Low" if consumption < 1700 else "Medium" if consumption < 3500 else "High" if consumption < 4500 else "Very High"
                solar_level = "None" if solar < 100 else "Poor" if solar < 800 else "Moderate" if solar < 2200 else "Good" if solar < 2700 else "Excellent"
                temp_level = "Very Cold" if temperature < 15 else "Cold" if temperature < 22 else "Moderate" if temperature < 30 else "Hot" if temperature < 35 else "Very Hot"
                time_level = format_time_period(hour)
                
                membership_data = {
                    'Variable': ['⚡ Demand', '☀️ Solar', '🌡️ Temp', '⏰ Time'],
                    'Value': [f'{consumption:.0f}W', f'{solar:.0f}W', f'{temperature}°C', f'{hour}:00'],
                    'Fuzzy Level': [demand_level, solar_level, temp_level, time_level]
                }
                
                membership_df = pd.DataFrame(membership_data)
                st.dataframe(membership_df, use_container_width=True, hide_index=True)
                
            else:
                st.warning("⚠️ No rules currently active with significant activation levels (threshold > 0.1)")
                st.info(f"""
                **Current Conditions:**
                - Demand: {consumption:.0f}W
                - Solar: {solar:.0f}W
                - Temperature: {temperature}°C
                - Time: {hour}:00
                
                Try adjusting the environment settings or wait for conditions to change.
                """)
    
    # TAB 4: APPLIANCES
    with tab4:
        st.markdown("### Appliance Status & Consumption")
        
        appliance_df = pd.DataFrame([
            {
                'Appliance': app['name'],
                'Status': '🟢 ON' if app['status'] else '⚫ OFF',
                'Power': f"{app['power']}W",
                'Priority': app['priority'].upper(),
                'Consumption': f"{app['consumption']:.3f} kWh",
                'Cost': f"${app['consumption'] * 0.15:.2f}"
            }
            for app in st.session_state.appliances
        ])
        
        st.dataframe(appliance_df, use_container_width=True, height=400)
        
        # Pie chart
        active_apps = [app for app in st.session_state.appliances if app['status']]
        if active_apps:
            fig_pie = go.Figure(data=[go.Pie(
                labels=[app['name'] for app in active_apps],
                values=[app['power'] for app in active_apps],
                hole=.4,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            
            fig_pie.update_layout(
                title="Active Appliances Power Distribution",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # TAB 5: RULES
    with tab5:
        st.markdown("### Complete Fuzzy Rule Base (22 Rules)")
        
        categories = st.session_state.fuzzy_controller.get_rule_categories()
        
        for category, rules in categories.items():
            with st.expander(f"📋 {category} ({len(rules)} rules)", expanded=False):
                for rule in rules:
                    st.markdown(f"""
                    <div class='rule-card'>
                        <strong>Rule {rule['id']}</strong>: {rule['description']}<br>
                        <small style='color: rgba(255,255,255,0.8);'>💡 {rule['rationale']}</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    # =============================================================================
    # FOOTER - COST & ENVIRONMENTAL ANALYSIS
    # =============================================================================
    
    st.markdown("---")
    st.markdown("## 💰 Financial & Environmental Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_consumption_kwh = sum(app['consumption'] for app in st.session_state.appliances)
    baseline_cost = total_consumption_kwh * 0.15
    savings_percent = (st.session_state.total_savings / max(baseline_cost, 0.01)) * 100
    
    with col1:
        st.metric("⚡ Total Usage", f"{total_consumption_kwh:.2f} kWh")
        st.metric("💸 Total Cost", f"${st.session_state.total_cost:.2f}")
    
    with col2:
        st.metric("✨ Total Savings", f"${st.session_state.total_savings:.2f}")
        st.metric("📊 Savings Rate", f"{savings_percent:.1f}%")
    
    with col3:
        monthly_projection = st.session_state.total_savings * 30
        yearly_projection = st.session_state.total_savings * 365
        st.metric("📅 Monthly Projection", f"${monthly_projection:.2f}")
        st.metric("🎉 Yearly Projection", f"${yearly_projection:.2f}")
    
    with col4:
        carbon_saved = total_consumption_kwh * 0.5  # kg CO2
        trees_equivalent = carbon_saved / 21
        st.metric("🌱 CO₂ Saved", f"{carbon_saved:.1f} kg")
        st.metric("🌳 Trees Equivalent", f"{trees_equivalent:.1f} trees")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); 
                border-radius: 15px; backdrop-filter: blur(10px);'>
        <h3 style='color: white; margin: 0;'>🏠 AI Smart Home Energy Management System</h3>
        <p style='color: rgba(255,255,255,0.9); margin-top: 1rem;'>
            <strong>Hybrid AI Architecture:</strong> LSTM Neural Networks + Fuzzy Logic Control
        </p>
        <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>
            Lab AI-P1 | Department of Electrical Engineering @ ISET Bizerte<br>
            Developed by: <strong>MATOUSSI Tasnim</strong>
        </p>
        <p style='color: rgba(255,255,255,0.7); font-size: 0.8rem; margin-top: 1rem;'>
            © 2025 | Advanced AI for Sustainable Energy Management
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()