"""
=============================================================================
QUICK TRAINING SCRIPT - Train Models Once and Save
=============================================================================
Run this script ONCE to train and save models.
Then the dashboard will load pre-trained models instantly.

Author: MATOUSSI Tasnim
Lab: AI-P1 | ISET Bizerte
=============================================================================
"""

import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# =============================================================================
# ENERGY LSTM MODEL
# =============================================================================

class EnhancedEnergyLSTMModel:
    def __init__(self):
        self.model = Sequential([
            LSTM(64, input_shape=(1, 9), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.scaler = StandardScaler()
    
    def generate_synthetic_data(self, num_samples=1000):
        """Generate realistic synthetic training data"""
        print(f"Generating {num_samples} synthetic energy samples...")
        
        X = []
        y = []
        
        for i in range(num_samples):
            hour = i % 24
            temp = 20 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.randn() * 3
            occupancy = np.random.randint(0, 5)
            humidity = 60 + np.random.randn() * 10
            is_weekend = np.random.choice([0, 1], p=[0.7, 0.3])
            
            # Base consumption with patterns
            base = 500
            time_factor = 1.5 if 7 <= hour <= 22 else 0.8
            weekend_factor = 1.1 if is_weekend else 1.0
            
            consumption = (base * time_factor * weekend_factor + 
                          temp * 10 + occupancy * 150 + 
                          np.random.randn() * 80)
            
            features = [
                temp, hour, occupancy, humidity, is_weekend,
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                1 if 6 <= hour <= 9 or 17 <= hour <= 22 else 0,
                1 if 17 <= hour <= 20 else 0
            ]
            
            X.append(features)
            y.append(consumption)
        
        return np.array(X), np.array(y)
    
    def train(self, epochs=30, batch_size=64):
        print("\n" + "="*80)
        print("🧠 TRAINING ENERGY CONSUMPTION LSTM MODEL")
        print("="*80)
        
        # Generate data
        X_train, y_train = self.generate_synthetic_data(2000)
        
        # Fit scaler
        print("Scaling features...")
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        
        # Reshape for LSTM: (samples, timesteps=1, features)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Train
        print(f"Training for {epochs} epochs...")
        history = self.model.fit(
            X_scaled, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2,
            verbose=1
        )
        
        # Save
        self.model.save("models/energy_lstm.h5")
        joblib.dump(self.scaler, "models/energy_scaler.pkl")
        
        print("\n✅ Energy model saved to 'models/energy_lstm.h5'")
        print("✅ Energy scaler saved to 'models/energy_scaler.pkl'")
        
        return history
    
    def predict(self, features):
        """Make prediction from features array"""
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        features_scaled = features_scaled.reshape((1, 1, features_scaled.shape[1]))
        prediction = self.model.predict(features_scaled, verbose=0)
        return max(0, prediction[0, 0])


# =============================================================================
# SOLAR LSTM MODEL
# =============================================================================

class EnhancedSolarLSTMModel:
    def __init__(self):
        self.model = Sequential([
            LSTM(48, input_shape=(1, 7), return_sequences=True),
            Dropout(0.2),
            LSTM(24, return_sequences=False),
            Dropout(0.2),
            Dense(12, activation='relu'),
            Dense(1, activation='relu')  # ReLU for non-negative solar
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.scaler = StandardScaler()
    
    def generate_synthetic_data(self, num_samples=1000):
        """Generate realistic synthetic solar data"""
        print(f"Generating {num_samples} synthetic solar samples...")
        
        X = []
        y = []
        
        for i in range(num_samples):
            hour = i % 24
            temp = 20 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.randn() * 3
            humidity = 60 + np.random.randn() * 10
            pressure = 1013 + np.random.randn() * 5
            cloud_cover = np.random.beta(2, 5) * 100
            
            # Solar generation (only during daylight)
            if 6 <= hour <= 20:
                solar_angle = np.sin((hour - 6) * np.pi / 14)
                cloud_factor = 1 - (cloud_cover / 100) * 0.7
                solar_gen = 2000 * solar_angle * cloud_factor + np.random.randn() * 100
                solar_gen = max(0, solar_gen)
            else:
                solar_gen = 0
            
            features = [
                temp, hour, humidity, pressure, cloud_cover,
                np.sin(2 * np.pi * hour / 24),
                1 if 6 <= hour <= 20 else 0
            ]
            
            X.append(features)
            y.append(solar_gen)
        
        return np.array(X), np.array(y)
    
    def train(self, epochs=30, batch_size=64):
        print("\n" + "="*80)
        print("☀️ TRAINING SOLAR GENERATION LSTM MODEL")
        print("="*80)
        
        # Generate data
        X_train, y_train = self.generate_synthetic_data(2000)
        
        # Fit scaler
        print("Scaling features...")
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        
        # Reshape for LSTM
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Train
        print(f"Training for {epochs} epochs...")
        history = self.model.fit(
            X_scaled, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2,
            verbose=1
        )
        
        # Save
        self.model.save("models/solar_lstm.h5")
        joblib.dump(self.scaler, "models/solar_scaler.pkl")
        
        print("\n✅ Solar model saved to 'models/solar_lstm.h5'")
        print("✅ Solar scaler saved to 'models/solar_scaler.pkl'")
        
        return history
    
    def predict(self, features):
        """Make prediction from features array"""
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        features_scaled = features_scaled.reshape((1, 1, features_scaled.shape[1]))
        prediction = self.model.predict(features_scaled, verbose=0)
        return max(0, prediction[0, 0])


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    print("\n" + "="*80)
    print("🚀 SMART HOME ENERGY SYSTEM - MODEL TRAINING")
    print("="*80)
    print("\nThis script will train both LSTM models and save them.")
    print("Run this ONCE, then use the dashboard with pre-trained models.")
    print("\n" + "="*80 + "\n")
    
    # Train Energy Model
    energy_model = EnhancedEnergyLSTMModel()
    energy_model.train(epochs=25, batch_size=64)
    
    # Test energy prediction
    test_features = [25, 14, 3, 65, 0, 0.9, -0.43, 1, 1]
    test_pred = energy_model.predict(test_features)
    print(f"\n🔮 Test Energy Prediction: {test_pred:.0f}W")
    
    # Train Solar Model
    solar_model = EnhancedSolarLSTMModel()
    solar_model.train(epochs=25, batch_size=64)
    
    # Test solar prediction
    test_features = [25, 14, 60, 1013, 30, 0.9, 1]
    test_pred = solar_model.predict(test_features)
    print(f"🔮 Test Solar Prediction: {test_pred:.0f}W")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\nSaved files:")
    print("  📁 models/energy_lstm.h5")
    print("  📁 models/energy_scaler.pkl")
    print("  📁 models/solar_lstm.h5")
    print("  📁 models/solar_scaler.pkl")
    print("\nNow run: streamlit run dashboard.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()