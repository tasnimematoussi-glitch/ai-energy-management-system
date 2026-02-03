"""
=============================================================================
PREDICTION MODULE - LSTM Neural Networks for Energy Forecasting
=============================================================================
Enhanced LSTM models for energy consumption and solar generation prediction
with improved architecture, data augmentation, and performance metrics.

Author: MATOUSSI Tasnim
Lab: AI-P1 | ISET Bizerte
=============================================================================
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class EnhancedEnergyLSTMModel:
    """
    Advanced LSTM model for energy consumption prediction with:
    - Bidirectional LSTM layers
    - Attention mechanism simulation
    - Enhanced feature engineering
    - Model checkpointing and early stopping
    """
    
    def __init__(self, input_features=10, sequence_length=24, lstm_units=128):
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.training_history = None
        
    def _build_model(self):
        """Build enhanced LSTM architecture"""
        model = keras.Sequential([
            # Input layer
            layers.InputLayer(input_shape=(self.sequence_length, self.input_features)),
            
            # First bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(self.lstm_units, return_sequences=True, activation='tanh',
                           kernel_regularizer=regularizers.l2(0.01))
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(self.lstm_units // 2, return_sequences=True, activation='tanh',
                           kernel_regularizer=regularizers.l2(0.01))
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Attention-like layer (Global Average Pooling)
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer
            layers.Dense(1, activation='linear')
        ])
        
        # Use Adam optimizer with learning rate scheduling
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
        
        return model
    
    def generate_enhanced_synthetic_data(self, num_samples=5000):
        """
        Generate enhanced synthetic data with realistic patterns:
        - Multiple seasonal components
        - Weather effects
        - Special events (holidays, weekends)
        - Random noise and anomalies
        """
        data = []
        start_date = datetime.now() - timedelta(days=num_samples // 24)
        
        for i in range(num_samples):
            hour = i % 24
            day_of_year = (i // 24) % 365
            day_of_week = (i // 24) % 7
            month = ((i // 24) % 365) // 30
            
            # Base temperature with seasonal variation
            temp = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365) + \
                   5 * np.sin(2 * np.pi * hour / 24) + np.random.randn() * 3
            
            # Humidity (inversely correlated with temperature)
            humidity = 70 - (temp - 20) * 2 + np.random.randn() * 10
            humidity = np.clip(humidity, 30, 90)
            
            # Occupancy pattern (more realistic)
            occupancy = 0
            if 0 <= hour <= 7:  # Night/early morning
                occupancy = np.random.choice([2, 3, 4], p=[0.6, 0.3, 0.1])
            elif 8 <= hour <= 17:  # Day (weekday vs weekend)
                if day_of_week < 5:  # Weekday
                    occupancy = np.random.choice([0, 1], p=[0.8, 0.2])
                else:  # Weekend
                    occupancy = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2])
            else:  # Evening
                occupancy = np.random.choice([3, 4, 5], p=[0.4, 0.4, 0.2])
            
            # Base load with multiple factors
            base_load = 400
            
            # Time of day factor (peak in morning and evening)
            if 6 <= hour <= 9 or 17 <= hour <= 22:
                time_factor = 2.0
            elif 10 <= hour <= 16:
                time_factor = 1.2
            else:
                time_factor = 0.7
            
            # Weekend factor
            weekend_factor = 1.15 if day_of_week >= 5 else 1.0
            
            # Seasonal factor (more consumption in winter/summer)
            seasonal_factor = 1 + 0.3 * abs(np.sin(2 * np.pi * day_of_year / 365))
            
            # Weather impact (heating/cooling)
            if temp < 18:
                weather_load = (18 - temp) * 50  # Heating
            elif temp > 28:
                weather_load = (temp - 28) * 60  # Cooling
            else:
                weather_load = 0
            
            # Calculate consumption
            consumption = (base_load * time_factor * weekend_factor * seasonal_factor +
                          occupancy * 150 +
                          weather_load +
                          np.random.randn() * 80)
            
            # Add occasional anomalies (1% chance)
            if np.random.rand() < 0.01:
                consumption *= np.random.uniform(1.5, 2.5)
            
            consumption = max(100, consumption)  # Minimum consumption
            
            # Additional features
            is_weekend = 1 if day_of_week >= 5 else 0
            is_peak_hour = 1 if (6 <= hour <= 9 or 17 <= hour <= 22) else 0
            
            data.append([
                consumption,
                temp,
                hour,
                occupancy,
                humidity,
                day_of_week,
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                is_weekend,
                is_peak_hour
            ])
        
        return np.array(data)
    
    def prepare_sequences(self, data):
        """Prepare sequential data for LSTM"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length, :])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)
    
    def train(self, epochs=50, batch_size=64, validation_split=0.2):
        """
        Train the LSTM model with enhanced callbacks
        """
        print("=" * 80)
        print("🧠 TRAINING ENERGY CONSUMPTION LSTM MODEL")
        print("=" * 80)
        
        # Generate data
        print("\n📊 Generating synthetic training data...")
        data = self.generate_enhanced_synthetic_data(5000)
        print(f"✓ Generated {len(data)} samples")
        
        # Scale data
        print("\n🔄 Scaling features...")
        data_scaled = self.scaler.fit_transform(data)
        
        # Prepare sequences
        print("\n📦 Preparing sequences...")
        X, y = self.prepare_sequences(data_scaled)
        print(f"✓ Created {len(X)} sequences")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Build model
        print("\n🏗️  Building model architecture...")
        self.model = self._build_model()
        print(f"✓ Model built with {self.model.count_params():,} parameters")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train
        print(f"\n🚀 Training for {epochs} epochs...")
        print("-" * 80)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.training_history = history.history
        
        # Evaluation
        print("\n" + "=" * 80)
        print("📈 TRAINING COMPLETE - FINAL METRICS")
        print("=" * 80)
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        train_mae = history.history['mae'][-1]
        val_mae = history.history['val_mae'][-1]
        
        print(f"✓ Training Loss: {train_loss:.4f}")
        print(f"✓ Validation Loss: {val_loss:.4f}")
        print(f"✓ Training MAE: {train_mae:.2f}W")
        print(f"✓ Validation MAE: {val_mae:.2f}W")
        print("=" * 80)
        
        return history.history
    
    def predict(self, features):
        """Make prediction from features"""
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
    
    # Ensure 2D array for scaler
        features = np.array(features).reshape(1, -1)  # shape (1, n_features)
    
    # Tile to sequence_length for LSTM
        sequence = np.tile(features, (self.sequence_length, 1))
        sequence_scaled = self.scaler.transform(sequence)  # must match training feature count
        sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, self.input_features)
    
    # Predict
        prediction = self.model.predict(sequence_scaled, verbose=0)
    
    # Inverse transform
        dummy = np.zeros((1, self.input_features))
        dummy[0, 0] = prediction[0, 0]  # only first column = target
        result = self.scaler.inverse_transform(dummy)[0, 0]
    
        return max(0, result)

    
    def save_model(self, filepath='models/energy_lstm_model'):
        """Save model and scalers"""
        self.model.save(f'{filepath}.keras')
        with open(f'{filepath}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/energy_lstm_model'):
        """Load model and scalers"""
        self.model = keras.models.load_model(f'{filepath}.keras')
        with open(f'{filepath}_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")


class EnhancedSolarLSTMModel:
    """
    Advanced LSTM model for solar generation prediction with:
    - Weather-aware architecture
    - ReLU activation for non-negative output
    - Enhanced meteorological features
    """
    
    def __init__(self, input_features=8, sequence_length=24, lstm_units=96):
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = MinMaxScaler()
        self.training_history = None
        
    def _build_model(self):
        """Build solar prediction LSTM architecture"""
        model = keras.Sequential([
            layers.InputLayer(input_shape=(self.sequence_length, self.input_features)),
            
            # Bidirectional LSTM layers
            layers.Bidirectional(
                layers.LSTM(self.lstm_units, return_sequences=True, activation='tanh')
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Bidirectional(
                layers.LSTM(self.lstm_units // 2, return_sequences=False, activation='tanh')
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Dense(48, activation='relu'),
            layers.Dropout(0.15),
            
            layers.Dense(24, activation='relu'),
            
            # Output with ReLU (no negative solar generation)
            layers.Dense(1, activation='relu')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def generate_enhanced_synthetic_data(self, num_samples=5000):
        """Generate realistic solar generation data"""
        data = []
        
        for i in range(num_samples):
            hour = i % 24
            day_of_year = (i // 24) % 365
            
            # Temperature with seasonal variation
            temp = 20 + 12 * np.sin(2 * np.pi * day_of_year / 365) + \
                   6 * np.sin(2 * np.pi * hour / 24) + np.random.randn() * 3
            
            # Weather conditions
            humidity = 60 + 20 * np.random.rand()
            pressure = 1013 + 10 * np.random.randn()
            cloud_cover = np.random.beta(2, 5) * 100  # Skewed toward clear skies
            
            # Solar generation (only during daylight)
            if 6 <= hour <= 20:
                # Solar angle (elevation)
                solar_angle = np.sin((hour - 6) * np.pi / 14)
                
                # Seasonal factor
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
                
                # Cloud impact
                cloud_factor = 1 - (cloud_cover / 100) * 0.8
                
                # Temperature efficiency (solar panels less efficient when hot)
                temp_efficiency = 1 - max(0, (temp - 25) * 0.004)
                
                # Maximum capacity
                max_generation = 2500
                
                solar_gen = (solar_angle * max_generation * seasonal_factor *
                            cloud_factor * temp_efficiency +
                            np.random.randn() * 80)
                
                solar_gen = max(0, solar_gen)
            else:
                solar_gen = 0
            
            is_daylight = 1 if 6 <= hour <= 20 else 0
            
            data.append([
                solar_gen,
                temp,
                hour,
                humidity,
                pressure,
                cloud_cover,
                np.sin(2 * np.pi * hour / 24),
                is_daylight
            ])
        
        return np.array(data)
    
    def prepare_sequences(self, data):
        """Prepare sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length, :])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)
    
    def train(self, epochs=50, batch_size=64):
        """Train solar prediction model"""
        print("=" * 80)
        print("☀️  TRAINING SOLAR GENERATION LSTM MODEL")
        print("=" * 80)
        
        print("\n📊 Generating synthetic training data...")
        data = self.generate_enhanced_synthetic_data(5000)
        print(f"✓ Generated {len(data)} samples")
        
        print("\n🔄 Scaling features...")
        data_scaled = self.scaler.fit_transform(data)
        
        print("\n📦 Preparing sequences...")
        X, y = self.prepare_sequences(data_scaled)
        print(f"✓ Created {len(X)} sequences")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("\n🏗️  Building model architecture...")
        self.model = self._build_model()
        print(f"✓ Model built with {self.model.count_params():,} parameters")
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        print(f"\n🚀 Training for {epochs} epochs...")
        print("-" * 80)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.training_history = history.history
        
        print("\n" + "=" * 80)
        print("📈 TRAINING COMPLETE - FINAL METRICS")
        print("=" * 80)
        val_loss = history.history['val_loss'][-1]
        val_mae = history.history['val_mae'][-1]
        print(f"✓ Validation Loss: {val_loss:.4f}")
        print(f"✓ Validation MAE: {val_mae:.2f}W")
        print("=" * 80)
        
        return history.history
    
    def predict(self, features):
        """Make prediction from features"""
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        full_features = np.concatenate([[0], features])
        sequence = np.tile(full_features, (self.sequence_length, 1))
        sequence_scaled = self.scaler.transform(sequence)
        sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, self.input_features)
        
        prediction = self.model.predict(sequence_scaled, verbose=0)
        
        dummy = np.zeros((1, self.input_features))
        dummy[0, 0] = prediction[0, 0]
        result = self.scaler.inverse_transform(dummy)[0, 0]
        
        return max(0, result)
    
    def save_model(self, filepath='models/solar_lstm_model'):
        """Save model"""
        self.model.save(f'{filepath}.keras')
        with open(f'{filepath}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Solar model saved to {filepath}")
    
    def load_model(self, filepath='models/solar_lstm_model'):
        """Load model"""
        self.model = keras.models.load_model(f'{filepath}.keras')
        with open(f'{filepath}_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Solar model loaded from {filepath}")


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ENHANCED LSTM PREDICTION MODULE - DEMONSTRATION")
    print("=" * 80)
    
    # Train Energy Model
    energy_model = EnhancedEnergyLSTMModel()
    energy_model.train(epochs=30, batch_size=64)
    
    # Train Solar Model
    solar_model = EnhancedSolarLSTMModel()
    solar_model.train(epochs=30, batch_size=64)
    
    # Test predictions
    print("\n" + "=" * 80)
    print("🔮 TESTING PREDICTIONS")
    print("=" * 80)
    
    # Test energy prediction
    test_features_energy = np.array([25, 14, 3, 65, 3, 0.90, -0.43, 0, 1])
    energy_pred = energy_model.predict(test_features_energy)
    print(f"\n⚡ Energy Consumption Prediction:")
    print(f"   Input: Temp=25°C, Hour=14, Occupancy=3, Peak Hour")
    print(f"   Predicted: {energy_pred:.0f}W")
    
    # Test solar prediction
    test_features_solar = np.array([25, 14, 60, 1013, 20, 0.90, 1])
    solar_pred = solar_model.predict(test_features_solar)
    print(f"\n☀️  Solar Generation Prediction:")
    print(f"   Input: Temp=25°C, Hour=14, Cloud=20%, Daylight")
    print(f"   Predicted: {solar_pred:.0f}W")
    
    print("\n" + "=" * 80)
    print("✅ PREDICTION MODULE READY FOR INTEGRATION")
    print("=" * 80)