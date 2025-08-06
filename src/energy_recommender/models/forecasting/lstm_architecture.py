import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Check TensorFlow availability
try:
    tf.config.list_physical_devices()
    TF_AVAILABLE = True
except Exception as e:
    print(f"⚠️  TensorFlow issue detected: {e}")
    print("   LSTM training may require TensorFlow configuration")
    TF_AVAILABLE = False

class MultiCohortLSTM:
    """
    Multi-output LSTM for forecasting energy demand across 15 building cohorts
    
    Design Philosophy: Production-ready architecture focused on solving the 
    forecasting problem with appropriate complexity and proper time series handling.
    """
    
    def __init__(self, cohort_names, sequence_length=48, forecast_horizon=24):
        self.cohort_names = cohort_names
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.n_cohorts = len(cohort_names)
        
        # Feature configuration
        self.weather_features = [
            'dry_bulb_temperature_c', 'relative_humidity_pct', 
            'wind_speed_ms', 'global_horizontal_radiation_wm2'
        ]
        
        # Simple temporal features - let LSTM learn patterns from sequences
        self.temporal_features = [
            'hour_of_day', 'day_of_week', 'is_weekend'
        ]
        
        # Scalers for normalization
        self.weather_scaler = StandardScaler()
        self.demand_scalers = {cohort: StandardScaler() for cohort in cohort_names}
        
        self.model = None
        self.training_history = None
        
    def create_temporal_features(self, datetime_series):
        """
        Create simple temporal features
        
        Note: pandas uses Monday=0 convention:
        - Monday through Friday = 0-4 (weekdays)
        - Saturday, Sunday = 5-6 (weekends)
        """
        df = pd.DataFrame(index=datetime_series)
        df['hour_of_day'] = datetime_series.hour  # 0-23, let LSTM handle scaling
        df['day_of_week'] = datetime_series.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = (datetime_series.dayofweek > 4).astype(int)  # Sat=5, Sun=6
        
        return df
    
    def validate_input_data(self, grid_data, scenario_name):
        """Validate input data quality and completeness"""
        required_columns = self.weather_features + [col for col in grid_data.columns if col.startswith('demand_mw_')]
        
        missing_cols = [col for col in required_columns if col not in grid_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {scenario_name}: {missing_cols}")
        
        # Check for NaN values
        nan_counts = grid_data[required_columns].isnull().sum()
        if nan_counts.any():
            print(f"⚠️  Warning: NaN values found in {scenario_name}:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"   {col}: {count} NaN values")
        
        # Check data ranges for weather features
        temp_range = grid_data['dry_bulb_temperature_c']
        if temp_range.min() < -30 or temp_range.max() > 50:
            print(f"⚠️  Warning: Unusual temperature range in {scenario_name}: {temp_range.min():.1f} to {temp_range.max():.1f}°C")
    
    def prepare_sequences(self, grid_data, scenario_name, fit_scalers=False):
        """Convert grid data into LSTM training sequences with proper validation"""
        print(f"📊 Preparing sequences for {scenario_name}...")
        
        # Validate input data
        self.validate_input_data(grid_data, scenario_name)
        
        # Handle any missing values with forward fill
        grid_data_clean = grid_data.fillna(method='ffill').fillna(method='bfill')
        
        # Extract features
        temporal_df = self.create_temporal_features(grid_data_clean['date_time'])
        weather_data = grid_data_clean[self.weather_features].values
        temporal_data = temporal_df[self.temporal_features].values
        
        # Get demand data for each cohort
        demand_columns = [col for col in grid_data_clean.columns if col.startswith('demand_mw_')]
        if len(demand_columns) != self.n_cohorts:
            print(f"⚠️  Warning: Expected {self.n_cohorts} cohorts, found {len(demand_columns)} in {scenario_name}")
        
        demand_data = grid_data_clean[demand_columns].values
        
        # Scale features
        if fit_scalers:
            weather_data = self.weather_scaler.fit_transform(weather_data)
            # Fit individual scalers for each cohort's demand
            for i, cohort in enumerate(self.cohort_names[:len(demand_columns)]):
                cohort_demand = demand_data[:, i].reshape(-1, 1)
                self.demand_scalers[cohort].fit(cohort_demand)
                demand_data[:, i] = self.demand_scalers[cohort].transform(cohort_demand).flatten()
        else:
            weather_data = self.weather_scaler.transform(weather_data)
            for i, cohort in enumerate(self.cohort_names[:len(demand_columns)]):
                cohort_demand = demand_data[:, i].reshape(-1, 1)
                demand_data[:, i] = self.demand_scalers[cohort].transform(cohort_demand).flatten()
        
        # Combine all input features: weather + temporal + historical demand
        all_features = np.concatenate([weather_data, temporal_data, demand_data], axis=1)
        
        # Create sequences for LSTM
        sequences_X = []
        sequences_y = []
        
        total_length = len(all_features)
        n_sequences = total_length - self.sequence_length - self.forecast_horizon + 1
        
        if n_sequences <= 0:
            raise ValueError(f"Insufficient data in {scenario_name}: need at least {self.sequence_length + self.forecast_horizon} hours")
        
        for i in range(self.sequence_length, total_length - self.forecast_horizon + 1):
            # Input: 48 hours of [weather + temporal + demand] features
            seq_x = all_features[i-self.sequence_length:i]
            # Output: next 24 hours of demand for each cohort
            seq_y = demand_data[i:i+self.forecast_horizon]  # Shape: (24, n_cohorts)
            
            sequences_X.append(seq_x)
            sequences_y.append(seq_y)
        
        sequences_X = np.array(sequences_X)
        sequences_y = np.array(sequences_y)
        
        print(f"   Generated {len(sequences_X)} sequences")
        print(f"   Input shape: {sequences_X.shape}")
        print(f"   Output shape: {sequences_y.shape}")
        
        return sequences_X, sequences_y
    
    def build_model(self, input_shape):
        """
        Build focused multi-output LSTM architecture
        
        Layer sizing principles:
        - LSTM layers: 64 units (sufficient for energy forecasting patterns)
        - Dense layers: Based on output complexity (number of cohorts)
        - Dropout: 0.2 for regularization without over-damping
        """
        n_features = input_shape[-1]
        
        # Principled architecture sizing
        lstm_units = 64      # Standard size for sequence modeling
        dense_units = max(32, self.n_cohorts * 2)  # Scale with output complexity
        
        print(f"🏗️ Architecture configuration:")
        print(f"   Input features: {n_features}")
        print(f"   LSTM units: {lstm_units}")
        print(f"   Dense units: {dense_units}")
        print(f"   Output cohorts: {self.n_cohorts}")
        
        # Build model
        inputs = layers.Input(shape=input_shape, name='sequence_input')
        
        # Two-layer LSTM for temporal pattern learning
        x = layers.LSTM(
            lstm_units, 
            return_sequences=True, 
            dropout=0.2,
            recurrent_dropout=0.1,
            name='lstm_layer_1'
        )(inputs)
        
        x = layers.LSTM(
            lstm_units // 2,  # 32 units for second layer
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.1,
            name='lstm_layer_2'
        )(x)
        
        # Shared dense processing
        x = layers.Dense(dense_units, activation='relu', name='shared_dense')(x)
        x = layers.Dropout(0.3)(x)
        
        # Individual output heads for each cohort
        cohort_outputs = {}
        for i, cohort_name in enumerate(self.cohort_names):
            # Cohort-specific forecasting head
            cohort_output = layers.Dense(
                self.forecast_horizon, 
                activation='linear',
                name=f'{cohort_name}_output'
            )(x)
            cohort_outputs[f'{cohort_name}_output'] = cohort_output
        
        model = Model(inputs=inputs, outputs=cohort_outputs, name='MultiCohortLSTM')
        
        # Compile with cohort-specific metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_temporal_validation_split(self, X_combined, y_dict, validation_split=0.2):
        """
        Create proper temporal validation split for time series
        
        Critical: Uses last 20% of time series as validation (no data leakage)
        """
        split_idx = int(len(X_combined) * (1 - validation_split))
        
        X_train = X_combined[:split_idx]
        X_val = X_combined[split_idx:]
        
        y_train_dict = {}
        y_val_dict = {}
        
        for cohort_key in y_dict.keys():
            y_train_dict[cohort_key] = y_dict[cohort_key][:split_idx]
            y_val_dict[cohort_key] = y_dict[cohort_key][split_idx:]
        
        print(f"📊 Temporal validation split:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        
        return X_train, X_val, y_train_dict, y_val_dict
    
    def train_model(self, all_scenarios, validation_split=0.2, epochs=50, batch_size=32):
        """Train LSTM with proper time series validation"""
        
        print("🧠 Starting Multi-Cohort LSTM Training...")
        print(f"   Training scenarios: {len(all_scenarios)}")
        
        # Combine data from all scenarios
        all_sequences_X = []
        all_sequences_y = []
        
        scenario_names = list(all_scenarios.keys())
        
        # Fit scalers on first scenario
        first_scenario = scenario_names[0]
        first_data = all_scenarios[first_scenario]['grid_data']
        X_first, y_first = self.prepare_sequences(first_data, first_scenario, fit_scalers=True)
        all_sequences_X.append(X_first)
        all_sequences_y.append(y_first)
        
        # Process remaining scenarios with fitted scalers
        for scenario_name in scenario_names[1:]:
            grid_data = all_scenarios[scenario_name]['grid_data']
            X_scenario, y_scenario = self.prepare_sequences(grid_data, scenario_name, fit_scalers=False)
            all_sequences_X.append(X_scenario)
            all_sequences_y.append(y_scenario)
        
        # Combine all training data
        X_combined = np.concatenate(all_sequences_X, axis=0)
        y_combined = np.concatenate(all_sequences_y, axis=0)
        
        print(f"📊 Combined training data: {X_combined.shape[0]:,} sequences")
        
        # Prepare multi-output targets
        y_dict = {}
        for i, cohort in enumerate(self.cohort_names):
            if i < y_combined.shape[2]:  # Handle cases with fewer cohorts than expected
                y_dict[f'{cohort}_output'] = y_combined[:, :, i]
        
        # Create proper temporal validation split
        X_train, X_val, y_train_dict, y_val_dict = self.create_temporal_validation_split(
            X_combined, y_dict, validation_split
        )
        
        # Build model
        input_shape = (self.sequence_length, X_combined.shape[2])
        self.model = self.build_model(input_shape)
        self.model.summary()
        
        # Conservative training callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print(f"\n🚀 Training for up to {epochs} epochs...")
        self.training_history = self.model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.training_history
    
    def prepare_prediction_sequence(self, recent_weather_data):
        """
        Prepare single prediction sequence from raw weather data
        
        This method bridges the gap between integration code and model prediction
        """
        if len(recent_weather_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} hours of recent data")
        
        # Take last 48 hours
        weather_recent = recent_weather_data.tail(self.sequence_length).copy()
        
        # Create temporal features
        temporal_df = self.create_temporal_features(weather_recent['date_time'])
        weather_data = self.weather_scaler.transform(weather_recent[self.weather_features].values)
        temporal_data = temporal_df[self.temporal_features].values
        
        # For prediction, use zeros for historical demand (or last known values if available)
        demand_data = np.zeros((self.sequence_length, self.n_cohorts))
        
        # Combine features
        all_features = np.concatenate([weather_data, temporal_data, demand_data], axis=1)
        
        return all_features.reshape(1, self.sequence_length, -1)
    
    def predict_cohort_demands(self, weather_sequence):
        """Generate 24-hour forecasts for all cohorts"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Handle different input types
        if isinstance(weather_sequence, pd.DataFrame):
            # Convert DataFrame to processed sequence
            processed_sequence = self.prepare_prediction_sequence(weather_sequence)
        else:
            # Assume already processed sequence
            if len(weather_sequence.shape) == 2:
                processed_sequence = weather_sequence.reshape(1, *weather_sequence.shape)
            else:
                processed_sequence = weather_sequence
        
        # Generate predictions
        predictions = self.model.predict(processed_sequence, verbose=0)
        
        # Inverse transform to original MW scale
        forecasts = {}
        for i, cohort in enumerate(self.cohort_names):
            if f'{cohort}_output' in predictions:
                cohort_pred = predictions[f'{cohort}_output'][0]
                # Transform back to MW
                cohort_pred_mw = self.demand_scalers[cohort].inverse_transform(
                    cohort_pred.reshape(-1, 1)
                ).flatten()
                forecasts[cohort] = cohort_pred_mw
        
        return forecasts
    
    def save_model(self, filepath):
        """Save model and scalers for deployment"""
        if self.model is None:
            raise ValueError("No trained model to save")
            
        self.model.save(f"{filepath}_model.keras")
        
        scalers = {
            'weather_scaler': self.weather_scaler,
            'demand_scalers': self.demand_scalers,
            'cohort_names': self.cohort_names,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'weather_features': self.weather_features,
            'temporal_features': self.temporal_features
        }
        joblib.dump(scalers, f"{filepath}_scalers.pkl")
        
        print(f"✅ Model saved: {filepath}_model.keras")
        print(f"✅ Scalers saved: {filepath}_scalers.pkl")
    
    def load_model(self, filepath):
        """Load trained model and scalers for deployment"""
        self.model = tf.keras.models.load_model(f"{filepath}_model.keras")
        
        scalers = joblib.load(f"{filepath}_scalers.pkl")
        self.weather_scaler = scalers['weather_scaler']
        self.demand_scalers = scalers['demand_scalers']
        self.cohort_names = scalers['cohort_names']
        self.sequence_length = scalers['sequence_length']
        self.forecast_horizon = scalers['forecast_horizon']
        self.weather_features = scalers.get('weather_features', self.weather_features)
        self.temporal_features = scalers.get('temporal_features', self.temporal_features)
        
        print(f"✅ Model loaded: {filepath}")

def create_lstm_trainer():
    """Factory function with your specific cohort configuration"""
    cohort_names = [
        'SmallOffice_Small', 'SmallOffice_Medium-Small', 
        'RetailStandalone_Medium-Small', 'RetailStandalone_Small',
        'FullServiceRestaurant_Small', 'SmallOffice_Medium-Large',
        'RetailStandalone_Medium-Large', 'RetailStripmall_Medium-Small',
        'Warehouse_Medium-Large', 'RetailStripmall_Medium-Large',
        'FullServiceRestaurant_Medium-Small', 'Warehouse_Medium-Small',
        'LargeHotel_Large', 'RetailStripmall_Small', 'SmallHotel_Medium-Small'
    ]
    
    return MultiCohortLSTM(cohort_names)

# Training script example
if __name__ == "__main__":
    # Example usage
    try:
        # Try relative import first (when imported as module)
        from .demand_simulation import generate_all_demand_scenarios
    except ImportError:
        # Fall back to absolute import (when run directly)
        from demand_simulation import generate_all_demand_scenarios
    
    print("🚀 Starting LSTM Training Example...")
    
    # Generate training data
    all_scenarios = generate_all_demand_scenarios()
    
    # Create and train model
    lstm_model = create_lstm_trainer()
    history = lstm_model.train_model(all_scenarios, epochs=30)
    
    # Save trained model (relative to project root)
    import os
    # From lstm_architecture.py, go up 4 levels to reach project root
    # lstm_architecture.py -> forecasting -> models -> energy_recommender -> src -> project_root
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
    model_path = os.path.join(project_root, "models", "multi_cohort_lstm")
    
    print(f"📁 Project root: {project_root}")
    print(f"💾 Saving model to: {model_path}")
    
    lstm_model.save_model(model_path)
    
    print("✅ Training and saving completed!")