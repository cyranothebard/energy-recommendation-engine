import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class MultiCohortLSTM(nn.Module):
    """
    PyTorch Multi-output LSTM for forecasting energy demand across 15 building cohorts
    
    Design Philosophy: Production-ready architecture focused on solving the 
    forecasting problem with appropriate complexity and proper time series handling.
    """
    
    def __init__(self, cohort_names, sequence_length=48, forecast_horizon=24, hidden_size=64):
        super(MultiCohortLSTM, self).__init__()
        
        self.cohort_names = cohort_names
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.n_cohorts = len(cohort_names)
        self.hidden_size = hidden_size
        
        # Feature configuration
        self.weather_features = [
            'dry_bulb_temperature_c', 'relative_humidity_pct', 
            'wind_speed_ms', 'global_horizontal_radiation_wm2'
        ]
        
        self.temporal_features = [
            'hour_of_day', 'day_of_week', 'is_weekend'
        ]
        
        # Calculate input size: weather + temporal + historical demand
        self.input_size = len(self.weather_features) + len(self.temporal_features) + self.n_cohorts
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.2
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            batch_first=True,
            dropout=0.2
        )
        
        # Shared dense layer
        self.shared_dense = nn.Linear(hidden_size // 2, max(32, self.n_cohorts * 2))
        self.dropout = nn.Dropout(0.3)
        
        # Individual output heads for each cohort
        self.output_heads = nn.ModuleDict({
            cohort_name: nn.Linear(max(32, self.n_cohorts * 2), forecast_horizon)
            for cohort_name in cohort_names
        })
        
        # Scalers for normalization
        self.weather_scaler = StandardScaler()
        self.demand_scalers = {cohort: StandardScaler() for cohort in cohort_names}
        
        # Training device
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        """Forward pass through the network"""
        batch_size = x.size(0)
        
        # LSTM layers
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Shared dense processing
        shared_out = torch.relu(self.shared_dense(last_output))
        shared_out = self.dropout(shared_out)
        
        # Individual cohort outputs
        outputs = {}
        for cohort_name in self.cohort_names:
            outputs[cohort_name] = self.output_heads[cohort_name](shared_out)
        
        return outputs
    
    def create_temporal_features(self, datetime_series):
        """Create simple temporal features"""
        # Convert to datetime if it's not already
        if hasattr(datetime_series, 'dt'):
            dt_accessor = datetime_series.dt
        else:
            dt_series = pd.to_datetime(datetime_series)
            dt_accessor = dt_series.dt
            
        df = pd.DataFrame(index=datetime_series.index if hasattr(datetime_series, 'index') else range(len(datetime_series)))
        df['hour_of_day'] = dt_accessor.hour
        df['day_of_week'] = dt_accessor.dayofweek
        df['is_weekend'] = (dt_accessor.dayofweek > 4).astype(int)
        
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
    
    def prepare_sequences(self, grid_data, scenario_name, fit_scalers=False):
        """Convert grid data into LSTM training sequences"""
        print(f"📊 Preparing sequences for {scenario_name}...")
        
        # Validate input data
        self.validate_input_data(grid_data, scenario_name)
        
        # Handle any missing values
        grid_data_clean = grid_data.fillna(method='ffill').fillna(method='bfill')
        
        # Extract features
        temporal_df = self.create_temporal_features(grid_data_clean['date_time'])
        weather_data = grid_data_clean[self.weather_features].values
        temporal_data = temporal_df[self.temporal_features].values
        
        # Get demand data for each cohort
        demand_columns = [col for col in grid_data_clean.columns if col.startswith('demand_mw_')]
        demand_data = grid_data_clean[demand_columns].values
        
        # Scale features
        if fit_scalers:
            weather_data = self.weather_scaler.fit_transform(weather_data)
            for i, cohort in enumerate(self.cohort_names[:len(demand_columns)]):
                cohort_demand = demand_data[:, i].reshape(-1, 1)
                self.demand_scalers[cohort].fit(cohort_demand)
                demand_data[:, i] = self.demand_scalers[cohort].transform(cohort_demand).flatten()
        else:
            weather_data = self.weather_scaler.transform(weather_data)
            for i, cohort in enumerate(self.cohort_names[:len(demand_columns)]):
                cohort_demand = demand_data[:, i].reshape(-1, 1)
                demand_data[:, i] = self.demand_scalers[cohort].transform(cohort_demand).flatten()
        
        # Combine all input features
        all_features = np.concatenate([weather_data, temporal_data, demand_data], axis=1)
        
        # Create sequences
        sequences_X = []
        sequences_y = []
        
        total_length = len(all_features)
        for i in range(self.sequence_length, total_length - self.forecast_horizon + 1):
            seq_x = all_features[i-self.sequence_length:i]
            seq_y = demand_data[i:i+self.forecast_horizon]
            
            sequences_X.append(seq_x)
            sequences_y.append(seq_y)
        
        sequences_X = np.array(sequences_X)
        sequences_y = np.array(sequences_y)
        
        print(f"   Generated {len(sequences_X)} sequences")
        print(f"   Input shape: {sequences_X.shape}")
        print(f"   Output shape: {sequences_y.shape}")
        
        return sequences_X, sequences_y
    
    def train_model(self, all_scenarios, validation_split=0.2, epochs=50, batch_size=32, learning_rate=0.001):
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
        
        # Process remaining scenarios
        for scenario_name in scenario_names[1:]:
            grid_data = all_scenarios[scenario_name]['grid_data']
            X_scenario, y_scenario = self.prepare_sequences(grid_data, scenario_name, fit_scalers=False)
            all_sequences_X.append(X_scenario)
            all_sequences_y.append(y_scenario)
        
        # Combine all training data
        X_combined = np.concatenate(all_sequences_X, axis=0)
        y_combined = np.concatenate(all_sequences_y, axis=0)
        
        print(f"📊 Combined training data: {X_combined.shape[0]:,} sequences")
        
        # Create temporal validation split
        split_idx = int(len(X_combined) * (1 - validation_split))
        X_train = torch.FloatTensor(X_combined[:split_idx]).to(self.device)
        X_val = torch.FloatTensor(X_combined[split_idx:]).to(self.device)
        y_train = torch.FloatTensor(y_combined[:split_idx]).to(self.device)
        y_val = torch.FloatTensor(y_combined[split_idx:]).to(self.device)
        
        print(f"📊 Training samples: {len(X_train):,}")
        print(f"📊 Validation samples: {len(X_val):,}")
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'epochs_trained': 0,
            'best_val_loss': float('inf')
        }
        
        print(f"\n🚀 Training for up to {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = self(batch_X)
                
                # Calculate loss for all cohorts
                total_loss = 0
                for i, cohort in enumerate(self.cohort_names):
                    if i < batch_y.size(2):
                        cohort_pred = outputs[cohort]
                        cohort_target = batch_y[:, :, i]
                        total_loss += criterion(cohort_pred, cohort_target)
                
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self(batch_X)
                    
                    total_loss = 0
                    for i, cohort in enumerate(self.cohort_names):
                        if i < batch_y.size(2):
                            cohort_pred = outputs[cohort]
                            cohort_target = batch_y[:, :, i]
                            total_loss += criterion(cohort_pred, cohort_target)
                    
                    val_loss += total_loss.item()
            
            val_loss /= len(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['epochs_trained'] = epoch + 1
            
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        print("✅ Training completed!")
        return history
    
    def predict_cohort_demands(self, weather_sequence):
        """Generate 24-hour forecasts for all cohorts"""
        self.eval()
        
        if isinstance(weather_sequence, pd.DataFrame):
            # Convert DataFrame to processed sequence
            weather_recent = weather_sequence.tail(self.sequence_length).copy()
            
            # Create temporal features
            temporal_df = self.create_temporal_features(weather_recent['date_time'])
            weather_data = self.weather_scaler.transform(weather_recent[self.weather_features].values)
            temporal_data = temporal_df[self.temporal_features].values
            
            # For prediction, use zeros for historical demand
            demand_data = np.zeros((self.sequence_length, self.n_cohorts))
            
            # Combine features
            all_features = np.concatenate([weather_data, temporal_data, demand_data], axis=1)
            processed_sequence = torch.FloatTensor(all_features).unsqueeze(0).to(self.device)
        else:
            processed_sequence = torch.FloatTensor(weather_sequence).unsqueeze(0).to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            outputs = self(processed_sequence)
        
        # Inverse transform to original MW scale
        forecasts = {}
        for i, cohort in enumerate(self.cohort_names):
            if cohort in outputs:
                cohort_pred = outputs[cohort][0].cpu().numpy()
                # Transform back to MW
                cohort_pred_mw = self.demand_scalers[cohort].inverse_transform(
                    cohort_pred.reshape(-1, 1)
                ).flatten()
                forecasts[cohort] = cohort_pred_mw
        
        return forecasts
    
    def save_model(self, filepath):
        """Save model and scalers for deployment"""
        # Save model state
        torch.save(self.state_dict(), f"{filepath}_model.pth")
        
        # Save configuration and scalers
        config = {
            'weather_scaler': self.weather_scaler,
            'demand_scalers': self.demand_scalers,
            'cohort_names': self.cohort_names,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'weather_features': self.weather_features,
            'temporal_features': self.temporal_features,
            'hidden_size': self.hidden_size,
            'input_size': self.input_size
        }
        joblib.dump(config, f"{filepath}_config.pkl")
        
        print(f"✅ Model saved: {filepath}_model.pth")
        print(f"✅ Config saved: {filepath}_config.pkl")
    
    def load_model(self, filepath):
        """Load trained model and scalers for deployment"""
        # Load configuration
        config = joblib.load(f"{filepath}_config.pkl")
        self.weather_scaler = config['weather_scaler']
        self.demand_scalers = config['demand_scalers']
        self.cohort_names = config['cohort_names']
        self.sequence_length = config['sequence_length']
        self.forecast_horizon = config['forecast_horizon']
        self.weather_features = config.get('weather_features', self.weather_features)
        self.temporal_features = config.get('temporal_features', self.temporal_features)
        
        # Load model state
        self.load_state_dict(torch.load(f"{filepath}_model.pth", map_location=self.device))
        
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
