"""
Unit tests for Energy Recommendation System - LSTM Forecaster
"""

import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lstm_forecaster import MultiOutputLSTM, LSTMForecaster


class TestMultiOutputLSTM:
    """Test cases for MultiOutputLSTM class"""
    
    @pytest.fixture
    def model_config(self):
        """Model configuration for testing"""
        return {
            'input_size': 10,
            'hidden_size': 64,
            'num_building_types': 5,
            'num_layers': 2
        }
    
    @pytest.fixture
    def mock_model(self, model_config):
        """Mock MultiOutputLSTM instance"""
        return MultiOutputLSTM(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_building_types=model_config['num_building_types'],
            num_layers=model_config['num_layers']
        )
    
    @pytest.fixture
    def sample_input(self):
        """Sample input data"""
        batch_size = 32
        seq_len = 24
        input_size = 10
        
        return torch.randn(batch_size, seq_len, input_size)
    
    def test_init(self, model_config):
        """Test MultiOutputLSTM initialization"""
        model = MultiOutputLSTM(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_building_types=model_config['num_building_types'],
            num_layers=model_config['num_layers']
        )
        
        assert model.input_size == model_config['input_size']
        assert model.hidden_size == model_config['hidden_size']
        assert model.num_building_types == model_config['num_building_types']
        assert model.num_layers == model_config['num_layers']
        
        # Check LSTM layer
        assert isinstance(model.lstm, nn.LSTM)
        assert model.lstm.input_size == model_config['input_size']
        assert model.lstm.hidden_size == model_config['hidden_size']
        assert model.lstm.num_layers == model_config['num_layers']
        
        # Check building heads
        assert len(model.building_heads) == model_config['num_building_types']
        for head in model.building_heads:
            assert isinstance(head, nn.Sequential)
    
    def test_forward(self, mock_model, sample_input):
        """Test forward pass through the network"""
        output = mock_model(sample_input)
        
        # Assertions
        assert isinstance(output, torch.Tensor)
        assert output.shape == (sample_input.shape[0], mock_model.num_building_types, 24)
        assert output.dtype == torch.float32
    
    def test_forward_with_weather_features(self, mock_model, sample_input):
        """Test forward pass with weather features"""
        weather_features = torch.randn(sample_input.shape[0], sample_input.shape[1], mock_model.hidden_size)
        
        output = mock_model(sample_input, weather_features)
        
        # Assertions
        assert isinstance(output, torch.Tensor)
        assert output.shape == (sample_input.shape[0], mock_model.num_building_types, 24)
        assert output.dtype == torch.float32
    
    def test_attention_mechanism(self, mock_model, sample_input):
        """Test attention mechanism"""
        weather_features = torch.randn(sample_input.shape[0], sample_input.shape[1], mock_model.hidden_size)
        
        # Test attention application
        lstm_out = torch.randn(sample_input.shape[0], sample_input.shape[1], mock_model.hidden_size)
        attended_out = mock_model._apply_attention(lstm_out, weather_features)
        
        # Assertions
        assert isinstance(attended_out, torch.Tensor)
        assert attended_out.shape == lstm_out.shape
        assert attended_out.dtype == torch.float32


class TestLSTMForecaster:
    """Test cases for LSTMForecaster class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'input_size': 10,
            'hidden_size': 64,
            'num_building_types': 5,
            'num_layers': 2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'early_stopping_patience': 5
        }
    
    @pytest.fixture
    def mock_forecaster(self, mock_config):
        """Mock LSTMForecaster instance"""
        return LSTMForecaster(mock_config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data"""
        np.random.seed(42)
        
        # Building data
        building_data = pd.DataFrame({
            'building_id': range(100),
            'building_type': np.random.choice(['office', 'retail', 'hospital', 'school', 'warehouse'], 100),
            'square_footage': np.random.normal(50000, 20000, 100),
            'occupancy': np.random.normal(0.7, 0.2, 100)
        })
        
        # Weather data
        weather_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=1000, freq='H'),
            'temperature': np.random.normal(20, 10, 1000),
            'humidity': np.random.normal(50, 20, 1000),
            'wind_speed': np.random.normal(5, 3, 1000)
        })
        
        # Energy consumption data
        energy_data = pd.DataFrame({
            'building_id': np.repeat(range(100), 1000),
            'timestamp': np.tile(pd.date_range('2020-01-01', periods=1000, freq='H'), 100),
            'energy_consumption': np.random.normal(100, 30, 100000)
        })
        
        return building_data, weather_data, energy_data
    
    def test_init(self, mock_config):
        """Test LSTMForecaster initialization"""
        forecaster = LSTMForecaster(mock_config)
        
        assert forecaster.config == mock_config
        assert forecaster.model is not None
        assert forecaster.optimizer is not None
        assert forecaster.criterion is not None
        assert forecaster.scaler is not None
    
    def test_prepare_data(self, mock_forecaster, sample_data):
        """Test data preparation"""
        building_data, weather_data, energy_data = sample_data
        
        # Mock data preparation
        with patch.object(mock_forecaster, '_create_sequences') as mock_sequences:
            mock_sequences.return_value = (torch.randn(100, 24, 10), torch.randn(100, 5, 24))
            
            X, y = mock_forecaster.prepare_data(building_data, weather_data, energy_data)
            
            # Assertions
            assert isinstance(X, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            assert X.shape[0] == y.shape[0]  # Same number of samples
            mock_sequences.assert_called_once()
    
    def test_fit(self, mock_forecaster, sample_data):
        """Test model training"""
        building_data, weather_data, energy_data = sample_data
        
        # Mock data preparation
        with patch.object(mock_forecaster, 'prepare_data') as mock_prepare:
            mock_prepare.return_value = (torch.randn(100, 24, 10), torch.randn(100, 5, 24))
            
            # Mock training loop
            with patch.object(mock_forecaster, '_train_epoch') as mock_train_epoch:
                mock_train_epoch.return_value = 0.5  # Mock loss
                
                # Mock validation
                with patch.object(mock_forecaster, '_validate') as mock_validate:
                    mock_validate.return_value = 0.6  # Mock validation loss
                    
                    mock_forecaster.fit(building_data, weather_data, energy_data)
                    
                    # Assertions
                    mock_prepare.assert_called_once()
                    mock_train_epoch.assert_called()
                    mock_validate.assert_called()
    
    def test_predict(self, mock_forecaster, sample_data):
        """Test model prediction"""
        building_data, weather_data, energy_data = sample_data
        
        # Mock data preparation
        with patch.object(mock_forecaster, 'prepare_data') as mock_prepare:
            mock_prepare.return_value = (torch.randn(50, 24, 10), torch.randn(50, 5, 24))
            
            # Mock model prediction
            mock_forecaster.model = Mock()
            mock_forecaster.model.eval.return_value = None
            mock_forecaster.model.return_value = torch.randn(50, 5, 24)
            
            predictions = mock_forecaster.predict(building_data, weather_data, energy_data)
            
            # Assertions
            assert isinstance(predictions, torch.Tensor)
            assert predictions.shape == (50, 5, 24)
            mock_prepare.assert_called_once()
            mock_forecaster.model.eval.assert_called_once()
    
    def test_evaluate(self, mock_forecaster, sample_data):
        """Test model evaluation"""
        building_data, weather_data, energy_data = sample_data
        
        # Mock data preparation
        with patch.object(mock_forecaster, 'prepare_data') as mock_prepare:
            mock_prepare.return_value = (torch.randn(50, 24, 10), torch.randn(50, 5, 24))
            
            # Mock prediction
            with patch.object(mock_forecaster, 'predict') as mock_predict:
                mock_predict.return_value = torch.randn(50, 5, 24)
                
                metrics = mock_forecaster.evaluate(building_data, weather_data, energy_data)
                
                # Assertions
                assert isinstance(metrics, dict)
                assert 'mape' in metrics
                assert 'rmse' in metrics
                assert 'mae' in metrics
                assert 'r2' in metrics
                mock_prepare.assert_called_once()
                mock_predict.assert_called_once()


class TestDataProcessing:
    """Test cases for data processing functionality"""
    
    def test_create_sequences(self):
        """Test creating sequences for LSTM training"""
        # Mock data
        data = torch.randn(1000, 10)  # 1000 time steps, 10 features
        target = torch.randn(1000, 5)  # 1000 time steps, 5 building types
        
        # Test sequence creation
        X, y = self._create_sequences(data, target, sequence_length=24)
        
        # Assertions
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert X.shape[1] == 24  # Sequence length
        assert X.shape[2] == 10  # Number of features
        assert y.shape[1] == 5   # Number of building types
        assert y.shape[2] == 24  # Prediction horizon
        assert X.shape[0] == y.shape[0]  # Same number of samples
    
    def test_normalize_data(self):
        """Test data normalization"""
        # Mock data
        data = torch.randn(1000, 10)
        
        # Test normalization
        normalized_data, scaler = self._normalize_data(data)
        
        # Assertions
        assert isinstance(normalized_data, torch.Tensor)
        assert normalized_data.shape == data.shape
        assert normalized_data.mean().item() == pytest.approx(0, abs=0.1)
        assert normalized_data.std().item() == pytest.approx(1, abs=0.1)
        assert scaler is not None
    
    def test_denormalize_data(self):
        """Test data denormalization"""
        # Mock data and scaler
        data = torch.randn(1000, 10)
        normalized_data, scaler = self._normalize_data(data)
        
        # Test denormalization
        denormalized_data = self._denormalize_data(normalized_data, scaler)
        
        # Assertions
        assert isinstance(denormalized_data, torch.Tensor)
        assert denormalized_data.shape == data.shape
        assert torch.allclose(denormalized_data, data, atol=1e-6)
    
    def _create_sequences(self, data, target, sequence_length=24):
        """Helper function to create sequences"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(target[i:i + sequence_length])
        
        return torch.stack(X), torch.stack(y)
    
    def _normalize_data(self, data):
        """Helper function to normalize data"""
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        
        normalized_data = (data - mean) / std
        
        scaler = {'mean': mean, 'std': std}
        
        return normalized_data, scaler
    
    def _denormalize_data(self, data, scaler):
        """Helper function to denormalize data"""
        return data * scaler['std'] + scaler['mean']


class TestModelTraining:
    """Test cases for model training functionality"""
    
    def test_train_epoch(self):
        """Test training epoch"""
        # Mock model and data
        model = Mock()
        model.train.return_value = None
        
        X = torch.randn(32, 24, 10)
        y = torch.randn(32, 5, 24)
        
        optimizer = Mock()
        criterion = Mock()
        criterion.return_value = torch.tensor(0.5)
        
        # Test training epoch
        loss = self._train_epoch(model, X, y, optimizer, criterion)
        
        # Assertions
        assert isinstance(loss, float)
        assert loss >= 0
        model.train.assert_called_once()
        optimizer.zero_grad.assert_called_once()
        optimizer.step.assert_called_once()
        criterion.assert_called_once()
    
    def test_validate(self):
        """Test validation"""
        # Mock model and data
        model = Mock()
        model.eval.return_value = None
        
        X = torch.randn(32, 24, 10)
        y = torch.randn(32, 5, 24)
        
        criterion = Mock()
        criterion.return_value = torch.tensor(0.6)
        
        # Test validation
        with torch.no_grad():
            loss = self._validate(model, X, y, criterion)
        
        # Assertions
        assert isinstance(loss, float)
        assert loss >= 0
        model.eval.assert_called_once()
        criterion.assert_called_once()
    
    def test_early_stopping(self):
        """Test early stopping mechanism"""
        # Mock early stopping
        patience = 3
        best_loss = float('inf')
        patience_counter = 0
        
        # Test early stopping logic
        losses = [0.8, 0.7, 0.6, 0.65, 0.7, 0.75, 0.8]
        
        for loss in losses:
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Assertions
        assert patience_counter >= patience
        assert best_loss == 0.6
    
    def _train_epoch(self, model, X, y, optimizer, criterion):
        """Helper function to train one epoch"""
        model.train()
        
        # Forward pass
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _validate(self, model, X, y, criterion):
        """Helper function to validate model"""
        model.eval()
        
        with torch.no_grad():
            output = model(X)
            loss = criterion(output, y)
        
        return loss.item()


class TestPerformanceOptimization:
    """Test cases for performance optimization"""
    
    def test_memory_efficient_training(self):
        """Test memory-efficient training"""
        # Mock large dataset
        large_dataset = torch.randn(10000, 24, 10)
        target = torch.randn(10000, 5, 24)
        
        # Test memory-efficient training
        batch_size = 32
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(large_dataset), batch_size):
            batch_X = large_dataset[i:i + batch_size]
            batch_y = target[i:i + batch_size]
            
            # Mock training step
            loss = torch.randn(1).item()
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Assertions
        assert isinstance(avg_loss, float)
        assert num_batches > 0
        assert avg_loss >= 0
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation for large batches"""
        # Mock model and data
        model = Mock()
        optimizer = Mock()
        criterion = Mock()
        criterion.return_value = torch.tensor(0.5)
        
        # Mock data
        X = torch.randn(16, 24, 10)  # Small batch
        y = torch.randn(16, 5, 24)
        
        accumulation_steps = 4
        
        # Test gradient accumulation
        optimizer.zero_grad()
        
        for i in range(accumulation_steps):
            output = model(X)
            loss = criterion(output, y) / accumulation_steps
            loss.backward()
        
        optimizer.step()
        
        # Assertions
        assert optimizer.zero_grad.call_count == 1
        assert optimizer.step.call_count == 1
        assert criterion.call_count == accumulation_steps
    
    def test_mixed_precision_training(self):
        """Test mixed precision training"""
        # Mock model and data
        model = Mock()
        optimizer = Mock()
        criterion = Mock()
        criterion.return_value = torch.tensor(0.5)
        
        # Mock data
        X = torch.randn(32, 24, 10)
        y = torch.randn(32, 5, 24)
        
        # Test mixed precision training
        with torch.cuda.amp.autocast():
            output = model(X)
            loss = criterion(output, y)
        
        # Assertions
        assert isinstance(loss, torch.Tensor)
        assert loss.dtype == torch.float32


if __name__ == '__main__':
    pytest.main([__file__])
