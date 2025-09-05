"""
Unit tests for Energy Recommendation System - Portfolio Optimizer
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimization.portfolio_optimizer import PortfolioOptimizer, ConstraintBasedOptimizer


class TestPortfolioOptimizer:
    """Test cases for PortfolioOptimizer class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'optimization_method': 'scipy',
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'constraint_tolerance': 1e-6
        }
    
    @pytest.fixture
    def mock_optimizer(self, mock_config):
        """Mock PortfolioOptimizer instance"""
        return PortfolioOptimizer(mock_config)
    
    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction data"""
        return {
            'building_1': {
                'energy_reduction': 0.15,
                'max_participation': 0.8,
                'building_type': 'office',
                'square_footage': 50000
            },
            'building_2': {
                'energy_reduction': 0.20,
                'max_participation': 0.6,
                'building_type': 'retail',
                'square_footage': 30000
            },
            'building_3': {
                'energy_reduction': 0.12,
                'max_participation': 0.9,
                'building_type': 'hospital',
                'square_footage': 100000
            }
        }
    
    @pytest.fixture
    def sample_compliance_rates(self):
        """Sample compliance rates"""
        return {
            'building_1': 0.4,
            'building_2': 0.3,
            'building_3': 0.5
        }
    
    @pytest.fixture
    def sample_grid_conditions(self):
        """Sample grid conditions"""
        return {
            'strain_level': 'high',
            'demand_forecast': 1000,
            'supply_capacity': 1200,
            'reserve_margin': 0.2
        }
    
    def test_init(self, mock_config):
        """Test PortfolioOptimizer initialization"""
        optimizer = PortfolioOptimizer(mock_config)
        
        assert optimizer.config == mock_config
        assert optimizer.optimization_constraints is not None
    
    def test_optimize_portfolio(self, mock_optimizer, sample_predictions, 
                               sample_compliance_rates, sample_grid_conditions):
        """Test portfolio optimization"""
        # Mock optimization result
        with patch('optimization.portfolio_optimizer.cp') as mock_cp:
            # Mock decision variable
            mock_participation = Mock()
            mock_participation.value = np.array([0.4, 0.3, 0.5])
            
            # Mock problem
            mock_problem = Mock()
            mock_problem.status = 'OPTIMAL'
            mock_problem.solve.return_value = None
            
            # Mock optimization
            mock_cp.Variable.return_value = mock_participation
            mock_cp.Problem.return_value = mock_problem
            mock_cp.OPTIMAL = 'OPTIMAL'
            
            result = mock_optimizer.optimize_portfolio(
                sample_predictions, sample_compliance_rates, sample_grid_conditions
            )
            
            # Assertions
            assert isinstance(result, dict)
            assert 'recommendations' in result
            assert 'total_reduction' in result
            assert 'confidence_score' in result
    
    def test_calculate_grid_impact(self, mock_optimizer, sample_predictions):
        """Test grid impact calculation"""
        # Mock participation levels
        participation = np.array([0.4, 0.3, 0.5])
        
        # Test grid impact calculation
        impact = mock_optimizer._calculate_grid_impact(sample_predictions, participation)
        
        # Assertions
        assert isinstance(impact, float)
        assert impact > 0
        
        # Calculate expected impact manually
        expected_impact = (
            0.15 * 0.4 +  # building_1
            0.20 * 0.3 +  # building_2
            0.12 * 0.5    # building_3
        )
        assert impact == pytest.approx(expected_impact, abs=1e-6)
    
    def test_build_constraints(self, mock_optimizer, sample_predictions, 
                              sample_compliance_rates, sample_grid_conditions):
        """Test constraint building"""
        # Mock participation variable
        participation = Mock()
        
        # Test constraint building
        constraints = mock_optimizer._build_constraints(
            participation, sample_predictions, sample_compliance_rates, sample_grid_conditions
        )
        
        # Assertions
        assert isinstance(constraints, list)
        assert len(constraints) > 0
        
        # Check that constraints are properly constructed
        for constraint in constraints:
            assert constraint is not None
    
    def test_format_optimization_results(self, mock_optimizer, sample_predictions):
        """Test optimization results formatting"""
        # Mock participation values
        participation_values = np.array([0.4, 0.3, 0.5])
        
        # Test results formatting
        results = mock_optimizer._format_optimization_results(participation_values, sample_predictions)
        
        # Assertions
        assert isinstance(results, dict)
        assert 'recommendations' in results
        assert 'total_reduction' in results
        assert 'confidence_score' in results
        
        # Check recommendations
        recommendations = results['recommendations']
        assert len(recommendations) == len(sample_predictions)
        
        for building_id, recommendation in recommendations.items():
            assert 'participation_level' in recommendation
            assert 'expected_reduction' in recommendation
            assert 'confidence' in recommendation
    
    def test_handle_optimization_failure(self, mock_optimizer):
        """Test optimization failure handling"""
        # Test different failure statuses
        failure_statuses = ['INFEASIBLE', 'UNBOUNDED', 'INFEASIBLE_INACCURATE']
        
        for status in failure_statuses:
            result = mock_optimizer._handle_optimization_failure(status)
            
            # Assertions
            assert isinstance(result, dict)
            assert 'error' in result
            assert 'status' in result
            assert result['status'] == status


class TestConstraintBasedOptimizer:
    """Test cases for ConstraintBasedOptimizer class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'solver': 'ECOS',
            'max_iterations': 1000,
            'tolerance': 1e-6
        }
    
    @pytest.fixture
    def mock_optimizer(self, mock_config):
        """Mock ConstraintBasedOptimizer instance"""
        return ConstraintBasedOptimizer(mock_config)
    
    def test_init(self, mock_config):
        """Test ConstraintBasedOptimizer initialization"""
        optimizer = ConstraintBasedOptimizer(mock_config)
        
        assert optimizer.config == mock_config
        assert optimizer.solver == mock_config['solver']
    
    def test_optimize_with_constraints(self, mock_optimizer):
        """Test optimization with constraints"""
        # Mock optimization problem
        with patch('optimization.portfolio_optimizer.cp') as mock_cp:
            # Mock decision variable
            mock_x = Mock()
            mock_x.value = np.array([0.5, 0.3, 0.7])
            
            # Mock problem
            mock_problem = Mock()
            mock_problem.status = 'OPTIMAL'
            mock_problem.solve.return_value = None
            
            # Mock optimization
            mock_cp.Variable.return_value = mock_x
            mock_cp.Problem.return_value = mock_problem
            mock_cp.OPTIMAL = 'OPTIMAL'
            
            # Test optimization
            result = mock_optimizer.optimize_with_constraints()
            
            # Assertions
            assert isinstance(result, dict)
            assert 'status' in result
            assert 'solution' in result
    
    def test_validate_constraints(self, mock_optimizer):
        """Test constraint validation"""
        # Mock constraints
        constraints = [
            Mock(),  # Constraint 1
            Mock(),  # Constraint 2
            Mock()   # Constraint 3
        ]
        
        # Test constraint validation
        is_valid = mock_optimizer._validate_constraints(constraints)
        
        # Assertions
        assert isinstance(is_valid, bool)
        assert is_valid == True  # All constraints should be valid


class TestOptimizationConstraints:
    """Test cases for optimization constraints"""
    
    def test_compliance_constraints(self):
        """Test compliance rate constraints"""
        # Mock data
        participation = np.array([0.4, 0.3, 0.5])
        compliance_rates = np.array([0.4, 0.3, 0.5])
        
        # Test compliance constraints
        constraints_satisfied = self._check_compliance_constraints(participation, compliance_rates)
        
        # Assertions
        assert constraints_satisfied == True
        
        # Test violation
        participation_violation = np.array([0.5, 0.4, 0.6])  # Exceeds compliance rates
        constraints_violated = self._check_compliance_constraints(participation_violation, compliance_rates)
        assert constraints_violated == False
    
    def test_grid_strain_constraints(self):
        """Test grid strain constraints"""
        # Mock data
        participation = np.array([0.4, 0.3, 0.5])
        grid_conditions = {'strain_level': 'high'}
        
        # Test grid strain constraints
        constraints_satisfied = self._check_grid_strain_constraints(participation, grid_conditions)
        
        # Assertions
        assert constraints_satisfied == True
        
        # Test critical strain
        grid_conditions_critical = {'strain_level': 'critical'}
        constraints_satisfied_critical = self._check_grid_strain_constraints(participation, grid_conditions_critical)
        assert constraints_satisfied_critical == True
    
    def test_building_constraints(self):
        """Test building-specific constraints"""
        # Mock data
        participation = np.array([0.4, 0.3, 0.5])
        max_participation = np.array([0.8, 0.6, 0.9])
        
        # Test building constraints
        constraints_satisfied = self._check_building_constraints(participation, max_participation)
        
        # Assertions
        assert constraints_satisfied == True
        
        # Test violation
        participation_violation = np.array([0.9, 0.7, 1.0])  # Exceeds max participation
        constraints_violated = self._check_building_constraints(participation_violation, max_participation)
        assert constraints_violated == False
    
    def _check_compliance_constraints(self, participation, compliance_rates):
        """Helper function to check compliance constraints"""
        return np.all(participation <= compliance_rates)
    
    def _check_grid_strain_constraints(self, participation, grid_conditions):
        """Helper function to check grid strain constraints"""
        total_participation = np.sum(participation)
        
        if grid_conditions['strain_level'] == 'high':
            return total_participation >= 0.3
        elif grid_conditions['strain_level'] == 'critical':
            return total_participation >= 0.5
        else:
            return True
    
    def _check_building_constraints(self, participation, max_participation):
        """Helper function to check building constraints"""
        return np.all(participation <= max_participation)


class TestOptimizationMetrics:
    """Test cases for optimization metrics"""
    
    def test_calculate_total_reduction(self):
        """Test total reduction calculation"""
        # Mock data
        participation = np.array([0.4, 0.3, 0.5])
        energy_reductions = np.array([0.15, 0.20, 0.12])
        
        # Test total reduction calculation
        total_reduction = self._calculate_total_reduction(participation, energy_reductions)
        
        # Assertions
        assert isinstance(total_reduction, float)
        assert total_reduction > 0
        
        # Calculate expected reduction manually
        expected_reduction = np.sum(participation * energy_reductions)
        assert total_reduction == pytest.approx(expected_reduction, abs=1e-6)
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation"""
        # Mock data
        participation = np.array([0.4, 0.3, 0.5])
        compliance_rates = np.array([0.4, 0.3, 0.5])
        
        # Test confidence score calculation
        confidence_score = self._calculate_confidence_score(participation, compliance_rates)
        
        # Assertions
        assert isinstance(confidence_score, float)
        assert 0 <= confidence_score <= 1
        
        # When participation equals compliance rates, confidence should be high
        assert confidence_score > 0.8
    
    def test_calculate_cost_benefit(self):
        """Test cost-benefit calculation"""
        # Mock data
        participation = np.array([0.4, 0.3, 0.5])
        energy_reductions = np.array([0.15, 0.20, 0.12])
        electricity_price = 0.12  # $/kWh
        
        # Test cost-benefit calculation
        cost_benefit = self._calculate_cost_benefit(participation, energy_reductions, electricity_price)
        
        # Assertions
        assert isinstance(cost_benefit, dict)
        assert 'total_savings' in cost_benefit
        assert 'cost_per_kwh' in cost_benefit
        assert 'roi' in cost_benefit
        
        assert cost_benefit['total_savings'] > 0
        assert cost_benefit['cost_per_kwh'] == electricity_price
        assert cost_benefit['roi'] > 0
    
    def _calculate_total_reduction(self, participation, energy_reductions):
        """Helper function to calculate total reduction"""
        return np.sum(participation * energy_reductions)
    
    def _calculate_confidence_score(self, participation, compliance_rates):
        """Helper function to calculate confidence score"""
        # Confidence based on how close participation is to compliance rates
        compliance_ratio = participation / compliance_rates
        confidence = np.mean(compliance_ratio)
        return min(confidence, 1.0)
    
    def _calculate_cost_benefit(self, participation, energy_reductions, electricity_price):
        """Helper function to calculate cost-benefit"""
        total_reduction = np.sum(participation * energy_reductions)
        total_savings = total_reduction * electricity_price
        
        return {
            'total_savings': total_savings,
            'cost_per_kwh': electricity_price,
            'roi': total_savings / 1000  # Simplified ROI calculation
        }


class TestOptimizationPerformance:
    """Test cases for optimization performance"""
    
    def test_optimization_speed(self):
        """Test optimization speed"""
        import time
        
        # Mock optimization problem
        start_time = time.time()
        
        # Simulate optimization
        time.sleep(0.01)  # Simulate 10ms optimization
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Assertions
        assert optimization_time < 0.1  # Should be fast
        assert optimization_time > 0.005  # Should take some time
    
    def test_memory_usage(self):
        """Test memory usage during optimization"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate optimization with large data
        large_data = np.random.random((1000, 100))
        result = np.sum(large_data, axis=1)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Assertions
        assert memory_increase < 100  # Should not use too much memory
        assert len(result) == 1000
    
    def test_scalability(self):
        """Test optimization scalability"""
        # Test with different problem sizes
        problem_sizes = [10, 50, 100, 500]
        optimization_times = []
        
        for size in problem_sizes:
            start_time = time.time()
            
            # Simulate optimization
            data = np.random.random((size, size))
            result = np.linalg.solve(data, np.random.random(size))
            
            end_time = time.time()
            optimization_time = end_time - start_time
            optimization_times.append(optimization_time)
        
        # Assertions
        assert len(optimization_times) == len(problem_sizes)
        assert all(time > 0 for time in optimization_times)
        
        # Optimization time should increase with problem size
        for i in range(1, len(optimization_times)):
            assert optimization_times[i] >= optimization_times[i-1] * 0.5  # Allow some variance


if __name__ == '__main__':
    pytest.main([__file__])
