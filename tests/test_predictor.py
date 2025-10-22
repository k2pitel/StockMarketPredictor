"""
Unit tests for the Stock Market Predictor.

Note: These tests require the dependencies to be installed.
Run: pip install -r requirements.txt
"""
import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from feature_engineering import create_target_variable, prepare_features
    from preprocessing import DataPreprocessor, train_test_split_time_series
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("Warning: Dependencies not installed. Install with: pip install -r requirements.txt")


class TestTargetCreation(unittest.TestCase):
    """Test target variable creation."""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Dependencies not installed")
    def test_target_binary(self):
        """Test that target is binary (0 or 1)."""
        # Create sample data
        df = pd.DataFrame({
            'Close': [100, 101, 99, 102, 98]
        })
        
        result = create_target_variable(df, horizon=1)
        
        # Target should only contain 0 and 1
        unique_values = result['target'].unique()
        self.assertTrue(all(val in [0, 1] for val in unique_values))
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Dependencies not installed")
    def test_target_rising_stock(self):
        """Test that rising stock gets target=1."""
        df = pd.DataFrame({
            'Close': [100, 105]  # Stock rises
        })
        
        result = create_target_variable(df, horizon=1)
        
        # First day should have target=1 (price rises next day)
        self.assertEqual(result['target'].iloc[0], 1)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Dependencies not installed")
    def test_target_falling_stock(self):
        """Test that falling stock gets target=0."""
        df = pd.DataFrame({
            'Close': [100, 95]  # Stock falls
        })
        
        result = create_target_variable(df, horizon=1)
        
        # First day should have target=0 (price falls next day)
        self.assertEqual(result['target'].iloc[0], 0)


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Dependencies not installed")
    def test_scaler_fit_transform(self):
        """Test that scaler properly transforms data."""
        preprocessor = DataPreprocessor(scaler_type='standard')
        
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_scaled = preprocessor.fit_transform(X_train)
        
        # Check that scaling was applied
        self.assertEqual(X_scaled.shape, X_train.shape)
        # After standard scaling, mean should be close to 0
        self.assertAlmostEqual(np.mean(X_scaled), 0, places=10)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Dependencies not installed")
    def test_time_series_split(self):
        """Test that time series split maintains order."""
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        
        X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, test_size=0.2)
        
        # Check split sizes
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        
        # Check that train comes before test (time order preserved)
        self.assertTrue(X_train[-1] < X_test[0])
        self.assertTrue(y_train[-1] < y_test[0])


class TestModels(unittest.TestCase):
    """Test model classes."""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Dependencies not installed")
    def test_logistic_regression_predict(self):
        """Test that Logistic Regression can train and predict."""
        from models import LogisticRegressionModel
        
        # Simple binary classification data
        X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[0.5, 0.5], [2.5, 2.5]])
        
        model = LogisticRegressionModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Check predictions are binary
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        self.assertEqual(len(predictions), len(X_test))
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Dependencies not installed")
    def test_random_forest_predict(self):
        """Test that Random Forest can train and predict."""
        from models import RandomForestModel
        
        # Simple binary classification data
        X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[0.5, 0.5], [2.5, 2.5]])
        
        model = RandomForestModel(n_estimators=10)
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Check predictions are binary
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        self.assertEqual(len(predictions), len(X_test))
        
        # Check feature importance exists
        importance = model.feature_importance()
        self.assertEqual(len(importance), 2)  # 2 features


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Dependencies not installed")
    def test_minimal_pipeline(self):
        """Test that minimal pipeline runs without errors."""
        from feature_engineering import add_technical_indicators
        
        # Create minimal stock data
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        df = pd.DataFrame({
            'Open': np.random.uniform(90, 110, 300),
            'High': np.random.uniform(95, 115, 300),
            'Low': np.random.uniform(85, 105, 300),
            'Close': np.random.uniform(90, 110, 300),
            'Volume': np.random.randint(1000000, 10000000, 300)
        }, index=dates)
        
        # Add technical indicators
        df_with_features = add_technical_indicators(df)
        
        # Check that features were added
        self.assertIn('RSI', df_with_features.columns)
        self.assertIn('MACD', df_with_features.columns)
        self.assertIn('BB_high', df_with_features.columns)
        self.assertIn('MA_5', df_with_features.columns)
        
        # Create target
        df_with_target = create_target_variable(df_with_features)
        
        # Check target exists
        self.assertIn('target', df_with_target.columns)
        
        # Prepare features
        feature_columns, clean_data = prepare_features(df_with_target)
        
        # Check we have features and data
        self.assertTrue(len(feature_columns) > 0)
        self.assertTrue(len(clean_data) > 0)


def run_tests():
    """Run all tests."""
    print("Running Stock Market Predictor Tests")
    print("=" * 60)
    
    if not IMPORTS_AVAILABLE:
        print("\nWARNING: Dependencies not installed.")
        print("Install with: pip install -r requirements.txt")
        print("\nShowing test structure only...\n")
    
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
