"""
Example usage of the Stock Market Predictor.

This script demonstrates how to use the predictor with minimal configuration.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock imports for demonstration when dependencies are not installed
try:
    from data_fetcher import fetch_stock_data, fetch_vix_data, combine_stock_and_vix
    from feature_engineering import add_technical_indicators, create_target_variable, prepare_features
    from preprocessing import DataPreprocessor, train_test_split_time_series
    from models import LogisticRegressionModel, RandomForestModel, NeuralNetworkModel, evaluate_model
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    IMPORTS_AVAILABLE = False


def quick_example():
    """
    Quick example showing the complete pipeline.
    """
    if not IMPORTS_AVAILABLE:
        print("\nDemo code structure:")
        print("=" * 60)
        print("""
# 1. Fetch data
stock_data = fetch_stock_data('AAPL', period='2y')
vix_data = fetch_vix_data(period='2y')
combined_data = combine_stock_and_vix(stock_data, vix_data)

# 2. Engineer features
data_with_features = add_technical_indicators(combined_data)
data_with_target = create_target_variable(data_with_features)
feature_columns, clean_data = prepare_features(data_with_target)

# 3. Prepare data
X = clean_data[feature_columns].values
y = clean_data['target'].values
X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)

# 4. Scale features
preprocessor = DataPreprocessor()
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# 5. Train models
# Logistic Regression
lr_model = LogisticRegressionModel()
lr_model.train(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
evaluate_model(y_test, lr_pred, 'Logistic Regression')

# Random Forest
rf_model = RandomForestModel()
rf_model.train(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
evaluate_model(y_test, rf_pred, 'Random Forest')

# Neural Network
nn_model = NeuralNetworkModel(input_dim=X_train_scaled.shape[1])
nn_model.train(X_train_scaled, y_train, epochs=50)
nn_pred = nn_model.predict(X_test_scaled)
evaluate_model(y_test, nn_pred, 'Neural Network')
        """)
        print("=" * 60)
        return
    
    print("Running quick example on AAPL (Apple) stock...")
    print("=" * 60)
    
    # 1. Fetch data
    print("\n1. Fetching data...")
    stock_data = fetch_stock_data('AAPL', period='2y')
    vix_data = fetch_vix_data(period='2y')
    combined_data = combine_stock_and_vix(stock_data, vix_data)
    print(f"   Data points: {len(combined_data)}")
    
    # 2. Engineer features
    print("\n2. Engineering features...")
    data_with_features = add_technical_indicators(combined_data)
    data_with_target = create_target_variable(data_with_features)
    feature_columns, clean_data = prepare_features(data_with_target)
    print(f"   Features: {len(feature_columns)}")
    print(f"   Clean samples: {len(clean_data)}")
    
    # 3. Prepare data
    print("\n3. Preparing train/test split...")
    X = clean_data[feature_columns].values
    y = clean_data['target'].values
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, test_size=0.2)
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # 4. Scale features
    print("\n4. Scaling features...")
    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    print("   Features scaled")
    
    # 5. Train a quick model (Logistic Regression for speed)
    print("\n5. Training Logistic Regression model...")
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    print("\n6. Results:")
    evaluate_model(y_test, lr_pred, 'Logistic Regression')
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("\nFor full training with all models, run:")
    print("  python train.py --ticker AAPL --period 5y")
    print("=" * 60)


if __name__ == '__main__':
    quick_example()
