"""
Main training script for stock market prediction.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import fetch_stock_data, fetch_vix_data, combine_stock_and_vix
from feature_engineering import add_technical_indicators, create_target_variable, prepare_features
from preprocessing import DataPreprocessor, train_test_split_time_series, create_time_series_splits
from models import LogisticRegressionModel, RandomForestModel, NeuralNetworkModel, evaluate_model


def main():
    """
    Main function to run the stock prediction pipeline.
    """
    parser = argparse.ArgumentParser(description='Stock Market Prediction')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--period', type=str, default='5y', help='Data period (e.g., 5y, 10y)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (0-1)')
    parser.add_argument('--models', nargs='+', default=['logistic', 'random_forest', 'neural_network'],
                        help='Models to train (logistic, random_forest, neural_network)')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"Stock Market Predictor - Binary Classification")
    print(f"{'='*60}")
    print(f"Ticker: {args.ticker}")
    print(f"Period: {args.period}")
    print(f"Test Size: {args.test_size}")
    print(f"Models: {', '.join(args.models)}")
    print(f"{'='*60}\n")
    
    # Step 1: Fetch data
    print("Step 1: Fetching stock data...")
    stock_data = fetch_stock_data(args.ticker, period=args.period)
    print(f"Stock data shape: {stock_data.shape}")
    
    print("\nFetching VIX data...")
    vix_data = fetch_vix_data(period=args.period)
    print(f"VIX data shape: {vix_data.shape}")
    
    print("\nCombining stock and VIX data...")
    combined_data = combine_stock_and_vix(stock_data, vix_data)
    print(f"Combined data shape: {combined_data.shape}")
    
    # Step 2: Feature engineering
    print("\nStep 2: Adding technical indicators...")
    data_with_features = add_technical_indicators(combined_data)
    print(f"Data with features shape: {data_with_features.shape}")
    
    print("\nCreating target variable...")
    data_with_target = create_target_variable(data_with_features, horizon=1)
    print(f"Data with target shape: {data_with_target.shape}")
    
    print("\nPreparing features...")
    feature_columns, clean_data = prepare_features(data_with_target)
    print(f"Number of features: {len(feature_columns)}")
    print(f"Clean data shape: {clean_data.shape}")
    print(f"Features: {feature_columns}")
    
    # Step 3: Prepare train/test split
    print("\nStep 3: Splitting data (time-series split)...")
    X = clean_data[feature_columns].values
    y = clean_data['target'].values
    
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, test_size=args.test_size)
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    print(f"Target distribution (train): {np.bincount(y_train)} (0: down, 1: up)")
    print(f"Target distribution (test): {np.bincount(y_test)} (0: down, 1: up)")
    
    # Step 4: Scale features
    print("\nStep 4: Scaling features...")
    preprocessor = DataPreprocessor(scaler_type='standard')
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    print("Features scaled.")
    
    # Step 5: Train models
    print("\nStep 5: Training models...")
    results = {}
    
    if 'logistic' in args.models:
        print("\n" + "="*60)
        print("Training Logistic Regression...")
        print("="*60)
        lr_model = LogisticRegressionModel()
        lr_model.train(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        results['Logistic Regression'] = evaluate_model(y_test, lr_pred, 'Logistic Regression')
    
    if 'random_forest' in args.models:
        print("\n" + "="*60)
        print("Training Random Forest...")
        print("="*60)
        rf_model = RandomForestModel(n_estimators=100, max_depth=10)
        rf_model.train(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        results['Random Forest'] = evaluate_model(y_test, rf_pred, 'Random Forest')
        
        # Feature importance
        print("\nTop 10 Most Important Features:")
        feature_importance = rf_model.feature_importance()
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        print(importance_df.head(10).to_string(index=False))
    
    if 'neural_network' in args.models:
        print("\n" + "="*60)
        print("Training Neural Network...")
        print("="*60)
        nn_model = NeuralNetworkModel(
            input_dim=X_train_scaled.shape[1],
            hidden_layers=[64, 32, 16],
            dropout_rate=0.3
        )
        
        # Split training data for validation
        val_size = int(len(X_train_scaled) * 0.2)
        X_train_nn = X_train_scaled[:-val_size]
        y_train_nn = y_train[:-val_size]
        X_val_nn = X_train_scaled[-val_size:]
        y_val_nn = y_train[-val_size:]
        
        print(f"Training samples: {len(X_train_nn)}, Validation samples: {len(X_val_nn)}")
        
        history = nn_model.train(
            X_train_nn, y_train_nn,
            X_val=X_val_nn, y_val=y_val_nn,
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        nn_pred = nn_model.predict(X_test_scaled)
        results['Neural Network'] = evaluate_model(y_test, nn_pred, 'Neural Network')
    
    # Step 6: Summary
    print("\n" + "="*60)
    print("Final Results Summary")
    print("="*60)
    
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())
    
    # Find best model
    best_model = summary_df['accuracy'].idxmax()
    best_accuracy = summary_df['accuracy'].max()
    print(f"\nBest Model: {best_model} (Accuracy: {best_accuracy:.4f})")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
