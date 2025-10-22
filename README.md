# Stock Market Predictor

A machine learning project for predicting stock price movements using historical market data from Yahoo Finance. This project implements binary classification to predict whether a stock will rise tomorrow based on technical indicators and historical patterns.

## Features

### Data Sources
- **Yahoo Finance**: Historical stock data (OHLC prices, volume)
- **VIX**: Volatility Index for market sentiment

### Technical Indicators
- **OHLC Prices**: Open, High, Low, Close
- **Volume**: Trading volume and volume-based features
- **RSI**: Relative Strength Index (14-day)
- **MACD**: Moving Average Convergence Divergence (12/26/9)
- **Bollinger Bands**: Upper, lower, middle bands with position and width
- **Moving Averages**: Simple MA (5, 10, 20, 50, 200 days)
- **Exponential Moving Averages**: EMA (12, 26 days)
- **Momentum**: Price momentum over 1, 5, and 10 days
- **VIX**: Market volatility indicator

### Machine Learning Models
1. **Logistic Regression**: Fast baseline model with good interpretability
2. **Random Forest**: Ensemble method capturing non-linear patterns
3. **Neural Network**: Deep learning model with multiple hidden layers

### Key Features
- Binary classification (1 = stock rises tomorrow, 0 = stock falls/stays same)
- Time-series cross-validation for proper evaluation
- Data preprocessing and standardization
- Feature importance analysis
- Comprehensive performance metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/k2pitel/StockMarketPredictor.git
cd StockMarketPredictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Train all models on Apple (AAPL) stock with default settings:
```bash
python train.py
```

### Advanced Usage

Train specific models on a different stock:
```bash
python train.py --ticker GOOGL --period 10y --test-size 0.2 --models logistic random_forest
```

### Parameters

- `--ticker`: Stock ticker symbol (default: AAPL)
- `--period`: Data period - 1y, 2y, 5y, 10y, max (default: 5y)
- `--test-size`: Test set proportion (default: 0.2 = 20%)
- `--models`: Models to train (default: all three)
  - `logistic`: Logistic Regression
  - `random_forest`: Random Forest
  - `neural_network`: Neural Network

## Project Structure

```
StockMarketPredictor/
├── src/
│   ├── data_fetcher.py          # Fetch stock and VIX data from Yahoo Finance
│   ├── feature_engineering.py   # Create technical indicators
│   ├── preprocessing.py         # Data scaling and time-series splits
│   └── models.py                # ML models (LR, RF, NN)
├── train.py                     # Main training script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## Technical Details

### Data Pipeline

1. **Data Fetching**: Download historical stock and VIX data from Yahoo Finance
2. **Feature Engineering**: Calculate 25+ technical indicators
3. **Target Creation**: Binary label (1 if tomorrow's close > today's close)
4. **Preprocessing**: Standard scaling with time-series aware splitting
5. **Training**: Train models with proper time-series validation
6. **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix

### Time-Series Validation

The project uses time-series aware data splitting to prevent look-ahead bias:
- Training set: Earlier time periods
- Test set: Most recent time period
- No shuffling to maintain temporal order
- Prevents data leakage from future to past

### Model Details

**Logistic Regression**
- Solver: LBFGS
- Max iterations: 1000
- Fast training, good baseline

**Random Forest**
- 100 trees
- Max depth: 10
- Feature importance analysis included

**Neural Network**
- Architecture: Input → 64 → 32 → 16 → 1
- Dropout: 0.3 for regularization
- Early stopping with patience=10
- Binary crossentropy loss

## Example Output

```
============================================================
Stock Market Predictor - Binary Classification
============================================================
Ticker: AAPL
Period: 5y
Test Size: 0.2
Models: logistic, random_forest, neural_network
============================================================

Step 1: Fetching stock data...
Stock data shape: (1258, 6)

Step 2: Adding technical indicators...
Data with features shape: (1258, 29)

Step 3: Splitting data (time-series split)...
Training set: (812, 27), (812,)
Test set: (203, 27), (203,)

Step 4: Scaling features...

Step 5: Training models...

Logistic Regression Performance:
Accuracy: 0.5468
Precision: 0.5632
Recall: 0.7234
F1-Score: 0.6332

Random Forest Performance:
Accuracy: 0.5616
Precision: 0.5789
Recall: 0.6879
F1-Score: 0.6287

Neural Network Performance:
Accuracy: 0.5567
Precision: 0.5701
Recall: 0.7123
F1-Score: 0.6333

============================================================
Final Results Summary
============================================================
                    accuracy  precision    recall  f1_score
Logistic Regression   0.5468     0.5632    0.7234    0.6332
Random Forest         0.5616     0.5789    0.6879    0.6287
Neural Network        0.5567     0.5701    0.7123    0.6333

Best Model: Random Forest (Accuracy: 0.5616)
```

## Performance Notes

Stock market prediction is inherently challenging due to:
- Market efficiency hypothesis
- Random walk theory
- Unpredictable external events
- High noise-to-signal ratio

Expected accuracy is typically in the 52-58% range for binary classification. Even small improvements over 50% can be valuable with proper risk management.

## Dependencies

- yfinance: Yahoo Finance data
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: ML models and preprocessing
- tensorflow: Neural network implementation
- ta: Technical analysis indicators
- matplotlib, seaborn: Visualization

## License

This project is open source and available for educational purposes.

## Disclaimer

This project is for educational and research purposes only. It should not be used as the sole basis for actual trading decisions. Past performance does not guarantee future results. Always conduct thorough research and consider consulting financial advisors before making investment decisions.