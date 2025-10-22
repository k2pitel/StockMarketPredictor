# Setup Guide for Stock Market Predictor

This guide will help you set up and run the Stock Market Predictor.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (for downloading stock data)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/k2pitel/StockMarketPredictor.git
cd StockMarketPredictor
```

### 2. Create a Virtual Environment (Recommended)

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- yfinance (Yahoo Finance data)
- pandas (data manipulation)
- numpy (numerical operations)
- scikit-learn (ML models)
- tensorflow (neural networks)
- matplotlib, seaborn (visualization)
- ta (technical analysis indicators)

### 4. Verify Installation

```bash
python example.py
```

This should run a quick demonstration of the predictor.

## Quick Start

### Run with Default Settings (Apple Stock, 5 years)

```bash
python train.py
```

### Run with Custom Settings

```bash
# Google stock, 10 years of data
python train.py --ticker GOOGL --period 10y

# Microsoft stock, train only Random Forest
python train.py --ticker MSFT --models random_forest

# Tesla stock, smaller test set
python train.py --ticker TSLA --test-size 0.1
```

## Available Parameters

- `--ticker`: Stock symbol (default: AAPL)
  - Examples: AAPL, GOOGL, MSFT, TSLA, AMZN, etc.

- `--period`: Time period to fetch (default: 5y)
  - Options: 1y, 2y, 5y, 10y, max

- `--test-size`: Proportion for test set (default: 0.2)
  - Range: 0.0 to 1.0
  - Example: 0.2 means 20% for testing

- `--models`: Which models to train (default: all)
  - Options: logistic, random_forest, neural_network
  - Can specify multiple: `--models logistic random_forest`

## Examples

### 1. Train All Models on Apple Stock

```bash
python train.py --ticker AAPL --period 5y
```

### 2. Quick Test with Logistic Regression Only

```bash
python train.py --ticker AAPL --period 2y --models logistic
```

### 3. Train on Multiple Stocks (Run Separately)

```bash
python train.py --ticker AAPL --period 5y
python train.py --ticker GOOGL --period 5y
python train.py --ticker MSFT --period 5y
```

### 4. Focus on Recent Data

```bash
python train.py --ticker AAPL --period 1y --test-size 0.3
```

## Running Tests

To run the unit tests:

```bash
python -m pytest tests/test_predictor.py -v
```

Or using unittest:

```bash
python tests/test_predictor.py
```

## Understanding the Output

The training script will output:

1. **Data Information**: Number of samples, features
2. **Target Distribution**: How many days stock went up vs. down
3. **Model Performance**:
   - Accuracy: Overall correctness
   - Precision: When model predicts "up", how often is it right?
   - Recall: Of all "up" days, how many did model catch?
   - F1-Score: Harmonic mean of precision and recall
4. **Confusion Matrix**: True positives, false positives, etc.
5. **Feature Importance**: (Random Forest only) Most influential features

## Troubleshooting

### Issue: Module not found

**Solution**: Make sure you've activated the virtual environment and installed dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Ticker not found

**Solution**: Verify the ticker symbol is correct. Visit Yahoo Finance to check valid symbols.

### Issue: Not enough data

**Solution**: Some stocks may not have data for the full period. Try a shorter period:
```bash
python train.py --ticker YOUR_TICKER --period 2y
```

### Issue: Memory error with Neural Network

**Solution**: The neural network requires more memory. Try:
- Reduce the period: `--period 2y`
- Skip the neural network: `--models logistic random_forest`

### Issue: Slow training

**Solution**: 
- Random Forest is fastest
- Neural Network is slowest
- Train one model at a time: `--models logistic`

## Performance Expectations

Stock market prediction is inherently difficult. Expected performance:

- **Random prediction**: 50% accuracy
- **Good model**: 52-58% accuracy
- **Excellent model**: 58-65% accuracy

Even small improvements over 50% can be valuable with proper risk management.

## Next Steps

1. **Experiment with different stocks**
2. **Try different time periods**
3. **Analyze feature importance** (Random Forest output)
4. **Modify hyperparameters** in `config.py`
5. **Add your own features** in `src/feature_engineering.py`

## Configuration

Edit `config.py` to customize:
- Model hyperparameters
- Technical indicator settings
- Feature engineering options
- Random seeds

## Getting Help

- Check the README.md for detailed documentation
- Review code comments in src/ directory
- Run example.py for a quick demonstration
- Check tests/test_predictor.py for usage examples

## Safety Reminder

⚠️ **Important**: This tool is for educational purposes only. Do not use it as the sole basis for trading decisions. Always:
- Do your own research
- Understand the risks
- Consider consulting financial advisors
- Never invest more than you can afford to lose
