# Quick Reference Guide

## Quick Start (5 Minutes)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with defaults** (AAPL stock, 5 years, all models):
   ```bash
   python train.py
   ```

3. **Check results** in terminal output

## Common Commands

### Different Stocks
```bash
python train.py --ticker GOOGL    # Google
python train.py --ticker MSFT     # Microsoft
python train.py --ticker TSLA     # Tesla
python train.py --ticker AMZN     # Amazon
```

### Different Time Periods
```bash
python train.py --period 1y       # 1 year
python train.py --period 2y       # 2 years
python train.py --period 5y       # 5 years (default)
python train.py --period 10y      # 10 years
```

### Specific Models
```bash
python train.py --models logistic              # Fastest
python train.py --models random_forest         # Most accurate
python train.py --models neural_network        # Deep learning
python train.py --models logistic random_forest # Multiple
```

### Combined Examples
```bash
# Quick test: Tesla, 2 years, Logistic Regression only
python train.py --ticker TSLA --period 2y --models logistic

# Full analysis: Google, 10 years, all models
python train.py --ticker GOOGL --period 10y

# Conservative split: Apple, 5 years, 10% test
python train.py --ticker AAPL --period 5y --test-size 0.1
```

## File Guide

| File | Purpose |
|------|---------|
| `train.py` | Main training script - run this |
| `example.py` | Quick demo without full training |
| `config.py` | Modify hyperparameters here |
| `requirements.txt` | Install these dependencies |
| `README.md` | Full documentation |
| `SETUP.md` | Detailed setup instructions |
| `src/` | Source code modules |
| `tests/` | Unit tests |

## Parameters Reference

| Parameter | Options | Default | Example |
|-----------|---------|---------|---------|
| `--ticker` | Any stock symbol | AAPL | `--ticker MSFT` |
| `--period` | 1y, 2y, 5y, 10y, max | 5y | `--period 10y` |
| `--test-size` | 0.0 to 1.0 | 0.2 | `--test-size 0.3` |
| `--models` | logistic, random_forest, neural_network | all | `--models logistic` |

## Understanding Output

### What the numbers mean:

- **Accuracy**: % of correct predictions (higher is better)
  - 50% = random guessing
  - 55%+ = good model
  - 60%+ = excellent model

- **Precision**: When model says "up", how often is it right?
  - Higher = fewer false alarms

- **Recall**: Of all "up" days, how many did model catch?
  - Higher = catches more opportunities

- **F1-Score**: Balance of precision and recall
  - Best overall metric

### Confusion Matrix:
```
[[TN FP]   TN = Correctly predicted down
 [FN TP]]  TP = Correctly predicted up
           FP = False alarm (predicted up, went down)
           FN = Missed opportunity (predicted down, went up)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | Run `pip install -r requirements.txt` |
| "Ticker not found" | Check symbol on Yahoo Finance |
| "Not enough data" | Try shorter period: `--period 2y` |
| Training too slow | Use fewer models: `--models logistic` |
| Memory error | Reduce period or skip neural network |

## Tips

1. **Start simple**: Use Logistic Regression first
2. **More data ≠ always better**: 5 years is usually enough
3. **Check feature importance**: Random Forest shows which indicators matter
4. **Compare models**: Different stocks may favor different models
5. **Understand limitations**: Even 55% accuracy is good for stock prediction

## Model Comparison

| Model | Speed | Accuracy | Interpretability | When to Use |
|-------|-------|----------|------------------|-------------|
| Logistic Regression | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | Quick tests, baseline |
| Random Forest | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | Best all-around |
| Neural Network | ⚡ | ⭐⭐⭐ | ⭐ | Large datasets, complex patterns |

## Code Examples

### Modify hyperparameters in `config.py`:
```python
RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,    # More trees
    'max_depth': 15,        # Deeper trees
}
```

### Add custom features in `src/feature_engineering.py`:
```python
# Add your custom indicator
df['custom_indicator'] = df['Close'].rolling(window=30).mean()
```

## Testing

```bash
# Run all tests
python tests/test_predictor.py

# Run example
python example.py
```

## Best Practices

1. ✅ Always use time-series split (default)
2. ✅ Compare multiple models
3. ✅ Check feature importance
4. ✅ Validate on recent data (test set)
5. ❌ Don't use for real trading without additional research
6. ❌ Don't expect >60% accuracy consistently
7. ❌ Don't ignore risk management

## Popular Stocks to Try

```bash
# Tech
python train.py --ticker AAPL    # Apple
python train.py --ticker GOOGL   # Google
python train.py --ticker MSFT    # Microsoft
python train.py --ticker TSLA    # Tesla

# Finance
python train.py --ticker JPM     # JP Morgan
python train.py --ticker BAC     # Bank of America

# Indices
python train.py --ticker ^GSPC   # S&P 500
python train.py --ticker ^DJI    # Dow Jones
```

## Getting Help

1. Check `README.md` for details
2. Read `SETUP.md` for installation help
3. Review `IMPLEMENTATION.md` for technical details
4. Look at code comments in `src/`
5. Run `python train.py --help`

## Safety Warning

⚠️ **This is educational software**
- Do not use as sole basis for trading
- Stock market has inherent risks
- Past performance ≠ future results
- Always do your own research
- Consider professional financial advice
