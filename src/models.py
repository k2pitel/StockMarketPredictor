"""
Machine learning models for stock price prediction.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle


class LogisticRegressionModel:
    """
    Logistic Regression model for binary classification.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Logistic Regression model.
        
        Args:
            **kwargs: Parameters for LogisticRegression
        """
        default_params = {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        }
        default_params.update(kwargs)
        self.model = LogisticRegression(**default_params)
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Args:
            X: Features
        
        Returns:
            np.ndarray: Prediction probabilities
        """
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class RandomForestModel:
    """
    Random Forest model for binary classification.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Random Forest model.
        
        Args:
            **kwargs: Parameters for RandomForestClassifier
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = RandomForestClassifier(**default_params)
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Args:
            X: Features
        
        Returns:
            np.ndarray: Prediction probabilities
        """
        return self.model.predict_proba(X)
    
    def feature_importance(self):
        """
        Get feature importances.
        
        Returns:
            np.ndarray: Feature importances
        """
        return self.model.feature_importances_
    
    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class NeuralNetworkModel:
    """
    Neural Network model for binary classification using TensorFlow/Keras.
    """
    
    def __init__(self, input_dim, hidden_layers=[64, 32, 16], dropout_rate=0.3):
        """
        Initialize Neural Network model.
        
        Args:
            input_dim (int): Number of input features
            hidden_layers (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the neural network architecture.
        
        Returns:
            keras.Model: Compiled model
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.input_dim,)))
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
        
        Returns:
            History: Training history
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions.
        
        Args:
            X: Features
            threshold: Classification threshold
        
        Returns:
            np.ndarray: Binary predictions
        """
        probabilities = self.model.predict(X, verbose=0)
        return (probabilities > threshold).astype(int).flatten()
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Args:
            X: Features
        
        Returns:
            np.ndarray: Prediction probabilities
        """
        probabilities = self.model.predict(X, verbose=0)
        # Return in sklearn format [prob_class_0, prob_class_1]
        return np.hstack([1 - probabilities, probabilities])
    
    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        self.model.save(filepath)
    
    def load(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        self.model = keras.models.load_model(filepath)


def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    
    Returns:
        dict: Evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
