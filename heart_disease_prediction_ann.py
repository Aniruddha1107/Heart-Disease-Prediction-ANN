# Heart Disease Prediction using Artificial Neural Network (ANN)
# Complete Python Implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class HeartDiseasePredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def generate_dataset(self, n_samples=300):
        """Generate synthetic heart disease dataset"""
        np.random.seed(42)
        n_features = 13
        X = np.random.randn(n_samples, n_features) * np.array([30, 1, 5, 40, 100, 1, 3, 100, 1, 5, 3, 3, 3])
        X[:, 0] += 50  # age
        X = np.abs(X)
        
        # Create target based on features
        y = ((X[:, 0] > 50) & (X[:, 4] > 200) & (X[:, 7] > 100)).astype(int)
        
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        return df
    
    def prepare_data(self, df, test_size=0.2):
        """Preprocess and split the data"""
        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_model(self, input_dim=13):
        """Build the ANN model"""
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=16):
        """Train the model"""
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
        
        cm = confusion_matrix(y_test, y_pred)
        return metrics, cm, y_pred
    
    def visualize_results(self, metrics, cm):
        """Create visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Heart Disease Prediction - ANN Model Analysis', fontsize=16, fontweight='bold')
        
        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy Over Epochs', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss', linewidth=2, color='orange')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
        axes[0, 1].set_title('Model Loss Over Epochs', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        axes[1, 0].set_title('Confusion Matrix - Test Set', fontweight='bold')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
        
        # Metrics bar chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        axes[1, 1].bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[1, 1].set_title('Model Performance Metrics', fontweight='bold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim([0, 1.1])
        for i, v in enumerate(metrics_values):
            axes[1, 1].text(i, v + 0.03, f'{v:.4f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        return fig


def main():
    # Create model instance
    model_pipeline = HeartDiseasePredictionModel()
    
    # Generate dataset
    print("\n" + "="*60)
    print("HEART DISEASE PREDICTION USING ANN")
    print("="*60)
    print("\nStep 1: Generating dataset...")
    df = model_pipeline.generate_dataset(n_samples=300)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    # Prepare data
    print("\nStep 2: Preparing and preprocessing data...")
    X_train, X_test, y_train, y_test = model_pipeline.prepare_data(df)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model
    print("\nStep 3: Building ANN model...")
    model_pipeline.build_model(input_dim=13)
    print(model_pipeline.model.summary())
    
    # Train model
    print("\nStep 4: Training model (50 epochs)...")
    model_pipeline.train_model(X_train, y_train, epochs=50, batch_size=16)
    
    # Evaluate model
    print("\nStep 5: Evaluating model...")
    metrics, cm, y_pred = model_pipeline.evaluate_model(X_test, y_test)
    
    # Print results
    print(f"\nTest Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"\nClassification Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    # Visualize
    print("\nStep 6: Creating visualizations...")
    fig = model_pipeline.visualize_results(metrics, cm)
    plt.show()
    
    print("\n" + "="*60)
    print("PROJECT COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
