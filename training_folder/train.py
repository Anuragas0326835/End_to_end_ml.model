import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Configuration
DATA_FILE = "gym_attendance.csv"
MODEL_FILE = "gym_rf_model.pkl"
EXPERIMENT_NAME = "Gym_Attendance_Prediction"

def generate_dummy_data(filename):
    """Generates synthetic data if the file doesn't exist."""
    print(f"Generating dummy data: {filename}")
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    data = {
        'Date': dates,
        'Sleep_Hours': np.random.normal(7, 1.5, n),
        'Mood': np.random.choice(['Happy', 'Neutral', 'Stressed', 'Tired'], n),
        'Work_Load': np.random.choice(['Low', 'Medium', 'High'], n),
        'Weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], n),
        # Target: 1 = Went to gym, 0 = Skipped
        'Attended': np.random.choice([0, 1], n, p=[0.4, 0.6]) 
    }
    df = pd.DataFrame(data)
    # Introduce some missing values for cleaning demo
    df.loc[::50, 'Sleep_Hours'] = np.nan
    df.to_csv(filename, index=False)
    return df

def load_data(filename):
    if not os.path.exists(filename):
        return generate_dummy_data(filename)
    return pd.read_csv(filename)

def data_cleaning(df):
    print("--- Starting Data Cleaning ---")
    initial_shape = df.shape
    
    # 1. Handle Missing Values
    if df.isnull().sum().sum() > 0:
        print(f"Found missing values:\n{df.isnull().sum()}")
        # Fill numeric with median, categorical with mode
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # 2. Remove Duplicates
    df.drop_duplicates(inplace=True)
    
    print(f"Data cleaned. Shape changed from {initial_shape} to {df.shape}")
    return df

def feature_engineering(df):
    print("--- Starting Feature Engineering ---")
    
    # Convert Date to datetime objects
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract features
        df['Day_Of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Is_Weekend'] = df['Day_Of_Week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Drop original Date column as models can't handle raw dates usually
        df.drop('Date', axis=1, inplace=True)
        
    return df

def perform_eda(df):
    print("--- Performing EDA & Visualization ---")
    
    # Create a directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Attended', data=df)
    plt.title('Distribution of Gym Attendance')
    plt.savefig("plots/target_dist.png")
    plt.close()
    
    # 2. Sleep vs Attendance
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Attended', y='Sleep_Hours', data=df)
    plt.title('Sleep Hours vs Attendance')
    plt.savefig("plots/sleep_vs_attendance.png")
    plt.close()
    
    print("EDA plots saved to 'plots/' directory.")

def preprocessing(df):
    print("--- Preprocessing Data ---")
    
    # Separate Target and Features
    X = df.drop('Attended', axis=1)
    y = df['Attended']
    
    # Encoding Categorical Variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
    # Scaling Numerical Variables
    scaler = StandardScaler()
    numerical_cols = ['Sleep_Hours'] # Add others if available
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, scaler

def train_model():
    # 1. Load and Prepare
    df = load_data(DATA_FILE)
    df = data_cleaning(df)
    df = feature_engineering(df)
    perform_eda(df)
    X, y, scaler = preprocessing(df)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLflow Experiment Setup
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run():
        print("--- Training Random Forest Model ---")
        
        # Hyperparameters
        n_estimators = 100
        max_depth = 10
        
        # Model Initialization
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted') # Weighted for potential class imbalance
        
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        # --- MLflow Logging ---
        # Log Parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log Artifacts (EDA Plots)
        mlflow.log_artifacts("plots")
        
        # Log Model
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        # --- Save Model Locally (Pickle) ---
        print(f"Saving model to {MODEL_FILE}...")
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(rf, f)
            
        print("Training run complete.")

if __name__ == "__main__":
    train_model()