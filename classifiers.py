# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import json

# Step 1: Load the data from an external JSON file
def load_data(file_path):
    """
    Load customer data from a JSON file into a Pandas DataFrame.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

# Step 2: Ensure the target column exists and assign a default category if missing
def ensure_target_column(df, target_column, default_value="Unknown"):
    """
    Ensure the target column exists in the DataFrame. If it does not exist,
    create it and assign a default value to all rows.
    
    Parameters:
        - df: The input DataFrame.
        - target_column: The name of the target column.
        - default_value: The default value to assign if the column is missing.
    
    Returns:
        - Updated DataFrame with the target column ensured.
    """
    if target_column not in df.columns:
        print(f"Warning: Target column '{target_column}' not found. Creating it with default value '{default_value}'.")
        df[target_column] = default_value
    else:
        # Fill any missing values in the existing target column with the default value
        df[target_column] = df[target_column].fillna(default_value)
    
    return df

# Step 3: Preprocess the data
def preprocess_data(df):
    """
    Preprocess the data by encoding categorical variables and selecting features/target.
    """
    # Encode categorical variables (e.g., Gender)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    # Select features and target variable
    features = ['Age', 'Frequency (for visits)', 'Coffee', 'Cold drinks', 
                'Jaws chip', 'Pastries', 'Juices', 'Sandwiches', 'cake']
    target = 'Recommended Product'
    
    X = df[features]
    y = df[target]
    
    return X, y

# Step 4: Train a classification model
def train_model(X_train, y_train, model_type='random_forest'):
    """
    Train a classification model (Random Forest or Decision Tree).
    
    Parameters:
        - X_train: Training features
        - y_train: Training labels
        - model_type: Type of model ('random_forest' or 'decision_tree')
    
    Returns:
        - Trained model
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose 'random_forest' or 'decision_tree'.")
    
    model.fit(X_train, y_train)
    return model

# Step 5: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using accuracy and classification report.
    """
    predictions = model.predict(X_test)
    
    print("Model Evaluation:")
    print(f"Accuracy Score: {accuracy_score(y_test, predictions):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    return predictions

# Step 6: Save predictions to a CSV file
def save_predictions(X_test, y_test, predictions, output_file):
    """
    Save test data with actual and predicted labels to a CSV file.
    """
    results = X_test.copy()
    results['Actual'] = y_test.values
    results['Predicted'] = predictions
    
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Main function to execute the workflow
def main():
    # Load data from external JSON file
    file_path = 'starbucks.json'  # Replace with your actual JSON file path
    df = load_data(file_path)

    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Ensure the target column exists and assign default category if missing
    target_column = 'Recommended Product'
    df = ensure_target_column(df, target_column)

    # Preprocess the data
    X, y = preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate Decision Tree classifier
    print("\nTraining Decision Tree Classifier...")
    decision_tree_model = train_model(X_train, y_train, model_type='decision_tree')
    dt_predictions = evaluate_model(decision_tree_model, X_test, y_test)
    
    # Save Decision Tree predictions to a CSV file
    save_predictions(X_test, y_test, dt_predictions, 'decision_tree_predictions.csv')

    # Train and evaluate Random Forest classifier
    print("\nTraining Random Forest Classifier...")
    random_forest_model = train_model(X_train, y_train, model_type='random_forest')
    rf_predictions = evaluate_model(random_forest_model, X_test, y_test)
    
    # Save Random Forest predictions to a CSV file
    save_predictions(X_test, y_test, rf_predictions, 'random_forest_predictions.csv')

if __name__ == "__main__":
    main()
