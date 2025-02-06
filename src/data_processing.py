import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Handle missing values using forcleward fill
    df = df.ffill()
    
    # Normalize text columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower().str.strip()
    
    # Check if 'treatment' column exists
    if 'treatment' not in df.columns:
        raise ValueError("The 'treatment' column is missing in the dataset.")
    
    # Encode 'treatment' column
    label_encoder = LabelEncoder()
    df['treatment'] = label_encoder.fit_transform(df['treatment'])
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    return df

def feature_selection(df):
    features = df.drop(columns=['treatment'])  # 'treatment' is the target
    labels = df['treatment']
    return features, labels

if __name__ == "__main__":
    try:
        df = load_data('dataset.csv')
        df = preprocess_data(df)
        features, labels = feature_selection(df)
        features.to_csv('features.csv', index=False)
        labels.to_csv('labels.csv', index=False)
        print("Data processing complete.")
    except Exception as e:
        print(f"Error: {e}")
print(df.columns)