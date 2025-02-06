import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("dataset.csv")

# Specify the actual feature columns and target column
feature_columns = ['Age', 'Gender', 'Country', 'self_employed', 'family_history', 'work_interfere', 'no_employees',
                  'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help',
                  'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 'coworkers',
                  'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical',
                  'obs_consequence']
target_column = 'treatment'

# Ensure all specified columns exist in the DataFrame
missing_columns = set(feature_columns) - set(df.columns)
if missing_columns:
    raise KeyError(f"The following columns are missing in the dataset: {missing_columns}")

# Prepare the data
X = df[feature_columns]
y = df[target_column]

# Preprocessing pipeline: scale numeric columns and encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age']),  # Scale 'Age'
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Country', 'self_employed', 'family_history',
                                                         'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                                                         'benefits', 'care_options', 'wellness_program', 'seek_help',
                                                         'anonymity', 'leave', 'mental_health_consequence',
                                                         'phys_health_consequence', 'coworkers', 'supervisor',
                                                         'mental_health_interview', 'phys_health_interview',
                                                         'mental_vs_physical', 'obs_consequence'])
    ]
)

# Create a pipeline with preprocessing and classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Save the entire pipeline
joblib.dump(pipeline, 'mental_health_pipeline.pkl')
print("Pipeline saved as 'mental_health_pipeline.pkl'.")
