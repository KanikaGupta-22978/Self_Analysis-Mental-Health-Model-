# Self-Analysis Mental Health Predictor  

Welcome to the **Self-Analysis Mental Health Predictor** project! This project is designed to predict possible mental health conditions based on user-provided symptoms. The model emphasizes **accuracy**, **efficiency**, and **interpretability** using SHAP (SHapley Additive exPlanations). It is built to be easily integrated into chatbots or standalone applications for personal mental health assessments.

---

## 📑 Table of Contents  
- [Project Overview](#project-overview)  
- [Data Preparation](#data-preparation)  
- [Model Development](#model-development)  
- [Inference and UI](#inference-and-ui)  
- [Project Structure](#project-structure)  
- [Setup Instructions](#setup-instructions)  
- [Usage Instructions](#usage-instructions)  
- [Additional Notes](#additional-notes)  

---

## 📌 Project Overview  
The **Self-Analysis Mental Health Predictor** leverages machine learning to analyze symptoms and predict potential mental health conditions.  

### Key Components:  
- **Data Preparation**: Cleaning, preprocessing, feature engineering, and exploratory analysis.  
- **Model Development**: Training multiple models and selecting the best-performing one based on evaluation metrics like accuracy, precision, recall, F1-score, and ROC-AUC.  
- **SHAP for Interpretability**: Providing explanations for model predictions using interactive SHAP plots.  
- **UI for Inference**:  
  - **Command-line tool** for terminal-based predictions and SHAP explanations.  
  - **Streamlit-based Web UI** for a more interactive experience.  

---

## 📊 Data Preparation  
The dataset used in this project is curated from publicly available mental health datasets on **Kaggle**.  

### Data Preprocessing Steps:  
- Data cleaning and normalization  
- Handling missing values  
- Exploratory Data Analysis (EDA)  
- Feature engineering and selection  

---

## 🔧 Model Development  
We trained and evaluated multiple models, including:  
- **Logistic Regression**  
- **Random Forest Classifier**  

### Evaluation Metrics:  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-score**  
- **ROC-AUC**  

The best model is saved as `models/best_model.pkl` and interpreted using **SHAP** for better understanding of predictions.

---

## 🔮 Inference and UI  

### 🖥️ Command-Line Inference  
The command-line tool (`predict_mental_health.py`) allows users to enter symptoms and receive a prediction along with a SHAP explanation saved as an HTML file.  

### 🌐 Web UI (Streamlit)  
The **Streamlit-based UI** (`mental_health_ui.py`) enables users to:  
- Input their symptoms  
- View the predicted mental health condition  
- See an interactive **SHAP force plot** explanation  

---

## 🗂️ Project Structure and Setup Instructions

```bash
📂 Self_Analysis_Mental_Health_Model
 ├── 📂 Basic UI               # UI and frontend scripts
 ├── 📂 Data                   # Dataset and related files
 ├── 📂 Inference Script       # Command-line prediction script
 ├── 📂 Models                 # Pre-trained models
 ├── 📂 Video Demonstration    # Video explanations
 ├── 📂 src                    # Data processing and model training scripts
 └── README.md                 # Project documentation

# Setup Instructions

1️⃣ Clone the Repository:  
    git clone https://github.com/KanikaGupta-22978/Self_Analysis-Mental-Health-Model.git  
    cd Self_Analysis-Mental-Health-Model  

2️⃣ Create a Virtual Environment:  
    python -m venv env  
    source env/bin/activate  # On Windows use `env\Scripts\activate`  

3️⃣ Install Dependencies:  
    pip install -r requirements.txt -
numpy==1.21.6
pandas==1.3.5
scikit-learn==1.0.2
shap==0.41.0
streamlit==1.10.0
joblib==1.1.0
matplotlib==3.5.1
seaborn==0.11.2


# 🚀 Running the Project  

## 📌 1. Model Training  
Run the training script to preprocess data, train models, evaluate them, and save the best model.  
This step also generates a **SHAP explanation** for a sample input.

### 🔧 Data Preparation  
By default, the project uses a dummy dataset defined in `src/data_preprocessing.py`.  
To use your own dataset:  
1. Place your CSV file in the project.  
2. Modify the `load_data` function in `src/data_preprocessing.py` to load from your CSV.

### 🛠️ Training the Model  
Run the following command to train the model:  

python src/train_model.py

📊 2. Inference via Command-Line
Use the inference script to enter symptoms and get a prediction along with a SHAP explanation (saved as shap_explanation.html):

bash
Copy
Edit
python src/predict_mental_health.py

🌐 3. Interactive UI (Streamlit)
Launch the basic UI to interactively input symptoms, view predictions, and see an embedded SHAP explanation:

bash
Copy
Edit
streamlit run mental_health_ui.py

📖 Usage Instructions
🖥️ Command-Line Interface
Run the inference script and follow the on-screen prompts.
The model outputs:
Predicted mental health condition
Prediction probabilities
The SHAP explanation is saved as an HTML file (shap_explanation.html).

🌍 Web UI (Streamlit)
Run the Streamlit UI using the command above.
Input your symptoms in the text area.
Click the "Predict" button.
View:
The predicted mental health condition
Associated probabilities
Interactive SHAP explanation displayed inline
✨ Tip: The Streamlit interface offers a more user-friendly experience for testing predictions and visualizing results in real time!




