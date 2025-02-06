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

## 🗂️ Project Structure  
