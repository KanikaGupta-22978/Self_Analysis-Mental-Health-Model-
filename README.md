# Self-Analysis Mental Health Predictor  
**Welcome to the Arogo AI project!**  
This project is part of the AI/ML Engineer Intern Assignment and focuses on developing a self-analysis mental health model that predicts possible mental health conditions based on user-provided symptoms. The model emphasizes **accuracy**, **interpretability (using SHAP)**, and **efficiency**, and is designed for seamless integration into chatbots or applications.

---

## 📑 Table of Contents  
- [Project Overview](#project-overview)  
- [Data Preparation](#data-preparation)  
- [Model Development](#model-development)  
- [Inference and UI](#inference-and-ui)  
- [Project Structure](#project-structure)  
- [Setup Instructions](#setup-instructions)  
- [Usage Instructions](#usage-instructions)  
- [Kaggle Dataset](#kaggle-dataset)  
- [Additional Notes](#additional-notes)  

---

## 📌 Project Overview  
The **Self-Analysis Mental Health Predictor** leverages machine learning to analyze user-reported symptoms and predict potential mental health conditions. Key components include:  

1. **Data Preparation**: Cleaning, preprocessing, feature engineering, and exploratory analysis.  
2. **Model Development**:  
   - Training & evaluating multiple models (e.g., Logistic Regression, Random Forest)  
   - Using metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC  
   - SHAP for model interpretability.  
3. **Inference & UI**:  
   - **Command-line script** (`predict_mental_health.py`) to generate predictions and SHAP explanations saved as interactive HTML.  
   - **Streamlit Web UI** (`mental_health_ui.py`) for interactive testing.  

---

## 📊 Data Preparation  
The dataset is curated from publicly available mental health datasets. For this project, we are using the dataset from Kaggle:  

**[Mental Health Symptoms Dataset](https://www.kaggle.com/)**  

### Data Preprocessing Includes:  
- Cleaning and normalization  
- Handling missing values  
- Exploratory Data Analysis (EDA)  
- Feature engineering and selection  

---

## 🔧 Model Development  
We trained multiple models for multi-class classification of mental health conditions. The evaluation metrics include:  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-score**  
- **ROC-AUC**  

The best performing model is selected (saved as `models/best_model.pkl`) and was interpreted using **SHAP** to provide interactive explanations.

---

## 🔮 Inference and UI  

### 🖥️ Command-Line Inference  
Use the `predict_mental_health.py` script to input symptoms via the terminal and obtain predictions along with a SHAP explanation saved as an HTML file.  

### 🌐 Web UI (Streamlit)  
A basic UI built with **Streamlit** (`mental_health_ui.py`) allows users to:  
- Input their symptoms  
- View the predicted mental health condition and associated probabilities  
- See an interactive **SHAP force plot** explanation embedded directly into the UI  

---

## 🗂️ Project Structure  
