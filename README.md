Churn Prediction

Churn Prediction is a machine learning project that forecasts which customers are likely to stop using a service.
By analyzing historical customer data, the model identifies at-risk customers so businesses can take proactive steps to retain them.

This project includes:

Data preprocessing (handling missing values, encoding categorical features)

Exploratory data analysis (visualizing churn patterns)

Training and evaluating multiple ML classifiers (Logistic Regression, Random Forest, XGBoost, etc.)

Class balancing with SMOTE

Detailed performance metrics and visual charts

Objective

Predict customer churn in advance so companies can reduce losses and improve customer retention.

Key Features

Automatic data cleaning and preprocessing

Categorical and numerical feature encoding

Model training & comparison across multiple algorithms

Evaluation with precision, recall, F1-score, and ROC-AUC

Easy-to-run script and reproducible results

Project Structure
Churn-Prediction/
│-- data/               # Datasets (e.g., p1.csv)
│-- notebooks/          # Jupyter notebooks for exploration
│-- src/                # Python scripts
│   └── churn_model.py  # Main training script
│-- requirements.txt    # Python dependencies
└-- README.md           # Project overview


This structure keeps data, code, and documentation neatly organized.

Tools & Technologies

Python 3.x – Core language

Pandas, NumPy – Data processing & manipulation

Matplotlib, Seaborn – Data visualization

Scikit-learn – ML utilities and model evaluation

XGBoost, LightGBM – High-performance boosting models

RandomForestClassifier – Ensemble model for baseline

imbalanced-learn (SMOTE) – Balancing imbalanced data

Jupyter Notebook – Interactive exploration

Git & GitHub – Version control

Setup & Installation

Clone the Repository

git clone https://github.com/<username>/Churn-Prediction.git
cd Churn-Prediction


(Optional) Create Virtual Environment

python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate


Install Dependencies

pip install -r requirements.txt


If requirements.txt is missing:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn


Add Dataset
Place p1.csv in the data/ folder.
Ensure columns like gender, Churn, MonthlyCharges, etc., match the script.

How to Run

Option 1 – Jupyter Notebook:

jupyter notebook


Open notebooks/churn_analysis.ipynb and run the cells.

Option 2 – Python Script:

python src/churn_model.py


The script will:

Load and preprocess data

Train Logistic Regression, Random Forest, and XGBoost models

Display evaluation metrics and plots

Example – XGBoost Training
from xgboost import XGBClassifier

model = XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

Visualizations

Churn distribution plots

Monthly Charges & Total Charges comparison

ROC curves and accuracy scores

Feature importance rankings

License

This project is licensed under the MIT License – free to use, modify, and share.

Author

Muhammad Sohail
Machine Learning Enthusiast & Data Analyst
Open to collaboration and contributions.
