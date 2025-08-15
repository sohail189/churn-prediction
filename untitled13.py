import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

a=pd.read_csv('/content/p1.csv')

a.head(5)

a.isnull().sum()

a.tail(5)

a.shape

a.dtypes

a.info()

a.describe()

a.describe(include='object')

a[['MonthlyCharges','TotalCharges']]

a.iloc[3]

a.loc[1]

a.loc[3,'tenure']

a[a['tenure']>5]

a.set_index('tenure')

a.drop_duplicates()

a.reset_index()

a.select_dtypes(include='object').columns

for col in a.select_dtypes(include='object').columns:
    print(f"{col} → {a[col].unique()}")

import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

a = pd.DataFrame({'gender': ['Male','Female']})
le = LabelEncoder()
a['gender_new'] = le.fit_transform(a['gender'])
print(a)

a= pd.DataFrame({'Partner':['Yes','No']})
le=LabelEncoder()
a['Partner_new']= le.fit_transform(a['Partner'])
print(a)

a = pd.DataFrame({'Dependents':['Yes','No']})
le=LabelEncoder()
a['Dependents_new']= le.fit_transform(a['Dependents'])
print(a)

a=pd.DataFrame({'PhoneService':['Yes','No']})
le=LabelEncoder()
a['PhoneService_new']=le.fit_transform(a['PhoneService'])
print(a)

a=pd.DataFrame({'OnlineSecurity':['Yes','No']})
le=LabelEncoder()
a['OnlineSecurity_new']=le.fit_transform(a['OnlineSecurity'])
print(a)

a=pd.DataFrame({'TechSupport':['Yes','No']})
le=LabelEncoder()
a['TechSupport_new']=le.fit_transform(a['TechSupport'])
print(a)
a=pd.DataFrame({'DeviceProtection':['Yes','No']})
le=LabelEncoder()
a['DeviceProtection_new']=le.fit_transform(a['DeviceProtection'])
print(a)

a=pd.DataFrame({'InternetService':['Fiber optic','DSL']})
le=LabelEncoder()
a['InternetService_new']=le.fit_transform(a['InternetService'])
print(a)
a=pd.DataFrame({'MultipleLines':['No phone service','Yes','No']})
le=LabelEncoder()
a['MultipleLines_new']=le.fit_transform(a['MultipleLines'])
print(a)

a=pd.DataFrame({'StreamingTV':['Yes','No']})
le=LabelEncoder()
a['StreamingTV_new']=le.fit_transform(a['StreamingTV'])
print(a)
a=pd.DataFrame({'StreamingMovies':['Yes','No']})
le=LabelEncoder()
a['StreamingMovies_new']=le.fit_transform(a['StreamingMovies'])
print(a)

a=pd.DataFrame({'Contract':['Month-to-month','One year','Two year']})
le=LabelEncoder()
a['Contract_new']=le.fit_transform(a['Contract'])
print(a)
a=pd.DataFrame({'PaperlessBilling':['Yes','No']})
le=LabelEncoder()
a['PaperlessBilling_new']=le.fit_transform(a['PaperlessBilling'])
print(a)
a=pd.DataFrame({'PaymentMethod':['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)']})
le=LabelEncoder()
a['PaymentMethod_new']=le.fit_transform(a['PaymentMethod'])
print(a)
a=pd.DataFrame({'Churn':['Yes','No']})
le=LabelEncoder()
a['Churn_new']=le.fit_transform(a['Churn'])
print(a)

a = pd.read_csv('/content/p1.csv')
a.drop(['gender'], axis=1)

a.columns = a.columns.str.strip().str.lower().str.replace(' ', '_')
cat_cols = [ 'partner', 'dependents', 'phoneservice', 'multiplelines',
            'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection',
            'techsupport', 'streamingtv', 'streamingmovies', 'contract',
            'paperlessbilling', 'paymentmethod', 'churn']
for col in cat_cols:
    le = LabelEncoder()
    a[col + '_new'] = le.fit_transform(a[col])

a=pd.read_csv('/content/p1.csv')
a.drop(['customerID'], axis=1,inplace=True)
a.columns = a.columns.str.strip().str.lower().str.replace(' ', '_')
cat_cols = ['gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
            'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection',
            'techsupport', 'streamingtv', 'streamingmovies', 'contract',
            'paperlessbilling', 'paymentmethod', 'churn']
for col in cat_cols:
    le = LabelEncoder()
    a[col + '_new'] = le.fit_transform(a[col])
display(a.head())

list(a.columns)

encoded_cols = [col + '_new' for col in cat_cols]
remaining_cols = [col for col in a.columns if col not in cat_cols and col not in encoded_cols]
final_df = a[remaining_cols + encoded_cols]
(final_df.head())

a.head(1)

sns.countplot(x='gender',data=a)
plt.show()

a['totalcharges'] = pd.to_numeric(a['totalcharges'], errors='coerce')
a.dropna(subset=['totalcharges'], inplace=True)
sns.histplot(x='totalcharges',data=a)
plt.show()

sns.boxplot(x='paperlessbilling',y='monthlycharges',data=a)
plt.show()

sns.scatterplot(x='totalcharges',y='monthlycharges',data=a)
plt.show()

sns.countplot(x='internetservice',data=a)
plt.show()

def tenure_group(tenure):
    if tenure <= 12:
        return '0-1 year'
    elif tenure <= 24:
        return '1-2 years'
    elif tenure <= 48:
        return '2-4 years'
    elif tenure <= 60:
        return '4-5 years'
    else:
        return '5+ years'

a['tenure_group'] = a['tenure'].apply(tenure_group)

service_cols = ['phoneservice', 'internetservice', 'onlinesecurity',
                'onlinebackup', 'deviceprotection', 'techsupport',
                'streamingtv', 'streamingmovies']

a['num_services'] = a[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

def secure_internet(row):
    if row['internetservice'] != 'No' and row['onlinesecurity'] == 'Yes':
        return 'Secure'
    elif row['internetservice'] != 'No':
        return 'Unsecure'
    else:
        return 'No Internet'

a['internet_security_status'] = a.apply(secure_internet, axis=1)

def monthly_category(charge):
    if charge < 35:
        return 'Low'
    elif charge < 70:
        return 'Medium'
    else:
        return 'High'

a['monthly_cost_group'] = a['monthlycharges'].apply(monthly_category)

a['gender'].head(7043)

from sklearn.model_selection import train_test_split
X = a.drop('churn', axis=1)
y = a['churn']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Select categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Apply one-hot encoding
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = ohe.fit_transform(X_train[categorical_cols])
X_test_encoded = ohe.transform(X_test[categorical_cols])

# Get feature names after one-hot encoding
encoded_cols = ohe.get_feature_names_out(categorical_cols)

# Create new dataframes with encoded columns
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

# Drop original categorical columns and concatenate with encoded columns
X_train_processed = X_train.drop(columns=categorical_cols).join(X_train_encoded_df)
X_test_processed = X_test.drop(columns=categorical_cols).join(X_test_encoded_df)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_processed,y_train)
y_pred=lr.predict(X_test_processed)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')

from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier model
rfc = RandomForestClassifier(random_state=42)

# Train the model
rfc.fit(X_train_processed, y_train)

# Make predictions on the test set
y_pred_rfc = rfc.predict(X_test_processed)

# Evaluate the model
print("Random Forest Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rfc)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rfc)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_rfc)}")

a['churn'].value_counts(normalize=True)

sns.countplot(x='churn', data=a)
plt.title("Churn Distribution")
plt.show()

a.groupby('churn').mean(numeric_only=True)

pip install imbalanced-learn

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
lr.fit(X_train_resampled, y_train_resampled)
y_pred=lr.predict(X_test_processed)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

y_proba = lr.predict_proba(X_test_processed)[:, 1]
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, pos_label='Yes'),
    "Recall": recall_score(y_test, y_pred, pos_label='Yes'),
    "F1 Score": f1_score(y_test, y_pred, pos_label='Yes'),
    "ROC-AUC": roc_auc_score(y_test, y_proba)
}

plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.title("Model Performance Metrics")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Churn','Churn'], yticklabels=['No Churn','Churn'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)

PrecisionRecallDisplay.from_predictions(y_test_encoded, y_proba, pos_label=1)
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_proba, pos_label=1)
plt.title("ROC Curve")
plt.grid(True)
plt.show()

# Example: check correlation
a.corr(numeric_only=True)['churn_new'].sort_values(ascending=False)

import gc
gc.collect()

import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

X = a.drop('churn', axis=1)
y = a['churn']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y_encoded,test_size=0.2,random_state=42)

# Select categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Apply one-hot encoding
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = ohe.fit_transform(X_train[categorical_cols])
X_test_encoded = ohe.transform(X_test[categorical_cols])

# Get feature names after one-hot encoding
encoded_cols = ohe.get_feature_names_out(categorical_cols)

# Create new dataframes with encoded columns
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

# Drop original categorical columns and concatenate with encoded columns
X_train_processed = X_train.drop(columns=categorical_cols).join(X_train_encoded_df)
X_test_processed = X_test.drop(columns=categorical_cols).join(X_test_encoded_df)

# Create an XGBoost classifier model
xgb_model = xgb.XGBClassifier(random_state=42)

# Train the model
xgb_model.fit(X_train_processed, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test_processed)

# Evaluate the model
print("XGBoost Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_xgb)}")

from xgboost import XGBClassifier
XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=100)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Split data for early stopping
X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.05,
    n_estimators=100,
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=10,  # stop if no improvement for 10 rounds
    random_state=42
)

model.fit(
    X_train_part, y_train_part,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Make predictions on the test set
y_pred_xgb_tuned = model.predict(X_test_processed)

# Evaluate the model
print("Tuned XGBoost Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb_tuned)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_xgb_tuned)}")



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Split data for early stopping
X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.05,
    n_estimators=100,
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=10,  # stop if no improvement for 10 rounds
    random_state=42
)


model.fit(
    X_train_part, y_train_part,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Make predictions on the test set
y_pred_xgb_tuned = model.predict(X_test_processed)

# Evaluate the model
print("Tuned XGBoost Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb_tuned)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_xgb_tuned)}")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in cv.split(X_train_processed, y_train):
    X_train_fold, X_val_fold = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )

    y_pred_fold = model.predict(X_val_fold)
    scores.append(accuracy_score(y_val_fold, y_pred_fold))

print(f'Cross-validated accuracy: {np.mean(scores)}')

(a.duplicated().sum())

a = a.drop_duplicates().dropna()

(a.duplicated().sum())

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in cv.split(X_train_processed, y_train):
    X_train_fold, X_val_fold = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )

    y_pred_fold = model.predict(X_val_fold)
    scores.append(accuracy_score(y_val_fold, y_pred_fold))

print(f'Cross-validated accuracy: {np.mean(scores)}')

a.shape

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

X = a.drop(['churn', 'churn_new'], axis=1)
y = a['churn_new']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Select categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Apply one-hot encoding
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = ohe.fit_transform(X_train[categorical_cols])
X_test_encoded = ohe.transform(X_test[categorical_cols])

# Get feature names after one-hot encoding
encoded_cols = ohe.get_feature_names_out(categorical_cols)

# Create new dataframes with encoded columns
X_train_processed = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
X_test_processed = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

# Drop original categorical columns and concatenate with encoded columns
X_train_processed = X_train.drop(columns=categorical_cols).join(X_train_processed)
X_test_processed = X_test.drop(columns=categorical_cols).join(X_test_processed)

model = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.05,
    n_estimators=100,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train_processed, y_train)
y_pred = model.predict(X_test_processed)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Churn','Churn'], yticklabels=['No Churn','Churn'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import LabelEncoder

y_proba = model.predict_proba(X_test_processed)[:, 1]

le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)

PrecisionRecallDisplay.from_predictions(y_test_encoded, y_proba, pos_label=1)
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()

from sklearn.model_selection import GridSearchCV

a.columns

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

# Select categorical columns from the original training data
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Apply one-hot encoding to the training data
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = ohe.fit_transform(X_train[categorical_cols])

# Create a new DataFrame with the encoded columns
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=ohe.get_feature_names_out(categorical_cols), index=X_train.index)

# Drop the original categorical columns and concatenate with the encoded columns
X_train_processed = X_train.drop(columns=categorical_cols).join(X_train_encoded_df)


grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',      # or 'roc_auc' for classification problems
    cv=5,
    n_jobs=-1,
    verbose=2,
    refit=True
)

grid_search.fit(X_train_processed, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# Apply one-hot encoding to the test data using the same fitted encoder
X_test_encoded = ohe.transform(X_test[categorical_cols])
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=ohe.get_feature_names_out(categorical_cols), index=X_test.index)
X_test_processed = X_test.drop(columns=categorical_cols).join(X_test_encoded_df)

y_pred = best_model.predict(X_test_processed)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test_processed)[:, 1]))

"""Summary: Toolchain for Complex Chatbot
Stage	Recommended Tools/Frameworks
Model training:	scikit-learn, XGBoost, TensorFlow, PyTorch
API backend:	FastAPI, Flask
Conversational AI:	Rasa, Dialogflow, LangChain
Frontend UI:	React, Streamlit, Bot Framework
Deployment:	Docker, AWS, GCP"""

