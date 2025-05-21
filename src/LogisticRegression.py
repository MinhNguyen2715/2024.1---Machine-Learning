import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Excel data
df = pd.read_excel('rice.xlsx', sheet_name="Sheet1")
le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])  # Cammeo -> 0 (or 1)

# 2. Define feature matrix X and target vector y
X = df.drop("Class", axis=1)  # Replace 'label' with your actual target column
y = df["Class"]

# 3. Define a pipeline standard scaler and logistic regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(solver="liblinear"))
])

# 4. Define hyperparameter grid
param_grid = [{
    "classifier__penalty": ["l1"],
    "classifier__C": [7.5, 7.7, 7.9, 8, 8.1, 8.3, 8.5],
    "classifier__class_weight": [None, "balanced"],
    "classifier__fit_intercept": [True, False],
    "classifier__max_iter": [50, 100, 500, 1000],
    "classifier__tol": [1e-5, 1e-4, 1e-3, 1e-2],
    "classifier__dual": [False],
}, {
    "classifier__penalty": ["l2"],
    "classifier__C": [7.5, 7.7, 7.9, 8, 8.1, 8.3, 8.5],
    "classifier__class_weight": [None, "balanced"],
    "classifier__fit_intercept": [True, False],
    "classifier__max_iter": [50, 100, 500, 1000],
    "classifier__tol": [1e-5, 1e-4, 1e-3, 1e-2],
    "classifier__dual": [False, True],
}, ]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 6. Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",        # Change to 'accuracy' if preferred
    # cv=10,
    cv=skf,
    n_jobs=-1,
    verbose=2
)

# 7. Fit the model
grid_search.fit(X_train, y_train)

# 8. Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# 9. Final evaluation on test set
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

# === Confusion matrix ===
cm = confusion_matrix(y_test, y_pred);
tn, fp, fn, tp = cm.ravel()

# === Metrics ===
accuracy    = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0    # Recall
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
precision   = tp / (tp + fp) if (tp + fp) != 0 else 0
f1_score    = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
npv         = tn / (tn + fn) if (tn + fn) != 0 else 0    # Negative Predictive Value
fpr         = fp / (fp + tn) if (fp + tn) != 0 else 0
fdr         = fp / (tp + fp) if (tp + fp) != 0 else 0
fnr         = fn / (fn + tp) if (fn + tp) != 0 else 0

# === Print ===
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
print(f"Accuracy     = {accuracy:.4f}")
print(f"Sensitivity  = {sensitivity:.4f}")
print(f"Specificity  = {specificity:.4f}")
print(f"Precision    = {precision:.4f}")
print(f"F1 Score     = {f1_score:.4f}")
print(f"NPV          = {npv:.4f}")
print(f"FPR          = {fpr:.4f}")
print(f"FDR          = {fdr:.4f}")
print(f"FNR          = {fnr:.4f}")

# === Draw Confusion Matrix ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()