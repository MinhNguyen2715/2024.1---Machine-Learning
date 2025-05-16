#Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#@title Import data
rice = pd.read_excel('../data/rice.xlsx')
X = rice.drop("Class", axis=1)
y = rice["Class"].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Decision Tree for rice classification
#Find the best parameters for Decision Tree

rice_clf = DecisionTreeClassifier()

param_grid = {
    'max_depth': ['None', 1, 2, 3, 4, 5, 6],
    'min_samples_split': ['None', 1, 2, 3],
    'min_samples_leaf': ['None', 1, 2, 3],
}

grid_search_dt = GridSearchCV(rice_clf, param_grid, cv=10)
grid_search_dt.fit(X_train, y_train)

print("Best parameters: ", grid_search_dt.best_params_)

#Decision Tree

rice_clf = DecisionTreeClassifier(
    max_depth = grid_search_dt.best_params_['max_depth'],
    min_samples_leaf = grid_search_dt.best_params_['min_samples_leaf'],
    min_samples_split = grid_search_dt.best_params_['min_samples_split']
    )

rice_clf.fit(X_train, y_train)
y_pred = rice_clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cammeo', 'Osmancik'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# Compute all metrics
accuracy  = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
npv = tn / (tn + fn)
fpr = fp / (fp + tn)
fdr = fp / (tp + fp)
fnr = fn / (fn + tp)

print(f"Accuracy: {accuracy * 100:.4f}")
print(f"Sensitivity: {sensitivity*100:.4f}")
print(f"Specificity: {specificity*100:.4f}")
print(f"Precision: {precision*100:.4f}")
print(f"F1-Score: {f1_score*100:.4f}")
print(f"NPV: {npv*100:.4f}")
print(f"FPR: {fpr*100:.4f}")
print(f"FDR: {fdr*100:.4f}")
print(f"FNR: {fnr*100:.4f}")