import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


df = pd.read_excel("data/rice.xlsx")
start = time.time()
X = df.drop(columns=['Class'])
y = df['Class']
print(df.head(5))
le = LabelEncoder()
y_encoded = le.fit_transform(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=20)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)

parameters = {
    'n_neighbors': [13, 14, 15, 16, 17, 18],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [20, 30, 40],
    'metric': ['minkowski']
}
# GridSearchCV
# grid = GridSearchCV(KNeighborsClassifier(n_jobs=-1), parameters, cv=skf, scoring='accuracy', verbose=1)
# grid.fit(X_train, y_train)
# grid.fit(X_train, y_train)
# best_model = grid.best_estimator_
# print("✅ Best Parameters:", grid.best_params_)
# print("✅ Best Cross-Validation Accuracy:", grid.best_score_)



# RandomizedSearchCV
knn= KNeighborsClassifier()
random_search = RandomizedSearchCV(knn, param_distributions=parameters, n_iter=10000,
                                   cv=skf, scoring='accuracy', verbose=1, random_state=10)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
print("✅ Best Parameters:", random_search.best_params_)
print("✅ Best Cross-Validation Accuracy:", random_search.best_score_)

#Đánh giá trên tập test
best_knn = random_search.best_estimator_
y_pred = best_knn.predict(X_test)

print("\n🎯 Test Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
end = time.time()
print("🕒 Done in:", round(end - start, 2), "s")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Tính toán các chỉ số
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)           
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
npv = tn / (tn + fn)
fpr = fp / (fp + tn)
fdr = fp / (tp + fp)
fnr = fn / (fn + tp)

print(f"\n Accuracy     : {accuracy:.4f}")
print(f" Sensitivity  : {sensitivity:.4f}")
print(f" Specificity  : {specificity:.4f}")
print(f" Precision    : {precision:.4f}")
print(f" F1 Score     : {f1:.4f}")
print(f" NPV          : {npv:.4f}")
print(f" FPR          : {fpr:.4f}")
print(f" FDR          : {fdr:.4f}")
print(f" FNR          : {fnr:.4f}")


#Biểu đồ accuracy theo k 
accuracies = []
k_range = range(1,20)
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    accuracies.append(acc)

plt.figure(figsize=(8, 4))
plt.plot(k_range, accuracies, marker='o')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy on Test Set")
plt.title("KNN Accuracy for Different k Values")
plt.grid(True)
plt.tight_layout()
plt.show()
