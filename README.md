# Rice Variety Classification Using Machine Learning

This project focuses on the binary classification of rice grain varieties — **Osmancik** and **Cammeo** — using various machine learning models. It was developed as part of the ***IT3190E - Machine Learning*** course, semester **2024.2** at the **Hanoi University of Science and Technology (HUST)**.

## 📁 Project Overview

- **Goal**: Classify rice varieties based on their morphological characteristics.
- **Techniques Used**: Logistic Regression, Support Vector Machine, Decision Tree, Random Forest, Naive Bayes, and k-Nearest Neighbor.
- **Tools**: Python, Scikit-learn, NumPy, Pandas, Matplotlib.
- **Data**: 3,810 rice grains labeled by type, with 7 extracted features from grain images.

## 📊 Dataset Description

| Feature             | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| Area                | Number of pixels inside the grain's boundary                                |
| Perimeter           | Distance around the grain                                                   |
| MajorAxisLength     | Longest diameter of the ellipse enclosing the grain                         |
| MinorAxisLength     | Shortest diameter of the ellipse enclosing the grain                        |
| Eccentricity        | Roundness of the grain's elliptical shape                                   |
| ConvexArea          | Number of pixels in the convex hull around the grain                        |
| Extent              | Ratio of region area to bounding box area                                   |

Grain counts:

- Osmancik: 2,180 samples
- Cammeo: 1,630 samples

## ⚙️ Methodology

### Preprocessing
- Encoded categorical labels (`Osmancik` → 1, `Cammeo` → 0)
- Standardized features using `StandardScaler`
- Stratified train-test split with 80:20 ratio

### Classification Algorithms
- **Logistic Regression (LR)** with hyperparameter tuning via `GridSearchCV`
- **Support Vector Machine (SVM)** with kernel selection and soft margin optimization
- **Decision Tree (DT)** using ID3 and CART algorithms
- **Random Forest (RF)** for ensemble-based classification
- **Naive Bayes (NB)** with GaussianNB
- **k-Nearest Neighbor (k-NN)** with randomized hyperparameter search

### Evaluation Metrics
- Accuracy
- Sensitivity (Recall)
- Specificity
- Precision
- F1-Score
- Negative Predictive Value
- False Positive Rate
- False Discovery Rate
- False Negative Rate

## 📈 Performance Comparison of Models

| **Measure**                | **LR**  | **SVM** | **DT**  | **RF**  | **NB**  | **k-NN** |
|----------------------------|---------|---------|---------|---------|---------|----------|
| **Accuracy**               | 91.73%  | 91.73%  | 92.51%  | 92.12%  | 87.01%  | **92.91%** |
| **Sensitivity**            | **94.50%**  | 92.43%  | **92.96%**  | 92.71%  | 92.66%  | 93.66% |
| **Specificity**            | 88.04%  | 90.80%  | 92.00%  | 91.42%  | 79.45%  | **91.96%** |
| **Precision**              | 91.35%  | 93.07%  | 93.18%  | 92.71%  | 85.77%  | **93.66%** |
| **F1-Score**               | 92.90%  | 92.75%  | 93.07%  | 92.71%  | 89.08%  | **93.66%** |
| **Negative Predictive Value** | **92.28%**  | 89.97%  | 91.73%  | 91.42%  | 89.00%  | **91.96%** |
| **False Positive Rate**    | 11.96%  | 9.20%   | **8.00%**   | 8.57%   | 20.55%  | **8.04%**  |
| **False Discovery Rate**   | 8.65%   | 6.93%   | **6.81%**   | 7.28%   | 7.34%   | **6.34%**  |
| **False Negative Rate**    | **5.50%**   | 7.57%   | 7.03%   | 7.28%   | 14.23%  | 6.34%   |


**Conclusion**: The **k-NN model** yielded the highest overall performance and was the most effective classifier for this dataset.
