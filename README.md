# Dry Bean Classification — Model Benchmark Arena

**Machine Learning — Assignment 2**  
**Programme:** M.Tech (Artificial Intelligence and Machine Learning), BITS Pilani WILP

##  Problem Statement

The objective of this project is to classify dry beans into seven varieties using 16 numerical morphological features derived from bean images.

This supervised multiclass classification problem compares five machine learning algorithms on a common dataset to determine which provides the best predictive performance.

The project also includes an interactive Streamlit application, **Model Benchmark Arena**, for evaluating saved classification models, analyzing their performance and generating predictions for uploaded datasets without retraining.

##  Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/602/dry+bean

The original dataset contains **13,611 observations**, 16 numerical predictors and one categorical target variable.

**Input Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3 and ShapeFactor4.

**Target Variable:** `Class`

**Classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA.

**Data Preparation:**

- Exact duplicate records were removed before splitting the dataset.
- An 80:20 stratified train-test split was performed using `random_state=42`.
- Median imputation was incorporated into the preprocessing pipelines.
- Standard scaling was applied to models requiring feature normalization.
- Five-fold stratified cross-validation was performed on the training data.
- All five models were evaluated on the same untouched held-out test dataset.

##  GitHub Repository Link

**GitHub Repository:** https://github.com/mk-aakil/2026AJ05523_ML_Assignment_2

**Live Streamlit Application:** https://2026aj05523-ml-assignment-2.streamlit.app/

The repository includes source code, Jupyter notebooks, saved model pipelines, dependency requirements, test data, evaluation reports and the Streamlit application.

##  Models Used

Five classification models were implemented:

1. **Logistic Regression:** A linear classification model trained using standardized features.
2. **Decision Tree:** A tree-based classifier that learns hierarchical decision rules.
3. **K-Nearest Neighbors (KNN):** A distance-based classifier using seven neighbors and distance-weighted voting.
4. **Naive Bayes:** A Gaussian probabilistic classifier.
5. **Support Vector Machine (SVM):** A calibrated linear SVM supporting multiclass probability estimates.

### Model Performance Comparison

All metrics below were obtained from the same held-out test dataset.

Precision, Recall and F1 are macro-averaged across the seven bean classes. AUC represents macro-averaged one-vs-rest multiclass ROC-AUC. MCC is the multiclass Matthews Correlation Coefficient.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.9192** | **0.9947** | **0.9307** | **0.9300** | **0.9302** | **0.9023** |
| Decision Tree | 0.9040 | 0.9700 | 0.9179 | 0.9179 | 0.9178 | 0.8840 |
| KNN | 0.9136 | 0.9854 | 0.9268 | 0.9237 | 0.9250 | 0.8956 |
| Naive Bayes | 0.7630 | 0.9670 | 0.7736 | 0.7685 | 0.7669 | 0.7143 |
| Support Vector Machine | 0.9147 | 0.9943 | 0.9298 | 0.9264 | 0.9277 | 0.8971 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieved the highest performance across all six evaluation metrics, with 91.92% accuracy, 0.9302 macro F1 and 0.9023 MCC. Its high AUC of 0.9947 indicates excellent discrimination between bean classes. With a prediction time of approximately 13.37 ms, it provides the strongest overall balance of predictive quality and inference efficiency. |
| **Decision Tree** | Achieved 90.40% accuracy and 0.9178 macro F1, demonstrating strong classification performance. It was the fastest model, with a prediction time of approximately 4.83 ms. However, its AUC of 0.9700 and MCC of 0.8840 were below those of Logistic Regression, SVM and KNN. It is a useful option when inference speed and interpretability are priorities. |
| **KNN** | Achieved 91.36% accuracy, 0.9250 macro F1 and 0.8956 MCC, placing it third in overall predictive performance. Its AUC of 0.9854 demonstrates strong class discrimination. However, its prediction time of approximately 250.66 ms was substantially higher than the other models, making inference efficiency its main disadvantage in this comparison. |
| **Naive Bayes** | Recorded the lowest accuracy (76.30%), macro F1 (0.7669) and MCC (0.7143). Although its AUC of 0.9670 suggests useful class-ranking capability, its classification predictions were less accurate than those of the other four models. Correlations between morphological features may contribute to this limitation because Gaussian Naive Bayes assumes conditional feature independence. |
| **Support Vector Machine** | Achieved the second-highest accuracy (91.47%), macro F1 (0.9277) and MCC (0.8971), with an excellent AUC of 0.9943. Its prediction time was approximately 14.70 ms, close to Logistic Regression. It is a strong alternative, but Logistic Regression performed slightly better across all six evaluation metrics. |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the overall winner. It achieved the highest Accuracy (0.9192), AUC (0.9947), Precision (0.9307), Recall (0.9300), F1 (0.9302) and MCC (0.9023). It also maintained efficient inference at approximately 13.37 ms. Therefore, it provides the best overall combination of classification quality and prediction efficiency for this dataset and evaluation setup. |

### Prediction Time Comparison

Prediction times were measured on the held-out test dataset using the saved model pipelines.

| Model | Prediction Time (ms) |
|---|---:|
| Logistic Regression | 13.3697 |
| Decision Tree | **4.8289** |
| KNN | 250.6566 |
| Naive Bayes | 11.5966 |
| Support Vector Machine | 14.6987 |

Decision Tree provided the fastest inference, while KNN required the longest prediction time. Logistic Regression offered substantially better predictive performance than Decision Tree while remaining relatively fast.

These timings represent measurements from the evaluation environment and may vary with hardware, system load and input size.

### Conclusion

All five classification models were successfully trained and evaluated using a consistent preprocessing and evaluation methodology.

Logistic Regression achieved the strongest overall classification results, followed closely by SVM and KNN. Decision Tree delivered the fastest inference, while Naive Bayes showed comparatively weaker classification performance.

**Final Selected Model: Logistic Regression**

The Streamlit Model Benchmark Arena enables interactive comparison of all five saved classifiers, supports labeled and unlabeled CSV uploads, displays evaluation metrics and diagnostic visualizations, and provides downloadable predictions without retraining the models.