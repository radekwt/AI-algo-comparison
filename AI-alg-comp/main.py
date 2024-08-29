import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import data
import time
def calculate_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)

def visualize_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

X_train,y_train,X_test,y_test = data.cardio()

# Initialize classifiers
lr = LogisticRegression(max_iter=10000)
start_time = time.time()
lr.fit(X_train, y_train)
end_time = time.time()
lr_time = end_time - start_time

knn = KNeighborsClassifier()
start_time = time.time()
knn.fit(X_train, y_train)
end_time = time.time()
knn_time = end_time - start_time

dt = DecisionTreeClassifier()
start_time = time.time()
dt.fit(X_train, y_train)
end_time = time.time()
dt_time = end_time - start_time

print("Logistic Regression Time:", lr_time)
print("K-Nearest Neighbors Time:", knn_time)
print("Decision Tree Time:", dt_time)



# Step 3: Make predictions on the testing set
predictions_LR = lr.predict(X_test)
predictions_KNN = knn.predict(X_test)
predictions_DT = dt.predict(X_test)

# Calculate accuracy
acc_LR = calculate_accuracy(predictions_LR, y_test)
acc_KNN = calculate_accuracy(predictions_KNN, y_test)
acc_DT = calculate_accuracy(predictions_DT, y_test)

print("Accuracy (Logistic Regression):", acc_LR)
print("Accuracy (K-Nearest Neighbors):", acc_KNN)
print("Accuracy (Decision Tree):", acc_DT)

# Step 4: Evaluate the models
cm_LR = confusion_matrix(y_test, predictions_LR)
print("Confusion Matrix (Logistic Regression):")
print(cm_LR)

cm_KNN = confusion_matrix(y_test, predictions_KNN)
print("Confusion Matrix (K-Nearest Neighbors):")
print(cm_KNN)

cm_DT = confusion_matrix(y_test, predictions_DT)
print("Confusion Matrix (Decision Tree):")
print(cm_DT)

# Visualizing Confusion Matrix
visualize_confusion_matrix(cm_LR, "Confusion Matrix (Logistic Regression)")
visualize_confusion_matrix(cm_KNN, "Confusion Matrix (K-Nearest Neighbors)")
visualize_confusion_matrix(cm_DT, "Confusion Matrix (Decision Tree)")

# Classification Report
report_LR = classification_report(y_test, predictions_LR, digits=2)
print("Classification Report (Logistic Regression):")
print(report_LR)

report_KNN = classification_report(y_test, predictions_KNN, digits=2)
print("Classification Report (K-Nearest Neighbors):")
print(report_KNN)

report_DT = classification_report(y_test, predictions_DT, digits=2)
print("Classification Report (Decision Tree):")
print(report_DT)

# ROC Curve
y_scores_LR = lr.predict_proba(X_test)[:, 1]
fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, y_scores_LR)
auc_LR = roc_auc_score(y_test, y_scores_LR)
plt.plot(fpr_LR, tpr_LR, label='Logistic Regression (AUC = {:.2f})'.format(auc_LR))

y_scores_KNN = knn.predict_proba(X_test)[:, 1]
fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(y_test, y_scores_KNN)
auc_KNN = roc_auc_score(y_test, y_scores_KNN)
plt.plot(fpr_KNN, tpr_KNN, label='K-Nearest Neighbors (AUC = {:.2f})'.format(auc_KNN))

y_scores_DT = dt.predict(X_test)
fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, y_scores_DT)
auc_DT = roc_auc_score(y_test, y_scores_DT)
plt.plot(fpr_DT, tpr_DT, label='Decision Tree (AUC = {:.2f})'.format(auc_DT))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
precision_LR, recall_LR, _ = precision_recall_curve(y_test, y_scores_LR)
plt.plot(recall_LR, precision_LR, label='Logistic Regression')

precision_KNN, recall_KNN, _ = precision_recall_curve(y_test, y_scores_KNN)
plt.plot(recall_KNN, precision_KNN, label='K-Nearest Neighbors')

precision_DT, recall_DT, _ = precision_recall_curve(y_test, y_scores_DT)
plt.plot(recall_DT, precision_DT, label='Decision Tree')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()