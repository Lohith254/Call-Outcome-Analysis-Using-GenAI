import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
import pandas as pd

# Load model predictions
with open("output/mistral_call_outcome_results.json", "r") as f:
    predictions = json.load(f)

# Load ground truth labels
with open("output/manual_label_call_outcome.json", "r") as f:
    ground_truth = json.load(f)

# Ensure only matching keys are evaluated
matching_keys = set(predictions.keys()) & set(ground_truth.keys())

if not matching_keys:
    raise ValueError("No matching transcript IDs found between predictions and ground truth!")

# Extract labels
y_true = [ground_truth[call_id] for call_id in matching_keys]
y_pred = [predictions[call_id] for call_id in matching_keys]

# Compute Performance Metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
report = classification_report(y_true, y_pred)

print(f"\nModel Evaluation Completed!")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Create a table for better visualization
eval_metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score": [accuracy, precision, recall, f1]
})
print("\nEvaluation Metrics:\n", eval_metrics)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=list(set(y_true)))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_true), yticklabels=set(y_true))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Misclassified Samples Logging
misclassified = {call_id: {"True": ground_truth[call_id], "Predicted": predictions[call_id]}
                 for call_id in matching_keys if ground_truth[call_id] != predictions[call_id]}

if misclassified:
    with open("output/misclassified_samples.json", "w", encoding="utf-8") as f:
        json.dump(misclassified, f, indent=4)
    print(f"\n⚠️ Misclassified samples logged in `output/misclassified_samples.json`")

# Class Distribution Analysis
plt.figure(figsize=(6, 5))
sns.countplot(x=y_true, order=list(set(y_true)), palette="pastel")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.title("Class Distribution in Ground Truth")
plt.show()
