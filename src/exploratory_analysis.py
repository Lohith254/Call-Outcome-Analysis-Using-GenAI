import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from datetime import datetime
from sklearn.metrics import confusion_matrix

# Load Predictions
with open("output/mistral_call_outcome_results.json", "r") as f:
    predictions = json.load(f)

# Load Ground Truth Labels
with open("output/manual_label_call_outcome.json", "r") as f:
    ground_truth = json.load(f)

# Convert Data to Pandas DataFrame
df = pd.DataFrame({
    "Transcript": list(predictions.keys()),
    "Predicted Outcome": [predictions[call_id] for call_id in predictions.keys()],
    "Actual Outcome": [ground_truth.get(call_id, "Unknown") for call_id in predictions.keys()]
})

# Class Distribution (Check Balance)
plt.figure(figsize=(6, 5))
sns.countplot(x=df["Actual Outcome"], order=df["Actual Outcome"].value_counts().index, palette="pastel")
plt.xlabel("Call Outcome")
plt.ylabel("Count")
plt.title("Class Distribution of Call Outcomes")
plt.show()

# Confusion Matrix Heatmap
cm = confusion_matrix(df["Actual Outcome"], df["Predicted Outcome"], labels=df["Actual Outcome"].unique())
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=df["Actual Outcome"].unique(), yticklabels=df["Actual Outcome"].unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of Call Outcomes")
plt.show()

# Misclassification Analysis
df["Misclassified"] = df["Actual Outcome"] != df["Predicted Outcome"]
misclassified = df[df["Misclassified"]]

if not misclassified.empty:
    print(f"\n{len(misclassified)} misclassified calls detected!")
    misclassified.to_csv("output/misclassified_calls.csv", index=False)
    print("Misclassified calls saved in `output/misclassified_calls.csv`.")

# WordCloud Analysis for Each Outcome
for outcome in df["Actual Outcome"].unique():
    text_data = " ".join(df[df["Actual Outcome"] == outcome]["Transcript"])

    if len(text_data) < 10:  # Ensure there is enough text
        print(f"⚠️ Skipping WordCloud for '{outcome}' due to insufficient text data.")
        continue

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Most Common Words in '{outcome}' Calls")
    plt.show()
