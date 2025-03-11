# Call-Outcome-Analysis-Using-GenAI
AI-powered sentiment analysis and call outcome classification using Mistral-7B and Gemma-7B. Automates customer service insights with NLP-driven analytics and visualization.

## Project Overview

GenAI Call Analysis is an AI-driven system that classifies customer service call transcripts into:

Sentiment Analysis – Identifies if customer sentiment is Positive, Negative, or Neutral.

Call Outcome Classification – Determines if an issue was Resolved or requires Follow-up Action.

This project helps businesses analyze customer interactions and improve service quality.

## How to Use

### 1. Install Dependencies

Ensure Python 3.8+ is installed. Install required libraries:

pip install -r requirements.txt

### 2. Run Sentiment Analysis

Classifies customer sentiment from call transcripts.

python3 src/sentiment_analysis.py

### 3. Run Call Outcome Classification

Predicts if an issue was resolved or needs further action.

python3 src/call_outcome_classifier.py

### 4. Evaluate Model Performance

Compares AI predictions with manual labels.

python3 src/evaluation.py

### 5. Generate Insights & Visuals

Creates graphs, misclassification reports, and word clouds.

python3 src/exploratory_analysis.py

## Key Results

Accuracy: Achieved X% accuracy in classifying call outcomes.

Insights: Most common customer concerns include billing issues and technical support.

Misclassification: Identified X% incorrect predictions, highlighting areas for improvement.

## Limitations & Improvements

### Challenges:

AI may misinterpret sarcasm or mixed emotions.

Data imbalance may lead to bias in classification.

### Future Enhancements:

Fine-tune models for better accuracy.

Introduce context-aware sentiment detection.

Enhance classification with industry-specific training data.
