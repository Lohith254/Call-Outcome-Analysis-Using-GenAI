import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Gemma Model with Optimized Settings
model_name = "google/gemma-7b"
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Uses half-precision for faster inference
    device_map="auto"  # Automatically assigns to the best available hardware (CPU/GPU)
).eval()

# Define path for transcripts
transcripts_dir = os.path.expanduser("~/Documents/genai-call-analysis/data/transcripts_v3")

# Function to extract only customer messages from transcripts
def extract_customer_text(text):
    """
    Extracts the customer's side of the conversation from the transcript.
    Assumes agent messages start with 'Agent:' and customer messages with 'Customer:'.
    """
    customer_lines = []
    for line in text.split("\n"):
        if line.strip().startswith("Member:"):  # Extract only customer messages
            customer_lines.append(line.replace("Member:", "").strip())

    return " ".join(customer_lines)  # Join into a single text block

# Function to read transcripts
def read_transcripts(directory):
    transcripts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()

                # Extract only customer messages for sentiment analysis
                customer_text = extract_customer_text(text)

                transcripts[filename] = customer_text
    return transcripts

# Load transcripts
customer_data = read_transcripts(transcripts_dir)

# Limit input size to improve performance
MAX_WORDS = 300

def truncate_text(text):
    """Keeps only the first 300 words of the transcript to speed up processing."""
    return " ".join(text.split()[:MAX_WORDS])

# Function to analyze sentiment using Gemma
def classify_sentiment(text, file_name):
    print(f"Processing transcript: {file_name}")  # Debugging log

    # Apply truncation to reduce input size
    text = truncate_text(text)

    prompt = f"""
    You are an AI trained for sentiment analysis of customer service calls. 
    You will be given a conversation where only the customer’s statements are provided.
    Classify the sentiment of this conversation strictly as either:
    - "Positive" (if the customer expresses positive emotions, satisfaction, or gratitude).
    - "Negative" (if the customer expresses frustration, complaints, or dissatisfaction).
    - "Neutral" (if the conversation is factual, lacks strong emotions, or is mixed).

    Respond with ONLY ONE of these words: "Positive", "Negative", or "Neutral".

    Customer Conversation: "{text}"
    """

    # Tokenize and move inputs to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Generate response
    output = model.generate(**inputs, max_new_tokens=5)

    # Decode the result
    result = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    print(f"Result for {file_name}: {result}")  # Debugging log

    # Enforce strict classification
    if "positive" in result.lower():
        return "Positive"
    elif "negative" in result.lower():
        return "Negative"
    else:
        return "Neutral"

# Perform sentiment analysis only on customer messages
sentiment_results = {
    file_name: classify_sentiment(text, file_name)
    for file_name, text in customer_data.items()
}

# Ensure `output/` directory exists before writing results
output_dir = os.path.expanduser("~/Documents/genai-call-analysis/output")
os.makedirs(output_dir, exist_ok=True)

# Save results
output_path = os.path.join(output_dir, "gemma7b_sentiment_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sentiment_results, f, indent=4)

print("Sentiment Analysis Completed! Results saved in `output/gemma7b_sentiment_results.json`.")
