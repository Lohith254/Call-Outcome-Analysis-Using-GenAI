import os
import json
from llama_cpp import Llama

# Load Mistral model
model_path = os.path.expanduser("~/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
llm = Llama(model_path=model_path, n_ctx=4096, n_batch=512)

def classify_call_outcome(text):
    """
    Classify call outcome as 'Issue Resolved' or 'Follow-up Action Needed' using Mistral 7B.
    The response must be exactly 'Issue Resolved' or 'Follow-up Action Needed'.
    """
    prompt = f"""
    You are a strict classifier. Read the following call transcript and classify it strictly as either:
    - "Issue Resolved" if the problem was solved.
    - "Follow-up Action Needed" if further action is required.

    Respond with only one of these two phrases and nothing else:
    Call Transcript: "{text}"
    """

    response = llm(prompt, max_tokens=10)  # Limiting tokens to prevent extra text
    result = response["choices"][0]["text"].strip()

    # Ensure result is strictly one of the two expected outputs
    if "follow-up" in result.lower():
        return "Follow-up Action Needed"
    elif "issue resolved" in result.lower():
        return "Issue Resolved"
    else:
        return "Follow-up Action Needed"  # Default fallback

# Directory containing transcript files
transcripts_dir = os.path.expanduser("~/Documents/genai-call-analysis/data/transcripts_v3")

# Check if the directory exists
if not os.path.exists(transcripts_dir):
    raise FileNotFoundError(f"Error: The directory `{transcripts_dir}` does not exist. Please check the path.")

# Initialize dictionary to store classification results
call_outcome_results = {}

# Read and classify all transcripts in the folder
for filename in os.listdir(transcripts_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(transcripts_dir, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()

        # Perform classification and store results using the **full filename as the key**
        call_outcome_results[filename] = classify_call_outcome(transcript_text)

# Ensure `output/` directory exists before writing results
output_dir = os.path.expanduser("~/Documents/genai-call-analysis/output")
os.makedirs(output_dir, exist_ok=True)

# Save results in the correct format
output_path = os.path.join(output_dir, "mistral_call_outcome_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(call_outcome_results, f, indent=4)

print(f"Call Outcome Classification Completed! Results saved in `{output_path}`.")
