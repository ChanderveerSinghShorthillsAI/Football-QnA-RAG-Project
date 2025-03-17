import json
import numpy as np

# Load evaluation results
EVALUATION_RESULTS_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/evaluation_results.json"

with open(EVALUATION_RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

# Extract scores
bleu_scores = [r["bleu_score"] for r in results]
rouge_1_scores = [r["rouge_score"]["rouge-1"] for r in results]
rouge_2_scores = [r["rouge_score"]["rouge-2"] for r in results]
rouge_L_scores = [r["rouge_score"]["rouge-L"] for r in results]
f1_scores = [r["f1_score"] for r in results]

# Calculate averages
avg_bleu = np.mean(bleu_scores)
avg_rouge_1 = np.mean(rouge_1_scores)
avg_rouge_2 = np.mean(rouge_2_scores)
avg_rouge_L = np.mean(rouge_L_scores)
avg_f1 = np.mean(f1_scores)

# Find lowest-scoring answers
low_score_threshold = 0.1  # Define threshold for poor responses
bad_answers = [r for r in results if r["bleu_score"] < low_score_threshold]

# Print summary
print(" **Evaluation Summary**")
print(f" Average BLEU Score: {avg_bleu:.4f}")
print(f" Average ROUGE-1 Score: {avg_rouge_1:.4f}")
print(f" Average ROUGE-2 Score: {avg_rouge_2:.4f}")
print(f" Average ROUGE-L Score: {avg_rouge_L:.4f}")
print(f" Average F1 Score: {avg_f1:.4f}")
print(f"⚠️ Number of Low-Scoring Answers (BLEU < {low_score_threshold}): {len(bad_answers)}")

# Save bad answers for manual review
BAD_ANSWERS_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/low_scoring_answers_after_enhancement.json"
with open(BAD_ANSWERS_FILE, "w", encoding="utf-8") as f:
    json.dump(bad_answers, f, indent=4, ensure_ascii=False)

print(f" Low-scoring answers saved to `{BAD_ANSWERS_FILE}`")
