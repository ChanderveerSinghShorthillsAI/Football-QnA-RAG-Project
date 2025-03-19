import json
import numpy as np

class EvaluationAnalyzer:
    def __init__(self, results_file, bad_answers_file, low_score_threshold=0.1):
        self.results_file = results_file
        self.bad_answers_file = bad_answers_file
        self.low_score_threshold = low_score_threshold
        self.results = self.load_results()
    
    def load_results(self):
        with open(self.results_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def calculate_averages(self):
        bleu_scores = [r["bleu_score"] for r in self.results]
        rouge_1_scores = [r["rouge_score"]["rouge-1"] for r in self.results]
        rouge_2_scores = [r["rouge_score"]["rouge-2"] for r in self.results]
        rouge_L_scores = [r["rouge_score"]["rouge-L"] for r in self.results]
        f1_scores = [r["f1_score"] for r in self.results]
        
        return {
            "avg_bleu": np.mean(bleu_scores),
            "avg_rouge_1": np.mean(rouge_1_scores),
            "avg_rouge_2": np.mean(rouge_2_scores),
            "avg_rouge_L": np.mean(rouge_L_scores),
            "avg_f1": np.mean(f1_scores)
        }
    
    def find_low_scoring_answers(self):
        return [r for r in self.results if r["bleu_score"] < self.low_score_threshold]
    
    def save_low_scoring_answers(self, bad_answers):
        with open(self.bad_answers_file, "w", encoding="utf-8") as f:
            json.dump(bad_answers, f, indent=4, ensure_ascii=False)
    
    def generate_summary(self):
        averages = self.calculate_averages()
        bad_answers = self.find_low_scoring_answers()
        
        print(" **Evaluation Summary**")
        print(f" Average BLEU Score: {averages['avg_bleu']:.4f}")
        print(f" Average ROUGE-1 Score: {averages['avg_rouge_1']:.4f}")
        print(f" Average ROUGE-2 Score: {averages['avg_rouge_2']:.4f}")
        print(f" Average ROUGE-L Score: {averages['avg_rouge_L']:.4f}")
        print(f" Average F1 Score: {averages['avg_f1']:.4f}")
        print(f"⚠️ Number of Low-Scoring Answers (BLEU < {self.low_score_threshold}): {len(bad_answers)}")
        
        self.save_low_scoring_answers(bad_answers)
        print(f" Low-scoring answers saved to `{self.bad_answers_file}`")

if __name__ == "__main__":
    results_file = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/evaluation_results.json"
    bad_answers_file = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/low_scoring_answers_after_enhancement.json"
    analyzer = EvaluationAnalyzer(results_file, bad_answers_file)
    analyzer.generate_summary()



