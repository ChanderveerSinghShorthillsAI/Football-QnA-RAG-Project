# Description: This script generates a summary of test results and exports them to an Excel file.
import pandas as pd
import json


class EvaluationSummary:
    """Class to generate a summary of test results and export to Excel."""

    def __init__(self, results_file, output_file="Output/evaluation_summary.xlsx", threshold=0.5):
        self.results_file = results_file
        self.output_file = output_file
        self.threshold = threshold
        self.results = self.load_results()
        self.df = pd.DataFrame(self.results)

    def load_results(self):
        """Load test results from a JSON file."""
        try:
            with open(self.results_file, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            print(" Error: Unable to load results. Check file path or format.")
            return []

    def process_results(self):
        """Process and enrich results with pass/fail status and weighted scores."""
        # Create a DataFrame with required columns
        self.df = pd.DataFrame(self.results)

        # Fill NaN values with 0 for scoring consistency
        self.df.fillna(0, inplace=True)

        # Determine pass/fail for each metric based on threshold
        self.df["Faithfulness Test"] = self.df["faithfulness_score"].apply(
            lambda x: "Passed" if x >= self.threshold else "Failed"
        )
        self.df["Context Precision Test"] = self.df["context_precision_score"].apply(
            lambda x: "Passed" if x >= self.threshold else "Failed"
        )
        self.df["Correctness Test"] = self.df["correctness_score"].apply(
            lambda x: "Passed" if x >= self.threshold else "Failed"
        )

        # Calculate weighted overall score
        self.calculate_weighted_score()

    def calculate_weighted_score(self, faithfulness_weight=0.4, context_weight=0.3, correctness_weight=0.3):
        """Calculate the overall score using a weighted average."""
        self.df["overall_score"] = (
            self.df["faithfulness_score"] * faithfulness_weight
            + self.df["context_precision_score"] * context_weight
            + self.df["correctness_score"] * correctness_weight
        )

        # Determine overall pass/fail based on weighted score
        self.df["Overall Test Result"] = self.df["overall_score"].apply(
            lambda x: "Passed" if x >= self.threshold else "Failed"
        )

    def generate_summary(self):
        """Generate a summary sheet with test results."""
        if self.df.empty:
            print("⚠️ No data available for summary generation.")
            return

        # Count passed/failed results for each metric
        summary_data = {
            "Metric": ["Faithfulness", "Context Precision", "Correctness"],
            "Passed": [
                (self.df["Faithfulness Test"] == "Passed").sum(),
                (self.df["Context Precision Test"] == "Passed").sum(),
                (self.df["Correctness Test"] == "Passed").sum(),
            ],
            "Failed": [
                (self.df["Faithfulness Test"] == "Failed").sum(),
                (self.df["Context Precision Test"] == "Failed").sum(),
                (self.df["Correctness Test"] == "Failed").sum(),
            ],
        }

        # Calculate percentage of passed/failed test cases for each metric
        total_tests = len(self.df)
        summary_data["Pass Percentage (%)"] = [
            (summary_data["Passed"][i] / total_tests) * 100 if total_tests > 0 else 0
            for i in range(3)
        ]
        summary_data["Fail Percentage (%)"] = [
            (summary_data["Failed"][i] / total_tests) * 100 if total_tests > 0 else 0
            for i in range(3)
        ]

        # Count overall pass/fail
        total_passed = (self.df["Overall Test Result"] == "Passed").sum()
        total_failed = (self.df["Overall Test Result"] == "Failed").sum()

        overall_summary_data = {
            "Total Test Cases": total_tests,
            "Overall Passed": total_passed,
            "Overall Failed": total_failed,
            "Overall Pass Percentage (%)": (total_passed / total_tests) * 100 if total_tests > 0 else 0,
            "Overall Fail Percentage (%)": (total_failed / total_tests) * 100 if total_tests > 0 else 0,
            "Weighting Applied": "Faithfulness (40%), Context Precision (30%), Correctness (30%)",
        }

        return pd.DataFrame(summary_data), pd.DataFrame([overall_summary_data])

    def save_to_excel(self):
        """Save results and summary to an Excel file."""
        self.process_results()
        summary_df, overall_summary_df = self.generate_summary()

        # Define output structure
        with pd.ExcelWriter(self.output_file, engine="xlsxwriter") as writer:
            self.df.to_excel(writer, sheet_name="Test Cases", index=False)
            summary_df.to_excel(writer, sheet_name="Metric Summary", index=False)
            overall_summary_df.to_excel(writer, sheet_name="Overall Summary", index=False)

        print(f"✅ Summary generated and saved to '{self.output_file}'")


if __name__ == "__main__":
    # Input and output file paths
    results_file = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/evaluation_results/evaluation_result_ragas.json"
    output_file = "Output/evaluation_summary.xlsx"

    # Create and execute summary generation
    summarizer = EvaluationSummary(results_file, output_file, threshold=0.5)
    summarizer.save_to_excel()
