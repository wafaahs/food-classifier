import unittest
import os
import shutil
import pandas as pd
from analysis.prediction_analysis import generate_report, write_summary_file

class TestPredictionAnalysis(unittest.TestCase):

    def setUp(self):
        self.test_df = pd.DataFrame({
            "ground_truth": ["sushi", "burger", "sushi", "pizza", "burger", "pizza"],
            "prediction":    ["burger", "burger", "sushi", "pizza", "sushi", "pizza"]
        })
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_generate_report_returns_valid_output(self):
        report_df, output_dir = generate_report(self.test_df, self.output_dir)
        self.assertIsInstance(report_df, pd.DataFrame)
        self.assertTrue(os.path.exists(output_dir))
        self.assertTrue("f1-score" in report_df.columns)

    def test_summary_file_created_and_content(self):
        report_df, worst = generate_report(self.test_df, self.output_dir)
        path = write_summary_file(report_df, self.output_dir)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            content = f.read()
            self.assertIn("Worst 5 Classes", content)
            self.assertTrue(any(cls in content for cls in worst))

if __name__ == '__main__':
    unittest.main()
