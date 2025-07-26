import unittest
import os
import logging
from utils.logger import setup_logging

class TestLoggerSetup(unittest.TestCase):
    def test_logging_file_creation(self):
        log_dir = "logs"
        timestamp = "test1234"

        log_file, summary_file = setup_logging(log_dir, timestamp)
        session_dir = os.path.join(log_dir, f"log_{timestamp}")

        self.assertTrue(os.path.exists(log_file))
        self.assertTrue(os.path.exists(summary_file))

        # Write a test log
        logging.info("Test log message")

        with open(log_file, "r") as f:
            contents = f.read()
            self.assertIn("Test log message", contents)

        # Cleanup
        os.remove(log_file)
        os.remove(summary_file)
        os.rmdir(session_dir)

if __name__ == '__main__':
    unittest.main()
