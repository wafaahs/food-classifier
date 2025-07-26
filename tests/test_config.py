import unittest
from config import get_config

class TestConfigParsing(unittest.TestCase):
    def test_default_values(self):
        args = get_config([])
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.lr, 0.001)
        self.assertEqual(args.optimizer, "adam")

    def test_custom_args(self):
        custom_args = ["--epochs", "10", "--batch_size", "64", "--lr", "0.01", "--optimizer", "sgd"]
        args = get_config(custom_args)
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.lr, 0.01)
        self.assertEqual(args.optimizer, "sgd")

if __name__ == '__main__':
    unittest.main()
