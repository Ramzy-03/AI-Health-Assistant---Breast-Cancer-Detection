import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_processor import load_data, preprocess

class TestDataProcessor(unittest.TestCase):
    def test_load_data(self):
        X, y = load_data()
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(len(X), len(y))

    def test_preprocess(self):
        X_train, X_test, y_train, y_test, scaler = preprocess(test_size=0.2)
        self.assertEqual(len(X_train) + len(X_test), 569) # Total samples in breast cancer dataset
        self.assertIsNotNone(scaler)

if __name__ == '__main__':
    unittest.main()
