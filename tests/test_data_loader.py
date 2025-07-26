import unittest
import os
import shutil
from PIL import Image
import torch
from utils.data_loader import load_data

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.temp_data_dir = "temp_data"
        os.makedirs(os.path.join(self.temp_data_dir, "train", "class1"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_data_dir, "test", "class1"), exist_ok=True)

        # Create dummy images
        img = Image.new('RGB', (224, 224), color='red')
        for i in range(4):
            img.save(os.path.join(self.temp_data_dir, "train", "class1", f"img_{i}.jpg"))
            img.save(os.path.join(self.temp_data_dir, "test", "class1", f"img_{i}.jpg"))

    def tearDown(self):
        shutil.rmtree(self.temp_data_dir)

    def test_data_loader_shapes(self):
        batch_size = 2
        train_loader, val_loader, class_names = load_data(self.temp_data_dir, batch_size)

        self.assertEqual(class_names, ["class1"])

        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        self.assertEqual(train_batch[0].shape, (batch_size, 3, 224, 224))
        self.assertEqual(val_batch[0].shape, (batch_size, 3, 224, 224))
        self.assertEqual(train_batch[1].shape[0], batch_size)
        self.assertEqual(val_batch[1].shape[0], batch_size)

if __name__ == '__main__':
    unittest.main()
