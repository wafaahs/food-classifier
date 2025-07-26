import unittest
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from utils.model_builder import build_model

class TestPredictModule(unittest.TestCase):
    def setUp(self):
        self.model = build_model(num_classes=5)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.test_image_path = "test_sample.jpg"
        image = Image.new("RGB", (224, 224), color="blue")
        image.save(self.test_image_path)

    def tearDown(self):
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)

    def test_prediction_output_shape(self):
        image = Image.open(self.test_image_path)
        input_tensor = self.transform(image).unsqueeze(0)
        output = self.model(input_tensor)
        self.assertEqual(output.shape[1], 5)

    def test_prediction_is_tensor(self):
        image = Image.open(self.test_image_path)
        input_tensor = self.transform(image).unsqueeze(0)
        output = self.model(input_tensor)
        self.assertIsInstance(output, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
