import unittest
import torch
from torchvision import models
from utils.model_builder import build_model
from utils.trainer import train_one_epoch
from utils.validator import validate

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=10, num_classes=5):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.rand(3, 224, 224)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label

class TestModelPipeline(unittest.TestCase):
    def setUp(self):
        self.num_classes = 5
        self.batch_size = 2
        self.device = torch.device("cpu")
        self.model = build_model(self.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        dataset = DummyDataset(num_classes=self.num_classes)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    def test_model_forward_shape(self):
        inputs = torch.rand(2, 3, 224, 224).to(self.device)
        outputs = self.model(inputs)
        self.assertEqual(outputs.shape, (2, self.num_classes))

    def test_training_loop(self):
        loss = train_one_epoch(self.model, self.loader, self.criterion, self.optimizer, self.device, 0, 1, start_time=0, current_epoch_start_time=0)
        self.assertIsInstance(loss, float)

    def test_validation_loop(self):
        avg_loss, accuracy = validate(self.model, self.loader, self.criterion, [str(i) for i in range(self.num_classes)], self.num_classes, self.device, start_time=0)
        self.assertIsInstance(avg_loss, float)
        self.assertTrue(0.0 <= accuracy <= 100.0)

if __name__ == '__main__':
    unittest.main()
