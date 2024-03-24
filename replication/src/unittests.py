import torch
import unittest
from metrics import IoU
from metrics import SEG
from metrics import MuCov
from metrics import Metrics

class TestMetrics(unittest.TestCase):
    def test_compute(self):
        metrics = Metrics(IoU, SEG, MuCov)

        predictions = torch.tensor([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        targets = torch.tensor([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        expected_metrics = {
            'IoU': 1.0,
            'SEG': 1.0,
            'MUCov': 1.0
        }

        actual_metrics = metrics.compute(predictions, targets)
        self.assertEqual(actual_metrics, expected_metrics)

class TestIoU(unittest.TestCase):

    def test_IoU(self):
        # Create some test data
        predictions = torch.tensor([1, 0, 1, 1, 0])
        targets = torch.tensor([1, 0, 1, 0, 0])

        # Calculate IoU
        iou = IoU(predictions, targets)

        # Manually calculate expected IoU
        intersection = torch.sum(predictions * targets)
        TP = intersection
        FP = torch.sum(predictions) - intersection
        FN = torch.sum(targets) - intersection
        expected_iou = TP / (TP + FP + FN)

        # Check that the calculated IoU matches the expected IoU
        self.assertEqual(iou, expected_iou)

class TestSEG(unittest.TestCase):
    def test_SEG(self):
        predictions = torch.tensor([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        targets = torch.tensor([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        expected_SEG_score = 1.0  # The predictions perfectly match the targets
        actual_SEG_score = SEG(predictions, targets)
        self.assertEqual(actual_SEG_score, expected_SEG_score)

class TestMuCov(unittest.TestCase):
    def test_MuCov(self):
        predictions = torch.tensor([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        targets = torch.tensor([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        expected_MuCov_score = 1.0  # The predictions perfectly match the targets
        actual_MuCov_score = MuCov(predictions, targets)
        self.assertEqual(actual_MuCov_score, expected_MuCov_score)

if __name__ == '__main__':
    unittest.main()