import torch
import unittest
from metrics import IoU
from metrics import SEG


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


if __name__ == '__main__':
    unittest.main()