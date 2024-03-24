import torch
import unittest
from metrics import IoU

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

if __name__ == '__main__':
    unittest.main()