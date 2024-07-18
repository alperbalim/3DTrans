import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self, threshold=0.5):
        super(IoULoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred_boxes, gt_boxes):
        iou = self.compute_iou(pred_boxes, gt_boxes)
        mod_iou = torch.sigmoid(iou - self.threshold)
        return 1 - mod_iou.mean()

    def compute_iou(self, boxes1, boxes2):
        inter = self.intersection(boxes1, boxes2)
        area1 = self.volume(boxes1)
        area2 = self.volume(boxes2)
        union = area1 + area2 - inter
        return inter / union

    def intersection(self, boxes1, boxes2):
        max_xyz = torch.min(boxes1[:, :3] + boxes1[:, 3:6] / 2, boxes2[:, :3] + boxes2[:, 3:6] / 2)
        min_xyz = torch.max(boxes1[:, :3] - boxes1[:, 3:6] / 2, boxes2[:, :3] - boxes2[:, 3:6] / 2)
        inter_dim = torch.clamp(max_xyz - min_xyz, min=0)
        return inter_dim[:, 0] * inter_dim[:, 1] * inter_dim[:, 2]

    def volume(self, boxes):
        return boxes[:, 3] * boxes[:, 4] * boxes[:, 5]

# Kullanım
pred_boxes = torch.tensor([[0.5, 0.5, 0.5, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]], requires_grad=True)
gt_boxes = torch.tensor([[0.5, 0.5, 0.5, 1.0, 1.0, 1.0], [1.5, 1.5, 1.5, 2.0, 2.0, 2.0]])

criterion = IoULoss(threshold=0.5)
loss = criterion(pred_boxes, gt_boxes)
loss.backward()

print(pred_boxes.grad)
