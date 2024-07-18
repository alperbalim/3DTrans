import torch

def compute_intersection(boxes1, boxes2):
    max_xyz = torch.min(boxes1[:, :3] + boxes1[:, 3:6] / 2, boxes2[:, :3] + boxes2[:, 3:6] / 2)
    min_xyz = torch.max(boxes1[:, :3] - boxes1[:, 3:6] / 2, boxes2[:, :3] - boxes2[:, 3:6] / 2)
    inter_dim = torch.clamp(max_xyz - min_xyz, min=0)
    return inter_dim[:, 0] * inter_dim[:, 1] * inter_dim[:, 2], inter_dim

def compute_volume(boxes):
    return boxes[:, 3] * boxes[:, 4] * boxes[:, 5]

def compute_iou(boxes1, boxes2):
    inter, inter_dim = compute_intersection(boxes1, boxes2)
    area1 = compute_volume(boxes1)
    area2 = compute_volume(boxes2)
    union = area1 + area2 - inter
    return inter / union

def compute_manual_grad(pred_boxes, gt_boxes):
    inter, inter_dim = compute_intersection(pred_boxes, gt_boxes)
    area1 = compute_volume(pred_boxes)
    area2 = compute_volume(gt_boxes)
    union = area1 + area2 - inter

    dI_dx = torch.zeros_like(pred_boxes)
    dU_dx = torch.zeros_like(pred_boxes)

    # Kesişim hacminin türevlerini hesapla
    for i in range(3):  # x, y, z için
        dI_dx[:, i] = ((pred_boxes[:, i] + pred_boxes[:, i+3] / 2 > gt_boxes[:, i] - gt_boxes[:, i+3] / 2) & \
                       (pred_boxes[:, i] - pred_boxes[:, i+3] / 2 < gt_boxes[:, i] + gt_boxes[:, i+3] / 2)).float() * \
                      (inter_dim[:, (i+1)%3] * inter_dim[:, (i+2)%3])

    for i in range(3, 6):  # w, h, l için
        dI_dx[:, i] = ((pred_boxes[:, i-3] + pred_boxes[:, i] / 2 > gt_boxes[:, i-3] - gt_boxes[:, i] / 2) & \
                       (pred_boxes[:, i-3] - pred_boxes[:, i] / 2 < gt_boxes[:, i-3] + gt_boxes[:, i] / 2)).float() * \
                      (inter_dim[:, (i+1-3)%3] * inter_dim[:, (i+2-3)%3]) * 0.5

    # Birleşim hacminin türevlerini hesapla
    for i in range(3):
        dU_dx[:, i] = dI_dx[:, i]

    dU_dx[:, 3] = (union / pred_boxes[:, 3])
    dU_dx[:, 4] = (union / pred_boxes[:, 4])
    dU_dx[:, 5] = (union / pred_boxes[:, 5])

    dIoU_dx = (dI_dx * union.unsqueeze(1) - inter.unsqueeze(1) * dU_dx) / (union.unsqueeze(1) ** 2)

    return dIoU_dx

# PyTorch türev hesaplama fonksiyonları
class IoULoss(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super(IoULoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred_boxes, gt_boxes):
        iou = self.compute_iou(pred_boxes, gt_boxes)
        mod_iou = torch.sigmoid(iou - self.threshold)
        return 1 - mod_iou.mean()

    def compute_iou(self, boxes1, boxes2):
        inter, inter_dim = self.compute_intersection(boxes1, boxes2)
        area1 = self.compute_volume(boxes1)
        area2 = self.compute_volume(boxes2)
        union = area1 + area2 - inter
        return inter / union

    def compute_intersection(self, boxes1, boxes2):
        max_xyz = torch.min(boxes1[:, :3] + boxes1[:, 3:6] / 2, boxes2[:, :3] + boxes2[:, 3:6] / 2)
        min_xyz = torch.max(boxes1[:, :3] - boxes1[:, 3:6] / 2, boxes2[:, :3] - boxes2[:, 3:6] / 2)
        inter_dim = torch.clamp(max_xyz - min_xyz, min=0)
        return inter_dim[:, 0] * inter_dim[:, 1] * inter_dim[:, 2], inter_dim

    def compute_volume(self, boxes):
        return boxes[:, 3] * boxes[:, 4] * boxes[:, 5]

# Örnek veri
pred_boxes = torch.tensor([[0.5, 0.5, 0.5, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]], requires_grad=True)
gt_boxes = torch.tensor([[0.5, 0.5, 0.5, 1.0, 1.0, 1.0], [1.5, 1.5, 1.5, 2.0, 2.0, 2.0]])

# PyTorch türev hesaplaması
criterion = IoULoss(threshold=0.5)
loss = criterion(pred_boxes, gt_boxes)
loss.backward()
pytorch_grads = pred_boxes.grad.clone()

# Manuel türev hesaplaması
manual_grads = compute_manual_grad(pred_boxes, gt_boxes)

print("PyTorch Gradyanları:\n", pytorch_grads)
print("Manuel Gradyanlar:\n", manual_grads)
