import torch
from torchvision.ops import nms         # 不区分类别对所有bbox进行过滤。如果有不同类别的bbox重叠的话会导致被过滤掉并不会分开计算。
from torchvision.ops import batched_nms # 根据每个类别进行过滤，只对同一种类别进行计算IOU和阈值过滤。


boxes =  torch.Tensor([[2, 2, 4, 4], [1, 1, 5, 5], [3, 3, 3.5, 3.9]])
scores = torch.Tensor([0.9, 0.8, 0.9])
classes = torch.Tensor([0, 1, 0]) # 类别
iou_threshold = 0.1


idx = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)
print(idx)  # [0]
idx = batched_nms(boxes=boxes, scores=scores, idxs=classes, iou_threshold=iou_threshold)
print(idx)  # [0, 1]
