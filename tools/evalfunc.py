import numpy as np
import torch
import cv2
import math


def compute_iou_matrix(box_1: np.ndarray, box_2):
    box1 = box_1.copy()
    box1[:, 2:] = box_1[:, :2] + box_1[:, 2:]
    box2 = box_2.copy()
    box2[:, 2:] = box_2[:, :2] + box_2[:, 2:]
    box1_area = box1[:, 2] * box1[:, 3]
    box2_area = box2[:, 2] * box2[:, 3]

    box1_area = np.expand_dims(box1_area, -1)
    box2_area = np.expand_dims(box2_area, 0)

    min_point = np.maximum(np.expand_dims(box1[:, :2], 1), box2[:, :2])
    max_point = np.minimum(
        np.expand_dims(box1[:, :2] + box1[:, 2:4], 1), box2[:, :2] + box2[:, 2:4]
    )
    intersection = np.maximum(0.0, max_point - min_point)
    intersect_area = intersection[..., 0] * intersection[..., 1]
    union_area = box1_area + box2_area - intersect_area

    return intersect_area / union_area


def draw_bdbox(
    img: np.ndarray, boxes: torch.Tensor, color: tuple = (0, 255, 0), thick: int = 1
) -> np.ndarray:
    pic = img.copy()
    for b in boxes:
        x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        cv2.rectangle(pic, (x, y), (x + w, y + h), color, thick)
    return pic


def get_haar_adab_roc(
    pres: torch.Tensor, labels: torch.Tensor, class_idx: int, trust_p: torch.Tensor
):

    pres = pres[:, class_idx]

    fp_tp = torch.zeros(len(trust_p), 2)
    tr = torch.where(labels == class_idx, 1, 0)

    for i in range(len(trust_p)):
        pr = torch.where(pres > trust_p[i], 1, 0)
        tp = torch.sum(pr * tr)
        tn = torch.sum((1 - pr) * (1 - tr))
        fp = torch.sum(pr * (1 - tr))
        fn = torch.sum((1 - pr) * tr)

        fp_tp[i, 0] = fp / (fp + tn)
        fp_tp[i, 1] = tp / (tp + fn)

    return fp_tp


def draw_weight(weights: torch.Tensor):
    num = len(weights)
    mean = torch.mean(weights)
    line_num = int(math.sqrt(num))
    col_num = int(num / line_num) + 1
    canvas = np.ones((4 * line_num + 1, 4 * col_num + 1, 3), dtype=np.uint8) * 255

    for i in range(num):
        br = max(255 - weights[i] / mean * 64, 0)
        x = i % col_num
        y = i // col_num
        canvas[4 * y + 1 : 4 * y + 4, 4 * x + 1 : 4 * x + 4, [0, 2]] = br
    return canvas


def draw_ek(ek: torch.Tensor):
    num = len(ek)
    line_num = int(math.sqrt(num))
    col_num = int(num / line_num) + 1
    canvas = np.ones((4 * line_num + 1, 4 * col_num + 1, 3), dtype=np.uint8) * 255

    for i in range(num):
        bg = 223 - ek[i] * 223
        x = i % col_num
        y = i // col_num
        canvas[4 * y + 1 : 4 * y + 4, 4 * x + 1 : 4 * x + 4, :2] = bg
    return canvas
