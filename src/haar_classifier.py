import numpy as np
import os
import cv2
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
from .dataset import imgDataset


class haar_feature:
    """
    haar feature
    """

    def __init__(self) -> None:
        self.x2 = self.__get_haar_x2__()
        self.x3 = self.__get_haar_x3__()
        self.x4 = self.__get_haar_x4__()
        self.y2 = self.__get_haar_y2__()
        self.y3 = self.__get_haar_y3__()
        self.y4 = self.__get_haar_y4__()
        self.point = self.__get_point__()

        self.t_x2 = self.__get_tilted_haar_x2__()
        self.t_x3 = self.__get_tilted_haar_x3__()
        self.t_x4 = self.__get_tilted_haar_x4__()
        self.t_y2 = self.__get_tilted_haar_y2__()
        self.t_y3 = self.__get_tilted_haar_y3__()
        self.t_y4 = self.__get_tilted_haar_y4__()
        self.t_point = self.__get_tilted_point__()

        self.x2_y2 = self.__get_haar_x2_y2__()

        self.kernels = list()
        self.kernels.clear()

        self.kernels_torch = list()
        self.kernels_torch.clear()

    def init_shape(self, shapes: "list[tuple[int,int]]" = [(6, 6)]):
        """init numpy kernel

        Args:
            shapes (list[tuple[int,int]], optional): shapes of kernels. Defaults to [(6, 6)].
        """
        self.kernels.clear()
        for s in shapes:
            self.kernels.append(cv2.resize(self.x2, s))
            self.kernels.append(cv2.resize(self.x3, s))
            self.kernels.append(cv2.resize(self.x4, s))
            self.kernels.append(cv2.resize(self.y2, s))
            self.kernels.append(cv2.resize(self.y3, s))
            self.kernels.append(cv2.resize(self.y4, s))
            self.kernels.append(cv2.resize(self.point, s))

            self.kernels.append(cv2.resize(self.t_x2, s))
            self.kernels.append(cv2.resize(self.t_x3, s))
            self.kernels.append(cv2.resize(self.t_x4, s))
            self.kernels.append(cv2.resize(self.t_y2, s))
            self.kernels.append(cv2.resize(self.t_y3, s))
            self.kernels.append(cv2.resize(self.t_y4, s))
            self.kernels.append(cv2.resize(self.t_point, s))

            self.kernels.append(cv2.resize(self.x2_y2, s))

    def add_conv_torch(self, shape: "tuple[int,int]" = (6, 6), step: int = 1):
        """init torch kernel

        Args:
            shape (tuple[int,int], optional): shape of kernel. Defaults to (6,6).
            step (int, optional): stride. Defaults to 1.
        """
        s = shape
        conv_layer = nn.Conv2d(
            in_channels=1,
            out_channels=15,
            kernel_size=s,
            stride=step,
            padding=0,
            bias=False,
        )
        custom_kernel = torch.zeros(15, 1, s[0], s[1])
        custom_kernel[0, :, :] = torch.from_numpy(cv2.resize(self.x2, s))
        custom_kernel[1, :, :] = torch.from_numpy(cv2.resize(self.x3, s))
        custom_kernel[2, :, :] = torch.from_numpy(cv2.resize(self.x4, s))
        custom_kernel[3, :, :] = torch.from_numpy(cv2.resize(self.y2, s))
        custom_kernel[4, :, :] = torch.from_numpy(cv2.resize(self.y3, s))
        custom_kernel[5, :, :] = torch.from_numpy(cv2.resize(self.y4, s))
        custom_kernel[6, :, :] = torch.from_numpy(cv2.resize(self.point, s))

        custom_kernel[7, :, :] = torch.from_numpy(cv2.resize(self.t_x2, s))
        custom_kernel[8, :, :] = torch.from_numpy(cv2.resize(self.t_x3, s))
        custom_kernel[9, :, :] = torch.from_numpy(cv2.resize(self.t_x4, s))
        custom_kernel[10, :, :] = torch.from_numpy(cv2.resize(self.t_y2, s))
        custom_kernel[11, :, :] = torch.from_numpy(cv2.resize(self.t_y3, s))
        custom_kernel[12, :, :] = torch.from_numpy(cv2.resize(self.t_y4, s))
        custom_kernel[13, :, :] = torch.from_numpy(cv2.resize(self.t_point, s))

        custom_kernel[14, :, :] = torch.from_numpy(cv2.resize(self.x2_y2, s))

        conv_layer.weight.data = custom_kernel
        for param in conv_layer.parameters():
            param.requires_grad = False

        self.kernels_torch.append(conv_layer)

    def get_haars(self, img: np.ndarray) -> np.ndarray:
        ft = list()
        ft.clear()
        for k in self.kernels:
            o = cv2.filter2D(img, -1, k)
            o = o.flatten()
            ft.append(o)
        rst = np.concatenate(ft)
        return rst

    def get_haars_torch(self, img: torch.Tensor) -> torch.Tensor:
        rst = list()
        rst.clear()
        for st in self.kernels_torch:
            o = st(img)
            o = torch.flatten(o, start_dim=1)
            rst.append(o)
        rst = torch.cat(rst, dim=1)
        return rst

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.get_haars_torch(img)

    @staticmethod
    def __get_haar_x2__() -> np.ndarray:
        haar_x2 = np.ones((25, 25))
        haar_x2[:, :12] = -1
        haar_x2[:, 12] = 0
        return haar_x2

    @staticmethod
    def __get_haar_x3__() -> np.ndarray:
        haar_x3 = np.ones((25, 25))
        haar_x3[:, :8] = -1
        haar_x3[:, -8:] = -1
        return haar_x3

    @staticmethod
    def __get_haar_x4__() -> np.ndarray:
        haar_x4 = np.ones((25, 25))
        haar_x4[:, :6] = -1
        haar_x4[:, -6:] = -1
        return haar_x4

    @staticmethod
    def __get_point__() -> np.ndarray:
        point = np.ones((25, 25))
        point *= -1
        point[8:-8, 8:-8] = 1
        return point

    @staticmethod
    def __get_haar_y2__() -> np.ndarray:
        haar_y2 = np.ones((25, 25))
        haar_y2[:12, :] = -1
        haar_y2[12, :] = 0
        return haar_y2

    @staticmethod
    def __get_haar_x2_y2__() -> np.ndarray:
        haar_x2 = np.ones((25, 25))
        haar_x2[:12, :12] = -1
        haar_x2[-12:, -12:] = -1
        haar_x2[:, 12] = 0
        haar_x2[12, :] = 0
        return haar_x2

    @staticmethod
    def __get_tilted_haar_x2__() -> np.ndarray:
        haar_x2 = haar_feature.__get_haar_x2__()
        tilted_haar_x2 = haar_feature.rotate_no_pad(haar_x2, angle=135, scale=1.0)
        return tilted_haar_x2

    @staticmethod
    def __get_tilted_haar_y2__() -> np.ndarray:
        haar_y2 = haar_feature.__get_haar_y2__()
        tilted_haar_y2 = haar_feature.rotate_no_pad(haar_y2, angle=135, scale=1.0)
        return tilted_haar_y2

    @staticmethod
    def __get_tilted_haar_x3__() -> np.ndarray:
        haar_x3 = haar_feature.__get_haar_x3__()
        tilted_haar_x3 = haar_feature.rotate_no_pad(haar_x3, angle=135, scale=1.0)
        return tilted_haar_x3

    @staticmethod
    def __get_tilted_haar_x4__() -> np.ndarray:
        haar_x4 = haar_feature.__get_haar_x4__()
        tilted_haar_x4 = haar_feature.rotate_no_pad(haar_x4, angle=135, scale=1.0)
        return tilted_haar_x4

    @staticmethod
    def __get_haar_y3__() -> np.ndarray:
        haar_x3 = haar_feature.__get_haar_x3__()
        haar_y3 = haar_feature.rotate_no_pad(haar_x3, angle=-90, scale=1.0)
        return haar_y3

    @staticmethod
    def __get_haar_y4__() -> np.ndarray:
        haar_x4 = haar_feature.__get_haar_x4__()
        haar_y4 = haar_feature.rotate_no_pad(haar_x4, angle=-90, scale=1.0)
        return haar_y4

    @staticmethod
    def __get_tilted_haar_y3__() -> np.ndarray:
        haar_y3 = haar_feature.__get_haar_y3__()
        tilted_haar_y3 = haar_feature.rotate_no_pad(haar_y3, angle=135, scale=1.0)
        return tilted_haar_y3

    @staticmethod
    def __get_tilted_haar_y4__() -> np.ndarray:
        haar_y4 = haar_feature.__get_haar_y4__()
        tilted_haar_y4 = haar_feature.rotate_no_pad(haar_y4, angle=135, scale=1.0)
        return tilted_haar_y4

    @staticmethod
    def __get_tilted_point__() -> np.ndarray:
        point = haar_feature.__get_point__()
        tilted_point = haar_feature.rotate_no_pad(point, angle=135, scale=1.0)
        return tilted_point

    @staticmethod
    def rotate_no_pad(mat: np.ndarray, angle: float, scale: float) -> np.ndarray:
        (h, w) = mat.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        after_rotate = cv2.warpAffine(mat, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        return after_rotate


class MLP(nn.Module):
    """
    MLP
    """

    def __init__(self, node, act):
        super().__init__()
        self.node = node
        self.n = len(node)
        self.act = act
        self.model = nn.Sequential()
        for i in range(self.n - 1):
            self.model.append(nn.Linear(self.node[i], self.node[i + 1]))
            if self.act[i] != None:
                self.model.append(self.act[i])

    def forward(self, input):
        output = self.model(input)
        return output


class haar_adaboost:
    def __init__(
        self, haar_f: haar_feature, classifiers: "list[tuple[float,nn.Module]]"
    ):
        self.cls = classifiers
        self.hr_f = haar_f
        self.z = 0
        for alp, _ in self.cls:
            self.z += alp

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        output = 0
        feture = self.hr_f(input)
        for alp, cl in self.cls:
            output += alp * F.softmax(cl(feture), dim=1)
        output /= self.z
        return output

    def to(self, dev: torch.device):
        for k in self.hr_f.kernels_torch:
            k = k.to(dev)
        for _, c in self.cls:
            c = c.to(device=dev)


def train_haar_model(
    dataset: "Dataset|TensorDataset",
    model: nn.Module,
    lossfunc: nn.Module,
    batchsize: int,
    epoch: float,
    lr: float,
    haar_f: haar_feature = None,
    device=torch.device("cpu"),
):
    model = model.to(device=device)
    train_iter = Data.DataLoader(dataset, batchsize, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    sample_num = len(dataset)

    model.eval()
    total_train_step = 0

    start_time = time.time()

    logcontent = ""
    if haar_f != None:
        for k in haar_f.kernels_torch:
            k = k.to(device)

    for i in range(epoch):
        total_train_loss = 0
        for data in train_iter:
            features, targets, weight = data
            features = features.to(device)
            targets = targets.to(device)
            weight = weight.to(device)
            if haar_f != None:
                features = haar_f.get_haars_torch(features)
            outputs = model(features)
            loss = lossfunc(outputs, targets)
            weighted_loss = loss * weight
            total_loss = weighted_loss.sum()

            optimizer.zero_grad()
            total_loss.backward()

            optimizer.step()
            total_train_step += 1
            total_train_loss += total_loss
        src_head = "epoch: {}".format(i + 1)
        src = " mean loss: {}".format(total_train_loss / sample_num)
        logcontent += src_head + src + "\n"

    end_time = time.time()
    totaltime = "{:.3f}".format(end_time - start_time)
    log = "batchsize: {}\nepoch: {}\nlr: {}\ntotal_time: {}s\n".format(
        batchsize, epoch, lr, totaltime
    )

    if not os.path.exists("logs"):
        os.mkdir("logs")
    logpath = "logs\\{}.txt".format(start_time)
    file = open(logpath, "w")
    file.write(log + logcontent)
    file.close()


def adboost_haar_update(
    haar_f: haar_feature,
    classifier: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    device=torch.device("cpu"),
    ek_out: torch.Tensor = None,
) -> "tuple[float,torch.Tensor]":

    sample_num = len(x)
    for k in haar_f.kernels_torch:
        k = k.to(device)
    classifier = classifier.to(device)
    x = x.to(device)
    y = y.to(device)
    weight = weight.to(device)
    weight /= torch.sum(weight)

    haars = haar_f(x)
    gx = classifier(haars)
    gx = torch.argmax(gx, dim=1)
    diff = y - gx

    ek = torch.where(diff != 0, torch.ones_like(diff), torch.zeros_like(diff))
    if ek_out != None:
        ek_out.copy_(ek)
    ek = ek * weight
    ek = torch.sum(ek)
    ek = ek + 0.01 / sample_num
    alpha_k = 0.5 * torch.log2((1 - ek) / ek)

    yg = torch.where(diff == 0, torch.ones_like(diff), torch.ones_like(diff) * -1)
    exp_ayg = torch.exp(-alpha_k * yg)
    exp_ayg = exp_ayg * weight
    wk = exp_ayg / exp_ayg.sum()
    wk *= sample_num
    return alpha_k, wk


def mv_windows(
    cls: haar_adaboost,
    img: torch.Tensor,
    window_shapes: "list[tuple[int,int]]",
    cls_shape: "tuple[int,int]",
    dev: torch.device = torch.device("cpu"),
    need_flat: bool = False,
):
    y_max, x_max = img.shape[-2], img.shape[-1]
    boxes = []
    pics = []
    dim = img.dim()
    if dim > 4:
        return torch.zeros(0, 6)
    for _ in range(4 - dim):
        img = img.unsqueeze(0)

    for w_x, w_y in window_shapes:
        step_x = max(w_x // 2, 1)
        step_y = max(w_y // 2, 1)
        point_y = 0
        while point_y + w_y < y_max:
            point_x = 0
            while point_x + w_x < x_max:
                pic = img[..., point_y : point_y + w_y, point_x : point_x + w_x]
                pic = F.interpolate(
                    pic, size=cls_shape, mode="bilinear", align_corners=False
                )
                pics.append(pic)
                boxes.append([point_x, point_y, w_x, w_y])
                point_x += step_x
            point_y += step_y

    boxes = torch.Tensor(boxes)
    pics = torch.cat(pics, dim=0)
    if need_flat:
        pics = torch.flatten(pics, start_dim=1)

    pics = pics.to(dev)
    boxes = boxes.to(dev)

    boxes_tr = cls(pics)
    boxes_tr = torch.cat([boxes, boxes_tr], dim=1)

    return boxes_tr


def nms(
    boxes_tr: torch.Tensor, class_idx: int, trust_threshold: float, iou_threshold: float
) -> torch.Tensor:
    boxes_tr_c = boxes_tr[boxes_tr[:, class_idx + 4] >= trust_threshold]
    if boxes_tr_c.shape[0] == 0:
        return boxes_tr_c
    boxes = boxes_tr_c[:, :4].clone()
    boxes[:, 2:] = boxes_tr_c[:, 2:4] + boxes_tr_c[:, :2]
    scores = boxes_tr_c[:, class_idx + 4]
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    return boxes_tr_c[keep]
