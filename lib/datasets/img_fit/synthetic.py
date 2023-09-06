import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        view = kwargs['view']
        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.split = split
        self.batch_size = cfg.task_arg.N_pixels

        # read image
        image_paths = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format('train'))))
        for frame in json_info['frames']:
            image_paths.append(os.path.join(self.data_root, frame['file_path'][2:] + '.png'))# 去掉json文件中的./相对路径表示方法

        img = imageio.imread(image_paths[view])/255.# 归一化
        img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])# img[..., :3]是取RGB通道，img[..., -1:]取透明通道，这个操作用来获得一个考虑了透明度的RGB图像(注意相乘和相加的时候np会将数组进行广播)
        if self.input_ratio != 1.:# 检查是否需要缩放
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
        # set image
        self.img = np.array(img).astype(np.float32)
        # set uv
        H, W = img.shape[:2]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        u, v = X.astype(np.float32) / (W-1), Y.astype(np.float32) / (H-1)# 减一是因为第一个元素是从零开始
        self.uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32)# stack这一步是沿着最后一个轴拼接数组，然后reshape成一维数组，每个元素包含两个值

    def __getitem__(self, index):
        if self.split == 'train':# 随机选取图片上的一个坐标，读取其之前计算出的含透明度的RGB值
            ids = np.random.choice(len(self.uv), self.batch_size, replace=False)# 随机选择坐标
            uv = self.uv[ids]
            rgb = self.img.reshape(-1, 3)[ids]
        else:
            uv = self.uv
            rgb = self.img.reshape(-1, 3)
        ret = {'uv': uv, 'rgb': rgb} # input and output. they will be sent to cuda 返回uv坐标及对应的RGB值
        ret.update({'meta': {'H': self.img.shape[0], 'W': self.img.shape[1]}}) # meta means no need to send to cuda
        return ret

    def __len__(self):
        # we only fit 1 images, so we return 1
        return 1
