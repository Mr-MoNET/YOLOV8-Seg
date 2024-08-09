import os
import random
import cv2
import numpy as np
import torch
import sys

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops
from ultralytics.utils.plotting import colors

# yolov8 预处理步骤
# BGR -> RGB       : 颜色通道转换
# HWC -> CHW       : 改变通道顺序
# CHW -> BCHW      : 增加batch维度,改变通道数
# torch.from_numpy : 将图像转成PyTorch的Tensor数组
# im /= 255        : 除以255,图像归一化操作

# yolov8 后处理步骤
# ops.scale_boxes         : pred检查结果框的decode
# ops.non_max_suppression : NMS非极大值抑制
# annotator.box_label     : 绘制矩形框到原图上
# annotator.masks         : 绘制分割掩码图到原图上

class YOLOV8SegmentInfer:
    def __init__(self, weights, cuda, conf_thres, iou_thres) -> None:
        self.imgsz = 416
        self.device = torch.device(cuda)
        self.model = AutoBackend(weights, device=self.device)
        self.model.eval()
        # 打印mask标签名称
        self.names = self.model.names
        print(self.names)
        self.half = False
        self.conf = conf_thres
        self.iou = iou_thres
        self.color = {"font": (255, 255, 255)}
        self.color.update(
            {self.names[i]: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
             for i in range(len(self.names))})

    def infer(self, img_src):
        # img shape : [1, 3, 416, 416]
        img = self.precess_image(img_src)
        # 获取预测结果,包含检测框和mask
        preds = self.model(img)
        # 进行置信度过滤和非极大值抑制
        det = ops.non_max_suppression(preds[0], self.conf, self.iou, classes=None, agnostic=False, max_det=300, nc=len(self.names))

        # 从模型预测结果中提取原型掩码
        # preds[0]     : 包含了边界框的信息  
        # preds[1]     : 包含了原型掩码的信息(列表),包含多个tensor,对应了不同尺度下的原型掩码
        # preds[1][-1] : 最大尺度的原型掩码张量,这个张量会被用来生成最终的分割掩码
        proto = preds[1][-1]

        # 循环遍历
        for i, pred in enumerate(det):
            # 如果当前pred中没有检测到物体,则直接continue
            if pred.shape[0] == 0:
                print("Empty predictions")
                continue

            lw = max(round(sum(img_src.shape) / 2 * 0.003), 2)  # line width
            tf = max(lw - 1, 1)  # font thickness
            sf = lw / 3  # font scale

            # pred[:, :4]  : 预测的边界框坐标
            # pred[:, 5:6] : 预测边界框的类别名称,同mask标签名称
            print(pred[:, 5:6])
            labels = pred[:, 5:6]
            
            # 这里顺序不能换,否则影响分割精度
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], img_src.shape)
            pred_bbox = pred[:, :6].cpu().detach().numpy()

            # 绘制mask语义信息
            self.draw_masks(img_src, masks.data, labels, colors=[colors(x, True) for x in pred[:, 5]], im_gpu=img.squeeze(0))

            # 遍历绘制bbox
            for bbox in pred_bbox:
                if int(bbox[5]) == 0:
                    continue
                else:
                    self.draw_box(img_src, bbox[:4], bbox[4], self.names[int(bbox[5])], lw, sf, tf)

        return img_src

    def draw_box(self, img_src, box, conf, cls_name, lw, sf, tf):
        color = self.color[cls_name]
        label = f'{cls_name} {conf:.2f}'
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img_src, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
        outside = box[1] - h - 3 >= 0
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img_src, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(img_src, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, sf, self.color["font"], thickness=2, lineType=cv2.LINE_AA)

    def draw_masks(self, img_src, masks, labels, colors, im_gpu, alpha=0.5):
        if len(masks) == 0:
            img_src[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        if im_gpu.device != masks.device:
            im_gpu = im_gpu.to(masks.device)

        # 过滤出需要绘制的掩码
        filtered_masks = []
        for label, mask in zip(labels, masks):
            if label == 0:  # 如果类别是0，则保留这个掩码
                filtered_masks.append(mask)
                print(filtered_masks)
                print(type(filtered_masks))
        
        # 将过滤后的掩码转换为Tensor
        filtered_masks = torch.stack(filtered_masks)
        print('')
        print(filtered_masks)

        # 根据类别获取颜色
        colors = torch.tensor(colors, device=filtered_masks.device, dtype=torch.float32) / 255.0
        colors = colors[:, None, None]
 
        filtered_masks = filtered_masks.unsqueeze(3)
        masks_color = filtered_masks * (colors * alpha)
        inv_alpha_masks = (1 - filtered_masks * alpha).cumprod(0)
        mcs = masks_color.max(dim=0).values
        im_gpu = im_gpu.flip(dims=[0])
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()
        im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
        im_mask = im_gpu * 255
        im_mask_np = im_mask.byte().cpu().numpy()
        img_src[:] = ops.scale_image(im_mask_np, img_src.shape)

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 默认是图像的最小缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        if not scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad_h = int(shape[1] * r)
        new_unpad_w = new_unpad_h
        dw, dh = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != (new_unpad_w, new_unpad_h):
            im = cv2.resize(im, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)

    def precess_image(self, img_src):
        img = self.letterbox(img_src, self.imgsz)[0]
        img = np.expand_dims(img, axis=0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        return img

if __name__ == '__main__':
    # best.pt         实例分割模型有两个ouput
    # output0(bbox)   1, 38, 3549
    # 38 -----------> cx, cy, w, h, 2 class confident, 32 mask proto
    # output1(seg)    1, 32, 104, 104
    # 32 -----------> 32个104*104的proto map
    # 32 mask proto 是bbox的属性
    # mask = crop(sigmoid(sum(32 mask proto * 32 * 160 * 160, dim=0)))
    weights = r'/home/gao/Desktop/yolov8/runs/train/seg-0802/weights/best.pt'
    cuda = 'cuda:0'

    model = YOLOV8SegmentInfer(weights, cuda, 0.6, 0.2)

    frame = cv2.imread('/home/gao/Desktop/yolov8/person_89.jpg')

    res = model.infer(frame)

    cv2.namedWindow('my_window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('my_window',int(frame.shape[1] * 1.4), int(frame.shape[0] * 1.4))
    cv2.imshow('my_window', res)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
