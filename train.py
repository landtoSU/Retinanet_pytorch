import sys

sys.path.insert(0, "../input/mydata/retinanet")

import os
import argparse
import random
import shutil
import time
import warnings
import json
import numpy as np
from thop import profile
from thop import clever_format

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from config import Config
from retinanet.detection.dataset.csvdataset import collate_fn
# from retinanet.detection.models.loss import RetinaLoss
from retinanet.detection.models.decode import RetinaDecoder
from retinanet.detection.models.retinanet import resnet50_retinanet
from retinanet.imagenet.utils import get_logger
from pycocotools.cocoeval import COCOeval

torch.set_default_tensor_type(torch.DoubleTensor)



def csvnext(batch_data_iter):
    batch_data = next(batch_data_iter)
    images = list(batch_data[0])  # ([c,h,w],[c,h,w])
    for i in range(len(images)):
        images[i] = images[i].unsqueeze(0)
    images = torch.cat(images, axis=0)
    annotations = [torch.cat((batch_data[1][i]['boxes'],
                              torch.tensor(torch.ones(batch_data[1][i]['boxes'].shape[0], 1))), axis=1) for i in
                   range(len(batch_data[1]))]

    max_num_annots = max(annot.shape[0] for annot in annotations)
    #     print(annotations[0].shape)
    #     print(annotations[1].shape)
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annotations), max_num_annots, 5)) * (-1)

        if max_num_annots > 0:
            for idx, annot in enumerate(annotations):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annotations), 1, 5)) * (-1)
    images_id = list(batch_data[2])
    #     print(annot_padded)
    return images, annot_padded, images_id



import torch.nn as nn
import torch.nn.functional as F


class RetinaLoss(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 alpha=0.25,
                 gamma=2,
                 beta=1.0 / 9.0,
                 epsilon=1e-4):
        super(RetinaLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.image_w = image_w
        self.image_h = image_h

    def forward(self, cls_heads, reg_heads, batch_anchors, annotations):
        """
        compute cls loss and reg loss in one batch
        """
        device = annotations.device
        cls_heads = torch.cat(cls_heads, axis=1)
        reg_heads = torch.cat(reg_heads, axis=1)
        batch_anchors = torch.cat(batch_anchors, axis=1)

        cls_heads, reg_heads, batch_anchors = self.drop_out_border_anchors_and_heads(
            cls_heads, reg_heads, batch_anchors, self.image_w, self.image_h)
        batch_anchors_annotations = self.get_batch_anchors_annotations(
            batch_anchors, annotations)

        cls_loss, reg_loss = [], []
        valid_image_num = 0
        for per_image_cls_heads, per_image_reg_heads, per_image_anchors_annotations in zip(
                cls_heads, reg_heads, batch_anchors_annotations):
            # valid anchors contain all positive anchors
            valid_anchors_num = (per_image_anchors_annotations[
                per_image_anchors_annotations[:, 4] > 0]).shape[0]

            if valid_anchors_num == 0:
                cls_loss.append(torch.tensor(0.).to(device))
                reg_loss.append(torch.tensor(0.).to(device))
            else:
                valid_image_num += 1
                one_image_cls_loss = self.compute_one_image_focal_loss(
                    per_image_cls_heads, per_image_anchors_annotations)
                one_image_reg_loss = self.compute_one_image_smoothl1_loss(
                    per_image_reg_heads, per_image_anchors_annotations)
                cls_loss.append(one_image_cls_loss)
                reg_loss.append(one_image_reg_loss)

        cls_loss = sum(cls_loss) / valid_image_num
        reg_loss = sum(reg_loss) / valid_image_num

        return cls_loss, reg_loss

    def compute_one_image_focal_loss(self, per_image_cls_heads,
                                     per_image_anchors_annotations):
        """
        compute one image focal loss(cls loss)
        per_image_cls_heads:[anchor_num,num_classes]
        per_image_anchors_annotations:[anchor_num,5]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate focal loss
        per_image_cls_heads = per_image_cls_heads[
            per_image_anchors_annotations[:, 4] >= 0]
        per_image_anchors_annotations = per_image_anchors_annotations[
            per_image_anchors_annotations[:, 4] >= 0]

        per_image_cls_heads = torch.clamp(per_image_cls_heads,
                                          min=self.epsilon,
                                          max=1. - self.epsilon)
        num_classes = per_image_cls_heads.shape[1]

        # generate 80 binary ground truth classes for each anchor
        #         print(per_image_anchors_annotations[per_image_anchors_annotations[:, 4] > 0].shape)
        #         print(per_image_anchors_annotations.shape)
        loss_ground_truth = F.one_hot(per_image_anchors_annotations[:,
                                      4].long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        alpha_factor = torch.ones_like(per_image_cls_heads) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(loss_ground_truth, 1.), per_image_cls_heads,
                         1. - per_image_cls_heads)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        bce_loss = -(
                loss_ground_truth * torch.log(per_image_cls_heads) +
                (1. - loss_ground_truth) * torch.log(1. - per_image_cls_heads))

        one_image_focal_loss = focal_weight * bce_loss

        one_image_focal_loss = one_image_focal_loss.sum()
        positive_anchors_num = per_image_anchors_annotations[
            per_image_anchors_annotations[:, 4] > 0].shape[0]
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        one_image_focal_loss = one_image_focal_loss / positive_anchors_num

        return one_image_focal_loss

    def compute_one_image_smoothl1_loss(self, per_image_reg_heads,
                                        per_image_anchors_annotations):
        """
        compute one image smoothl1 loss(reg loss)
        per_image_reg_heads:[anchor_num,4]
        per_image_anchors_annotations:[anchor_num,5]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate smoothl1 loss
        device = per_image_reg_heads.device
        per_image_reg_heads = per_image_reg_heads[
            per_image_anchors_annotations[:, 4] > 0]
        per_image_anchors_annotations = per_image_anchors_annotations[
            per_image_anchors_annotations[:, 4] > 0]
        positive_anchor_num = per_image_anchors_annotations.shape[0]

        if positive_anchor_num == 0:
            return torch.tensor(0.).to(device)

        # compute smoothl1 loss
        loss_ground_truth = per_image_anchors_annotations[:, 0:4]
        x = torch.abs(per_image_reg_heads - loss_ground_truth)
        one_image_smoothl1_loss = torch.where(torch.ge(x, self.beta),
                                              x - 0.5 * self.beta,
                                              0.5 * (x ** 2) / self.beta)
        one_image_smoothl1_loss = one_image_smoothl1_loss.mean(axis=1).sum()
        # according to the original paper,We divide the smoothl1 loss by the number of positive sample anchors
        one_image_smoothl1_loss = one_image_smoothl1_loss / positive_anchor_num

        return one_image_smoothl1_loss

    def drop_out_border_anchors_and_heads(self, cls_heads, reg_heads,
                                          batch_anchors, image_w, image_h):
        """
        dropout out of border anchors,cls heads and reg heads
        """
        final_cls_heads, final_reg_heads, final_batch_anchors = [], [], []
        for per_image_cls_head, per_image_reg_head, per_image_anchors in zip(
                cls_heads, reg_heads, batch_anchors):
            per_image_cls_head = per_image_cls_head[per_image_anchors[:,
                                                    0] > 0.0]
            per_image_reg_head = per_image_reg_head[per_image_anchors[:,
                                                    0] > 0.0]
            per_image_anchors = per_image_anchors[per_image_anchors[:,
                                                  0] > 0.0]

            per_image_cls_head = per_image_cls_head[per_image_anchors[:,
                                                    1] > 0.0]
            per_image_reg_head = per_image_reg_head[per_image_anchors[:,
                                                    1] > 0.0]
            per_image_anchors = per_image_anchors[per_image_anchors[:,
                                                  1] > 0.0]

            per_image_cls_head = per_image_cls_head[
                per_image_anchors[:, 2] < image_w]
            per_image_reg_head = per_image_reg_head[
                per_image_anchors[:, 2] < image_w]
            per_image_anchors = per_image_anchors[
                per_image_anchors[:, 2] < image_w]

            per_image_cls_head = per_image_cls_head[
                per_image_anchors[:, 3] < image_h]
            per_image_reg_head = per_image_reg_head[
                per_image_anchors[:, 3] < image_h]
            per_image_anchors = per_image_anchors[
                per_image_anchors[:, 3] < image_h]

            per_image_cls_head = per_image_cls_head.unsqueeze(0)
            per_image_reg_head = per_image_reg_head.unsqueeze(0)
            per_image_anchors = per_image_anchors.unsqueeze(0)

            final_cls_heads.append(per_image_cls_head)
            final_reg_heads.append(per_image_reg_head)
            final_batch_anchors.append(per_image_anchors)

        final_cls_heads = torch.cat(final_cls_heads, axis=0)
        final_reg_heads = torch.cat(final_reg_heads, axis=0)
        final_batch_anchors = torch.cat(final_batch_anchors, axis=0)

        # final cls heads shape:[batch_size, anchor_nums, class_num]
        # final reg heads shape:[batch_size, anchor_nums, 4]
        # final batch anchors shape:[batch_size, anchor_nums, 4]
        return final_cls_heads, final_reg_heads, final_batch_anchors

    def get_batch_anchors_annotations(self, batch_anchors, annotations):
        """
        Assign a ground truth box target and a ground truth class target for each anchor
        if anchor gt_class index = -1,this anchor doesn't calculate cls loss and reg loss
        if anchor gt_class index = 0,this anchor is a background class anchor and used in calculate cls loss
        if anchor gt_class index > 0,this anchor is a object class anchor and used in
        calculate cls loss and reg loss
        """
        device = annotations.device
        assert batch_anchors.shape[0] == annotations.shape[0]
        one_image_anchor_nums = batch_anchors.shape[1]

        batch_anchors_annotations = []
        for one_image_anchors, one_image_annotations in zip(
                batch_anchors, annotations):
            # drop all index=-1 class annotations
            one_image_annotations = one_image_annotations[
                one_image_annotations[:, 4] >= 0]

            if one_image_annotations.shape[0] == 0:
                one_image_anchor_annotations = torch.ones(
                    [one_image_anchor_nums, 5], device=device) * (-1)
            else:
                one_image_gt_bboxes = one_image_annotations[:, 0:4]
                one_image_gt_class = one_image_annotations[:, 4]
                one_image_ious = self.compute_ious_for_one_image(
                    one_image_anchors, one_image_gt_bboxes)

                # snap per gt bboxes to the best iou anchor
                overlap, indices = one_image_ious.max(axis=1)
                # assgin each anchor gt bboxes for max iou annotation
                per_image_anchors_gt_bboxes = one_image_gt_bboxes[indices]
                # transform gt bboxes to [tx,ty,tw,th] format for each anchor
                one_image_anchors_snaped_boxes = self.snap_annotations_as_tx_ty_tw_th(
                    per_image_anchors_gt_bboxes, one_image_anchors)

                one_image_anchors_gt_class = (torch.ones_like(overlap) *
                                              -1).to(device)
                # if iou <0.4,assign anchors gt class as 0:background
                one_image_anchors_gt_class[overlap < 0.4] = 0
                # if iou >=0.5,assign anchors gt class as same as the max iou annotation class:80 classes index from 1 to 80
                one_image_anchors_gt_class[
                    overlap >=
                    0.5] = one_image_gt_class[indices][overlap >= 0.5]

                one_image_anchors_gt_class = one_image_anchors_gt_class.unsqueeze(
                    -1)

                one_image_anchor_annotations = torch.cat([
                    one_image_anchors_snaped_boxes, one_image_anchors_gt_class
                ],
                    axis=1)
            one_image_anchor_annotations = one_image_anchor_annotations.unsqueeze(
                0)
            batch_anchors_annotations.append(one_image_anchor_annotations)

        batch_anchors_annotations = torch.cat(batch_anchors_annotations,
                                              axis=0)

        # batch anchors annotations shape:[batch_size, anchor_nums, 5]
        return batch_anchors_annotations

    def snap_annotations_as_tx_ty_tw_th(self, anchors_gt_bboxes, anchors):
        """
        snap each anchor ground truth bbox form format:[x_min,y_min,x_max,y_max] to format:[tx,ty,tw,th]
        """
        anchors_w_h = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_w_h

        anchors_gt_bboxes_w_h = anchors_gt_bboxes[:,
                                2:] - anchors_gt_bboxes[:, :2]
        anchors_gt_bboxes_w_h = torch.clamp(anchors_gt_bboxes_w_h, min=1.0)
        anchors_gt_bboxes_ctr = anchors_gt_bboxes[:, :
                                                     2] + 0.5 * anchors_gt_bboxes_w_h

        snaped_annotations_for_anchors = torch.cat(
            [(anchors_gt_bboxes_ctr - anchors_ctr) / anchors_w_h,
             torch.log(anchors_gt_bboxes_w_h / anchors_w_h)],
            axis=1)

        device = snaped_annotations_for_anchors.device
        factor = torch.tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)

        snaped_annotations_for_anchors = snaped_annotations_for_anchors / factor

        # snaped_annotations_for_anchors shape:[batch_size, anchor_nums, 4]
        return snaped_annotations_for_anchors

    def compute_ious_for_one_image(self, one_image_anchors,
                                   one_image_annotations):
        """
        compute ious between one image anchors and one image annotations
        """
        # make sure anchors format:[anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        # make sure annotations format: [annotation_nums,4],4:[x_min,y_min,x_max,y_max]
        annotation_num = one_image_annotations.shape[0]

        one_image_ious = []
        for annotation_index in range(annotation_num):
            annotation = one_image_annotations[
                         annotation_index:annotation_index + 1, :]
            overlap_area_top_left = torch.max(one_image_anchors[:, :2],
                                              annotation[:, :2])
            overlap_area_bot_right = torch.min(one_image_anchors[:, 2:],
                                               annotation[:, 2:])
            overlap_area_sizes = torch.clamp(overlap_area_bot_right -
                                             overlap_area_top_left,
                                             min=0)
            overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]
            # anchors and annotations convert format to [x1,y1,w,h]
            anchors_w_h = one_image_anchors[:,
                          2:] - one_image_anchors[:, :2] + 1
            annotations_w_h = annotation[:, 2:] - annotation[:, :2] + 1
            # compute anchors_area and annotations_area
            anchors_area = anchors_w_h[:, 0] * anchors_w_h[:, 1]
            annotations_area = annotations_w_h[:, 0] * annotations_w_h[:, 1]

            # compute union_area
            union_area = anchors_area + annotations_area - overlap_area
            union_area = torch.clamp(union_area, min=1e-4)
            # compute ious between one image anchors and one image annotations
            ious = (overlap_area / union_area).unsqueeze(-1)

            one_image_ious.append(ious)

        one_image_ious = torch.cat(one_image_ious, axis=1)

        # one image ious shape:[anchors_num,annotation_num]
        return one_image_ious



def train(train_loader, model, criterion, optimizer, scheduler, epoch, logger):
    cls_losses, reg_losses, losses = [], [], []

    # switch to train mode
    model.train()
    #     iters = len(train_loader.dataset) // 8
    iters = len(train_loader.dataset) // Config.batch_size
    batch_data_iter = iter(train_loader)
    images, annotations, images_id = csvnext(batch_data_iter)
    iter_index = 1

    while images is not None:
        images, annotations = images.cuda().double(), annotations.cuda()
        cls_heads, reg_heads, batch_anchors = model(images)  # list len=5 P7 P6 P5..[batch, 57600,1] [batch,14400,1]..
        cls_loss, reg_loss = criterion(cls_heads, reg_heads, batch_anchors,
                                       annotations)
        loss = cls_loss + reg_loss
        if cls_loss == 0.0 or reg_loss == 0.0:
            optimizer.zero_grad()  # 梯度清零
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()  # 更新权重参数
        optimizer.zero_grad()

        cls_losses.append(cls_loss.item())
        reg_losses.append(reg_loss.item())
        losses.append(loss.item())
        print(loss.item())

        images, annotations, images_id = csvnext(batch_data_iter)

        if iter_index % Config.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>5d}, {iters:0>5d}], cls_loss: {cls_loss.item():.2f}, reg_loss: {reg_loss.item():.2f}, loss_total: {loss.item():.2f}"
            )

        iter_index += 1

    scheduler.step(np.mean(losses))

    return np.mean(cls_losses), np.mean(reg_losses), np.mean(losses)



def validate(val_dataset, model, decoder):
    model = model.module
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        all_eval_result = evaluate_coco(val_dataset, model, decoder)

    return all_eval_result


def evaluate_coco(val_dataset, model, decoder):
    results, image_ids = [], []
    for index in range(len(val_dataset)):
        data = val_dataset[index]
        scale = data['scale']
        cls_heads, reg_heads, batch_anchors = model(data['img'].cuda().permute(
            2, 0, 1).float().unsqueeze(dim=0))
        scores, classes, boxes = decoder(cls_heads, reg_heads, batch_anchors)
        scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
        boxes /= scale

        # make sure decode batch_size=1
        # scores shape:[1,max_detection_num]
        # classes shape:[1,max_detection_num]
        # bboxes shape[1,max_detection_num,4]
        assert scores.shape[0] == 1

        scores = scores.squeeze(0)
        classes = classes.squeeze(0)
        boxes = boxes.squeeze(0)

        # for coco_eval,we need [x_min,y_min,w,h] format pred boxes
        boxes[:, 2:] -= boxes[:, :2]

        for object_score, object_class, object_box in zip(
                scores, classes, boxes):
            object_score = float(object_score)
            object_class = int(object_class)
            object_box = object_box.tolist()
            if object_class == -1:
                break

            image_result = {
                'image_id':
                    val_dataset.image_ids[index],
                'category_id':
                    val_dataset.find_category_id_from_coco_label(object_class),
                'score':
                    object_score,
                'bbox':
                    object_box,
            }
            results.append(image_result)

        image_ids.append(val_dataset.image_ids[index])

        print('{}/{}'.format(index, len(val_dataset)), end='\r')

    if not len(results):
        print("No target detected in test set images")
        return

    json.dump(results,
              open('{}_bbox_results.json'.format(val_dataset.set_name), 'w'),
              indent=4)

    # load results in COCO evaluation tool
    coco_true = val_dataset.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(
        val_dataset.set_name))

    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    all_eval_result = coco_eval.stats

    return all_eval_result



def main(logger):
    if not torch.cuda.is_available():
        raise Exception("need gpu to train network!")

    torch.cuda.empty_cache()

    if Config.seed is not None:
        random.seed(Config.seed)
        torch.cuda.manual_seed_all(Config.seed)  # 所有GPU都是这个seed
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')
    print(f'use {gpus} gpus')

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    # dataset and dataloader
    logger.info('start loading data')
    print('start loading data')
    # coco train是118287张，train——loader为nums//batchsize   ---有多少个batch
    train_loader = DataLoader(Config.train_dataset,
                              batch_size=Config.batch_size,
                              shuffle=True,
                              num_workers=Config.num_workers,
                              collate_fn=collate_fn)
    logger.info('finish loading data')
    # model的结构定义是在网络class的__init__中实例化类时定义，也是在那时候叠加网络模块，init的作用就是给参数赋值和堆叠网络结构
    # forward中得到具体的值
    model = resnet50_retinanet(**{
        "pretrained": Config.pretrained,
        "num_classes": Config.num_classes,
    })
    # 循环打印每一次迭代的元素名字和param
    for name, param in model.named_parameters():
        logger.info(f"{name},{param.requires_grad}")
        # name 可为backbone.model.layer1.0.bn1.weight\backbone.model.layer1.0.bn1.bias\reg_head.reg_head.8.weight

    flops_input = torch.randn(1, 3, Config.input_image_size,
                              Config.input_image_size)  # (1,3,640,640)
    # FLOPs是计算力消耗  浮点运算数 - 计算量   衡量算法、模型的复杂程度（即卷积中的乘加运算）
    # 网络模型复杂度通常用计算量和参数量（考虑内存情况）描述。
    # 计算量和图像大小有关，参数量和图像大小无关。
    # 这里就是对该尺寸图像计算乘加次数和参数量
    flops, params = profile(model, inputs=(flops_input,))
    flops, params = clever_format([flops, params], "%.3f")
    logger.info(f"model: '{Config.network}', flops: {flops}, params: {params}")
    print(f"model: '{Config.network}', flops: {flops}, params: {params}")
    criterion = RetinaLoss(image_w=Config.input_image_size,
                           image_h=Config.input_image_size).cuda()
    decoder = RetinaDecoder(image_w=Config.input_image_size,
                            image_h=Config.input_image_size).cuda()

    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=3,
                                                           verbose=True)

    model = nn.DataParallel(model)
    print('model prepared')

    if Config.evaluate:
        if not os.path.isfile(Config.evaluate):
            raise Exception(
                f"{Config.resume} is not a file, please check it again")
        logger.info('start only evaluating')
        logger.info(f"start resuming model from {Config.evaluate}")
        checkpoint = torch.load(Config.evaluate,
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        all_eval_result = validate(Config.val_dataset, model, decoder)
        if all_eval_result is not None:
            logger.info(
                f"val: epoch: {checkpoint['epoch']:0>5d}, IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result[
                    0]:.3f}, IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result[
                    1]:.3f}, IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result[
                    2]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result[
                    3]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result[
                    4]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result[
                    5]:.3f}, IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result[
                    6]:.3f}, IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result[
                    7]:.3f}, IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result[
                    8]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result[
                    9]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result[
                    10]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result[11]:.3f}"
            )

        return

    best_map = 0.0
    start_epoch = 1
    print(best_map)
    # resume training
    if os.path.exists(Config.resume):
        logger.info(f"start resuming model from {Config.resume}")
        checkpoint = torch.load(Config.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        best_map = checkpoint['best_map']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(
            f"finish resuming model from {Config.resume}, epoch {checkpoint['epoch']}, best_map: {checkpoint['best_map']}, "
            f"loss: {checkpoint['loss']:3f}, cls_loss: {checkpoint['cls_loss']:2f}, reg_loss: {checkpoint['reg_loss']:2f}"
        )

    if not os.path.exists(Config.checkpoint_path):
        os.makedirs(Config.checkpoint_path)

    logger.info('start training')
    print('start training')
    for epoch in range(start_epoch, Config.epochs + 1):
        cls_losses, reg_losses, losses = train(train_loader, model, criterion,
                                               optimizer, scheduler, epoch,
                                               logger)
        logger.info(
            f"train: epoch {epoch:0>3d}, cls_loss: {cls_losses:.2f}, reg_loss: {reg_losses:.2f}, loss: {losses:.2f}"
        )
        print(
            f"train: epoch {epoch:0>3d}, cls_loss: {cls_losses:.2f}, reg_loss: {reg_losses:.2f}, loss: {losses:.2f}"
        )
        torch.save(
            {
                'epoch': epoch,
                'cls_loss': cls_losses,
                'reg_loss': reg_losses,
                'loss': losses,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join('../input/output', f'epoch:{epoch}.pth'))
        if epoch % 5 == 0 or epoch == Config.epochs:
            all_eval_result = validate(Config.val_dataset, model, decoder)
            logger.info(f"eval done.")
            if all_eval_result is not None:
                logger.info(
                    f"val: epoch: {checkpoint['epoch']:0>5d}, IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result[
                    0]:.3f}, IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result[
                    1]:.3f}, IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result[
                    2]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result[
                    3]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result[
                    4]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result[
                    5]:.3f}, IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result[
                    6]:.3f}, IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result[
                    7]:.3f}, IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result[
                    8]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result[
                    9]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result[
                    10]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result[11]:.3f}"
                )
                if all_eval_result[0] > best_map:
                    torch.save(model.module.state_dict(),
                               os.path.join(Config.checkpoint_path, "best.pth"))
                    best_map = all_eval_result[0]
        torch.save(
            {
                'epoch': epoch,
                'best_map': best_map,
                'cls_loss': cls_losses,
                'reg_loss': reg_losses,
                'loss': losses,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join('../input/output', 'latest.pth'))

    logger.info(f"finish training, best_map: {best_map:.3f}")
    training_time = (time.time() - start_time) / 3600
    logger.info(
        f"finish training, total training time: {training_time:.2f} hours")



logger = get_logger(__name__, Config.log)
main(logger)


