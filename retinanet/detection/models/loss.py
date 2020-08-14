import torch
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
        print(per_image_anchors_annotations[:, 4].long())
        loss_ground_truth = F.one_hot(per_image_anchors_annotations[:,
                                                                    4].long(),
                                      num_classes=num_classes)
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
                                              0.5 * (x**2) / self.beta)
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
                    0.5] = one_image_gt_class[indices][overlap >= 0.5] + 1

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


if __name__ == '__main__':
    from retinanet import RetinaNet
    net = RetinaNet(resnet_type="resnet50")
    image_h, image_w = 600, 600
    cls_heads, reg_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    loss = RetinaLoss(image_w, image_h)
    cls_loss, reg_loss = loss(cls_heads, reg_heads, batch_anchors, annotations)
    print("1111", cls_loss, reg_loss)
