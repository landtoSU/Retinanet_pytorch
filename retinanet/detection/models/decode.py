import torch
import torch.nn as nn


class RetinaDecoder(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 min_score_threshold=0.05,
                 nms_threshold=0.5,
                 max_detection_num=100):
        super(RetinaDecoder, self).__init__()
        self.image_w = image_w
        self.image_h = image_h
        self.min_score_threshold = min_score_threshold
        self.nms_threshold = nms_threshold
        self.max_detection_num = max_detection_num

    def forward(self, cls_heads, reg_heads, batch_anchors):
        with torch.no_grad():
            device = cls_heads[0].device
            cls_heads = torch.cat(cls_heads, axis=1)
            reg_heads = torch.cat(reg_heads, axis=1)
            batch_anchors = torch.cat(batch_anchors, axis=1)

            batch_scores, batch_classes, batch_pred_bboxes = [], [], []
            for per_image_cls_heads, per_image_reg_heads, per_image_anchors in zip(
                    cls_heads, reg_heads, batch_anchors):
                pred_bboxes = self.snap_tx_ty_tw_th_reg_heads_to_x1_y1_x2_y2_bboxes(
                    per_image_reg_heads, per_image_anchors)
                scores, score_classes = torch.max(per_image_cls_heads, dim=1)
                score_classes = score_classes[
                    scores > self.min_score_threshold].float()
                pred_bboxes = pred_bboxes[
                    scores > self.min_score_threshold].float()
                scores = scores[scores > self.min_score_threshold].float()

                single_image_scores = (-1) * torch.ones(
                    (self.max_detection_num, ), device=device)
                single_image_classes = (-1) * torch.ones(
                    (self.max_detection_num, ), device=device)
                single_image_pred_bboxes = (-1) * torch.ones(
                    (self.max_detection_num, 4), device=device)

                if scores.shape[0] != 0:
                    scores, score_classes, pred_bboxes = self.nms(
                        scores, score_classes, pred_bboxes)

                    sorted_keep_scores, sorted_keep_scores_indexes = torch.sort(
                        scores, descending=True)
                    sorted_keep_classes = score_classes[
                        sorted_keep_scores_indexes]
                    sorted_keep_pred_bboxes = pred_bboxes[
                        sorted_keep_scores_indexes]

                    final_detection_num = min(self.max_detection_num,
                                              sorted_keep_scores.shape[0])

                    single_image_scores[
                        0:final_detection_num] = sorted_keep_scores[
                            0:final_detection_num]
                    single_image_classes[
                        0:final_detection_num] = sorted_keep_classes[
                            0:final_detection_num]
                    single_image_pred_bboxes[
                        0:final_detection_num, :] = sorted_keep_pred_bboxes[
                            0:final_detection_num, :]

                single_image_scores = single_image_scores.unsqueeze(0)
                single_image_classes = single_image_classes.unsqueeze(0)
                single_image_pred_bboxes = single_image_pred_bboxes.unsqueeze(
                    0)

                batch_scores.append(single_image_scores)
                batch_classes.append(single_image_classes)
                batch_pred_bboxes.append(single_image_pred_bboxes)

            batch_scores = torch.cat(batch_scores, axis=0)
            batch_classes = torch.cat(batch_classes, axis=0)
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, axis=0)

            # batch_scores shape:[batch_size,max_detection_num]
            # batch_classes shape:[batch_size,max_detection_num]
            # batch_pred_bboxes shape[batch_size,max_detection_num,4]
            return batch_scores, batch_classes, batch_pred_bboxes

    def nms(self, one_image_scores, one_image_classes, one_image_pred_bboxes):
        """
        one_image_scores:[anchor_nums],4:classification predict scores
        one_image_classes:[anchor_nums],class indexes for predict scores
        one_image_pred_bboxes:[anchor_nums,4],4:x_min,y_min,x_max,y_max
        """
        # Sort boxes
        sorted_one_image_scores, sorted_one_image_scores_indexes = torch.sort(
            one_image_scores, descending=True)
        sorted_one_image_classes = one_image_classes[
            sorted_one_image_scores_indexes]
        sorted_one_image_pred_bboxes = one_image_pred_bboxes[
            sorted_one_image_scores_indexes]
        sorted_pred_bboxes_w_h = sorted_one_image_pred_bboxes[:,
                                                              2:] - sorted_one_image_pred_bboxes[:, :
                                                                                                 2]

        sorted_pred_bboxes_areas = sorted_pred_bboxes_w_h[:,
                                                          0] * sorted_pred_bboxes_w_h[:,
                                                                                      1]
        detected_classes = torch.unique(sorted_one_image_classes, sorted=True)

        keep_scores, keep_classes, keep_pred_bboxes = [], [], []
        for detected_class in detected_classes:
            single_class_scores = sorted_one_image_scores[
                sorted_one_image_classes == detected_class]
            single_class_pred_bboxes = sorted_one_image_pred_bboxes[
                sorted_one_image_classes == detected_class]
            single_class_pred_bboxes_areas = sorted_pred_bboxes_areas[
                sorted_one_image_classes == detected_class]
            single_class = sorted_one_image_classes[sorted_one_image_classes ==
                                                    detected_class]

            single_keep_scores,single_keep_classes,single_keep_pred_bboxes=[],[],[]
            while single_class_scores.numel() > 0:
                top1_score, top1_class, top1_pred_bbox = single_class_scores[
                    0:1], single_class[0:1], single_class_pred_bboxes[0:1]

                single_keep_scores.append(top1_score)
                single_keep_classes.append(top1_class)
                single_keep_pred_bboxes.append(top1_pred_bbox)

                top1_areas = single_class_pred_bboxes_areas[0]

                if single_class_scores.numel() == 1:
                    break

                single_class_scores = single_class_scores[1:]
                single_class = single_class[1:]
                single_class_pred_bboxes = single_class_pred_bboxes[1:]
                single_class_pred_bboxes_areas = single_class_pred_bboxes_areas[
                    1:]

                overlap_area_top_left = torch.max(
                    single_class_pred_bboxes[:, :2], top1_pred_bbox[:, :2])
                overlap_area_bot_right = torch.min(
                    single_class_pred_bboxes[:, 2:], top1_pred_bbox[:, 2:])
                overlap_area_sizes = torch.clamp(overlap_area_bot_right -
                                                 overlap_area_top_left,
                                                 min=0)
                overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:,
                                                                             1]

                # compute union_area
                union_area = top1_areas + single_class_pred_bboxes_areas - overlap_area
                union_area = torch.clamp(union_area, min=1e-4)
                # compute ious for top1 pred_bbox and the other pred_bboxes
                ious = overlap_area / union_area

                single_class_scores = single_class_scores[
                    ious < self.nms_threshold]
                single_class = single_class[ious < self.nms_threshold]
                single_class_pred_bboxes = single_class_pred_bboxes[
                    ious < self.nms_threshold]
                single_class_pred_bboxes_areas = single_class_pred_bboxes_areas[
                    ious < self.nms_threshold]

            single_keep_scores = torch.cat(single_keep_scores, axis=0)
            single_keep_classes = torch.cat(single_keep_classes, axis=0)
            single_keep_pred_bboxes = torch.cat(single_keep_pred_bboxes,
                                                axis=0)

            keep_scores.append(single_keep_scores)
            keep_classes.append(single_keep_classes)
            keep_pred_bboxes.append(single_keep_pred_bboxes)

        keep_scores = torch.cat(keep_scores, axis=0)
        keep_classes = torch.cat(keep_classes, axis=0)
        keep_pred_bboxes = torch.cat(keep_pred_bboxes, axis=0)

        return keep_scores, keep_classes, keep_pred_bboxes

    def snap_tx_ty_tw_th_reg_heads_to_x1_y1_x2_y2_bboxes(
            self, reg_heads, anchors):
        """
        snap reg heads to pred bboxes
        reg_heads:[anchor_nums,4],4:[tx,ty,tw,th]
        anchors:[anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        """
        anchors_wh = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh

        device = anchors.device
        factor = torch.tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)

        reg_heads = reg_heads * factor

        pred_bboxes_wh = torch.exp(reg_heads[:, 2:]) * anchors_wh
        pred_bboxes_ctr = reg_heads[:, :2] * anchors_wh + anchors_ctr

        pred_bboxes_x_min_y_min = pred_bboxes_ctr - 0.5 * pred_bboxes_wh
        pred_bboxes_x_max_y_max = pred_bboxes_ctr + 0.5 * pred_bboxes_wh

        pred_bboxes = torch.cat(
            [pred_bboxes_x_min_y_min, pred_bboxes_x_max_y_max], axis=1)
        pred_bboxes = pred_bboxes.int()

        pred_bboxes[:, 0] = torch.clamp(pred_bboxes[:, 0], min=0)
        pred_bboxes[:, 1] = torch.clamp(pred_bboxes[:, 1], min=0)
        pred_bboxes[:, 2] = torch.clamp(pred_bboxes[:, 2],
                                        max=self.image_w - 1)
        pred_bboxes[:, 3] = torch.clamp(pred_bboxes[:, 3],
                                        max=self.image_h - 1)

        # pred bboxes shape:[anchor_nums,4]
        return pred_bboxes


if __name__ == '__main__':
    from retinanet import RetinaNet
    net = RetinaNet(resnet_type="resnet50")
    image_h, image_w = 640, 640
    cls_heads, reg_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    decode = RetinaDecoder(image_w, image_h)
    batch_scores, batch_classes, batch_pred_bboxes = decode(
        cls_heads, reg_heads, batch_anchors)
    print("1111", batch_scores.shape, batch_classes.shape,
          batch_pred_bboxes.shape)
