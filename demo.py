import numpy as np
import cv2
import onnxruntime

def softmax(x, axis=-1):

    f_x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return f_x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def roi_align(feat,boxes,stride,output_size=(7,7),aligned=True):
    import torchvision
    import torch
    
    feat = torch.from_numpy(feat).unsqueeze(0)
    boxes = torch.from_numpy((boxes[:,:4]).astype(np.float32))
    out = torchvision.ops.roi_align(feat,[boxes],output_size=output_size,aligned=aligned,sampling_ratio=4,spatial_scale=1/stride)
    
    return out.numpy().astype(np.float32)

def map_roi_levels(rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (np.ndarray): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            np.ndarray: Level index (0-based) of each RoI, shape (k, )
        """
        finest_scale = 56
        scale = np.sqrt(
            (rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1]))
        target_lvls = np.floor(np.log2(scale / finest_scale + 1e-6))
        target_lvls = target_lvls.clip(min=0, max=num_levels - 1).astype(int)
        return target_lvls
def nms(boxes, probs, overlapThresh=0.5):

    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(probs)

    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # integer data type
    return pick

def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    stride=32,
):

    shape = im.shape[:2] 


    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
    
    if auto: 
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = 0, int(dh + 0.1)
    left, right = 0, int(dw + 0.1)
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    ) 
    return im, r, (dw, dh)

def meshgrid(x, y, row_major=True):
    """Generate mesh grid of x and y.

    Args:
        x : Grids of x dimension.
        y : Grids of y dimension.
        row_major (bool, optional): Whether to return y grids first.
            Defaults to True.

    Returns:
        tuple: The mesh grids of x and y.
    """
    # use shape instead of len to keep tracing while exporting to onnx
    xx = x.reshape(1,-1).repeat(y.shape[0],0).reshape(-1)
    yy = y.reshape(-1, 1).repeat(x.shape[0],1).reshape(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx
    

def delta2bbox(rois,
               deltas,
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               stds=[1,1,1,1],
               clip_border=True,
               add_ctr_clamp=False,
               ctr_clamp=32):
    num_bboxes, num_classes = deltas.shape[0], deltas.shape[1] // 4
    if num_bboxes == 0:
        return deltas

    deltas = deltas.reshape(-1, 4)
    
    
    stds = np.array(stds).reshape(1, -1)
    denorm_deltas = deltas * stds


    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:4]

    # Compute width/height of each roi
    rois_ = np.tile(rois,(1,num_classes)).reshape(-1, 4)
    pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
    pwh = (rois_[:, 2:] - rois_[:, :2])

    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = np.clip(dxy_wh,-ctr_clamp, ctr_clamp)
        dwh = np.clip(dwh, a_max=max_ratio)
    else:
        dwh = np.clip(dwh,-max_ratio, max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * np.exp(dwh)
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = np.concatenate([x1y1, x2y2], axis=-1)
    if clip_border and max_shape is not None:
        bboxes[..., 0::2].clip(min=0, max=max_shape[1])
        bboxes[..., 1::2].clip(min=0, max=max_shape[0])
    bboxes = bboxes.reshape(num_bboxes, -1)
    return bboxes


class onnxModel:
    def __init__(self,model1_path,model2_path):
        self.mean = [123.675, 116.28, 103.53] #RGB
        self.std = [58.395, 57.12, 57.375]
        
        self.base_size = (800, 1344) # h,w

        self.strides = [4, 8, 16, 32, 64]
        self.ratios   = [0.5, 1.0, 2.0]
        self.scales=np.array([8])
        self.center_offset = 0
        self.num_levels = len(self.strides)
        self.scale_major = True
        self.model1 = onnxruntime.InferenceSession(
            str(model1_path), providers=["CPUExecutionProvider"]
        )
        self.model2 = onnxruntime.InferenceSession(
            str(model2_path), providers=["CPUExecutionProvider"]
        )

        
        self.base_anchors = self.gen_base_anchors()
    
    
    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list: Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for base_size in self.strides:
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    ))
        return multi_level_base_anchors
    

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales : Scales of the anchor.
            ratios : The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            np.ndarray: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = np.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).reshape(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).reshape(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).reshape(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).reshape(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = np.stack(base_anchors, axis=-1)

        return base_anchors


    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
        Returns:
            Anchors in the overall feature maps.
        """

        base_anchors = self.base_anchors[level_idx]
        feat_h, feat_w = featmap_size
        stride = self.strides[level_idx]
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride

        shift_xx, shift_yy = meshgrid(shift_x, shift_y)
        shifts = np.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.reshape(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors
    def detect(self, bgr_img):
        
        score_thr = 0.25
        
        
        
        image = bgr_img[:, :, ::-1].copy()

        
        image,r,_ = letterbox(
            image, self.base_size, stride=32, auto=False
        )
        image_norm = np.array((image - self.mean) / self.std, dtype=np.float32)
        image_norm = image_norm.transpose(2, 0, 1)[np.newaxis, :, :, :]
        image_norm = np.ascontiguousarray(image_norm)
        
        
        
        # 第一阶段，backbone+RPN
        
        ort_inputs = {self.model1.get_inputs()[0].name: image_norm}
        ort_outs = self.model1.run(None, ort_inputs)
        
        ort_outs  = [i.squeeze() for i in ort_outs]
        
        
        
        cls_score_list = ort_outs[:5] # 分类分数 
        bbox_pred_list = ort_outs[5:10] # 边框回归
        feats = ort_outs[10:] # 图像特征，用来做ROIAlgin
        
        featmap_sizes = [cls_score_list[i].shape[-2:] for i in range(self.num_levels)]
        
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i)
            multi_level_anchors.append(anchors)


        nms_pre = 1000

        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        
        for level_idx in range(len(cls_score_list)):
            anchors = multi_level_anchors[level_idx]
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.shape[-2:] == rpn_bbox_pred.shape[-2:]
            rpn_cls_score = rpn_cls_score.transpose((1, 2, 0))
            rpn_cls_score = rpn_cls_score.reshape(-1)
            scores = sigmoid(rpn_cls_score)
            rpn_bbox_pred = rpn_bbox_pred.transpose((1, 2, 0)).reshape(-1, 4)
            
            
            
            if 0 < nms_pre < scores.shape[0]:

                rank_inds = scores.argsort()
                topk_inds = rank_inds[-nms_pre:]
                scores = scores[topk_inds]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)

        
        
        
        scores = np.concatenate(mlvl_scores)
        anchors = np.concatenate(mlvl_valid_anchors)
        rpn_bbox_pred = np.concatenate(mlvl_bbox_preds)

        
        proposals = delta2bbox(anchors,rpn_bbox_pred,self.base_size)
        
        
        w = proposals[:, 2] - proposals[:, 0]
        h = proposals[:, 3] - proposals[:, 1]
        valid_mask = (w > 0) & (h > 0)&(scores > score_thr)
        if not valid_mask.all():
            proposals = proposals[valid_mask]
            scores = scores[valid_mask]
        
        if proposals.size > 0:
            i = nms(proposals,scores)
            proposals = proposals[i]
            scores = scores[i]
        
            dets = np.concatenate((proposals,scores.reshape(-1,1)),axis=-1)
        else:
            return np.empty((0, 5), dtype=np.float32)

        
        num_levels = 4
        target_lvls = map_roi_levels(dets,num_levels)
        roi_feats = np.zeros((dets.shape[0],256,7,7),dtype=np.float32)
        for i in range(num_levels):
            mask = target_lvls == i
            inds = np.nonzero(mask.astype(int))
            rois_ = dets[inds]
            if rois_.size > 0:
                roi_feats[inds] = roi_align(feats[i], rois_,self.strides[i])
        
        
        # 第二阶段，分类回归
        
        ort_inputs = {self.model2.get_inputs()[0].name: roi_feats}
        ort_outs = self.model2.run(None, ort_inputs)

        
        cls_score, bbox_pred = ort_outs
        cls_score = softmax(cls_score, axis=-1)
        
        bboxes = delta2bbox(dets[:,:4], bbox_pred, self.base_size, clip_border=True,stds=[0.1,0.1,0.2,0.2])
    
        labels = np.arange(cls_score.shape[1]-1)
        labels = np.tile(labels.reshape(1, -1),(cls_score.shape[0],1)).reshape(-1)
        
        bboxes = bboxes.reshape(-1, 4)
        scores = cls_score[:,:-1].reshape(-1)
        
        valid_mask = scores > score_thr


        inds = np.nonzero(valid_mask.astype(int))
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
        
        if bboxes.shape[0] == 0:
            return np.empty((0, 5), dtype=np.float32)


        i = nms(bboxes,scores)
        bboxes = bboxes[i]
        scores = scores[i]
        labels = labels[i]


        dets  = np.concatenate((bboxes/r,scores.reshape(-1,1),labels.reshape(-1,1)),axis=-1)


        return dets


        

if __name__=='__main__':
    path1=r'./ckpt/model1.onnx'
    path2=r'./ckpt/model2.onnx'
    im_path = "./demo.jpg"
    
    
    model = onnxModel(path1,path2)
    im = cv2.imread(im_path,1)
    out = model.detect(im)
    

    for det in out:
        x1,y1,x2,y2 = list(map(int,det[:4]))

        cv2.rectangle(im,(x1,y1),(x2,y2),[255,0,0],1)
        cv2.putText(im, f'{int(det[5])}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("im",im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
