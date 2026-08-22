import glob
import os

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from mmdet.apis import init_detector
from mmdet.structures.bbox import distance2bbox
from torchvision.ops import nms as tv_nms

CONFIG = 'mmdetection-configs/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
CHECKPOINT = 'ckpt/rtmdet_s.pth'

model = init_detector(CONFIG, CHECKPOINT, device='cpu')
model.eval()
print('Loaded. num_classes:', model.bbox_head.num_classes, 'strides:', model.bbox_head.prior_generator.strides)


class RTMDetRawExport(torch.nn.Module):
    def __init__(self, det_model):
        super().__init__()
        self.backbone = det_model.backbone
        self.neck = det_model.neck
        self.bbox_head = det_model.bbox_head

    def forward(self, img):
        feats = self.neck(self.backbone(img))
        cls_scores, bbox_preds = self.bbox_head(feats)

        featmap_sizes = [c.shape[-2:] for c in cls_scores]
        mlvl_priors = self.bbox_head.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)

        flatten_cls, flatten_bbox = [], []
        for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
            b, c, h, w = cls_score.shape
            flatten_cls.append(cls_score.permute(0, 2, 3, 1).reshape(b, h * w, c))
            bb, bc, bh, bw = bbox_pred.shape
            flatten_bbox.append(bbox_pred.permute(0, 2, 3, 1).reshape(bb, bh * bw, bc))

        cls_scores_cat = torch.cat(flatten_cls, dim=1)
        bbox_preds_cat = torch.cat(flatten_bbox, dim=1)
        priors_cat = torch.cat(mlvl_priors, dim=0)

        bboxes = distance2bbox(priors_cat[None, :, :2], bbox_preds_cat)
        scores = cls_scores_cat.sigmoid()
        return bboxes, scores


export_model = RTMDetRawExport(model)
export_model.eval()

dummy = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    b, s = export_model(dummy)
print('Wrapper output shapes -> boxes:', b.shape, 'scores:', s.shape)

# ---- sanity check on a real image, compared against mmdetection's own official inference ----
if not os.path.exists('test.jpg'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/open-mmlab/mmdetection/main/demo/demo.jpg', 'test.jpg')

img_bgr = cv2.imread('test.jpg')
orig_h, orig_w = img_bgr.shape[:2]
scale = min(640 / orig_h, 640 / orig_w)
new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
resized = cv2.resize(img_bgr, (new_w, new_h))
canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
canvas[0:new_h, 0:new_w] = resized

mean = np.array(model.data_preprocessor.mean).reshape(1, 1, 3)
std = np.array(model.data_preprocessor.std).reshape(1, 1, 3)
normed = (canvas.astype(np.float32) - mean) / std
tensor_in = torch.from_numpy(normed.transpose(2, 0, 1)[None]).float()

with torch.no_grad():
    raw_boxes, raw_scores = export_model(tensor_in)

score_thr = 0.3
raw_boxes0, raw_scores0 = raw_boxes[0], raw_scores[0]
max_scores, labels = raw_scores0.max(dim=1)
keep = max_scores > score_thr
boxes_k, scores_k, labels_k = raw_boxes0[keep], max_scores[keep], labels[keep]
final_boxes, final_scores, final_labels = [], [], []
for cls_id in labels_k.unique():
    cls_mask = labels_k == cls_id
    idx = tv_nms(boxes_k[cls_mask], scores_k[cls_mask], iou_threshold=0.5)
    final_boxes.append(boxes_k[cls_mask][idx])
    final_scores.append(scores_k[cls_mask][idx])
    final_labels.append(torch.full((len(idx),), cls_id))
our_boxes = torch.cat(final_boxes) if final_boxes else torch.empty(0, 4)
our_scores = torch.cat(final_scores) if final_scores else torch.empty(0)
our_labels = torch.cat(final_labels) if final_labels else torch.empty(0)
print(f'Our raw-export + manual NMS: {len(our_boxes)} detections')

with torch.no_grad():
    feats = model.extract_feat(tensor_in)
    cls_scores, bbox_preds = model.bbox_head(feats)
    meta = {'img_shape': (640, 640), 'scale_factor': (1.0, 1.0)}
    official = model.bbox_head.predict_by_feat(
        cls_scores, bbox_preds, batch_img_metas=[meta], cfg=model.test_cfg,
        rescale=False, with_nms=True)[0]
print(f'Official mmdetection inference: {len(official.bboxes)} detections')

from mmdet.datasets.coco import CocoDataset
COCO_CLASSES = CocoDataset.METAINFO['classes']

vis = canvas.copy()
for box, lbl, sc in zip(our_boxes, our_labels, our_scores):
    x1, y1, x2, y2 = box.int().tolist()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(vis, f'{COCO_CLASSES[int(lbl)]} {sc:.2f}', (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

vis2 = canvas.copy()
for box, lbl, sc in zip(official.bboxes, official.labels, official.scores):
    x1, y1, x2, y2 = box.int().tolist()
    cv2.rectangle(vis2, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(vis2, f'{COCO_CLASSES[int(lbl)]} {sc:.2f}', (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

combined = np.concatenate([vis, vis2], axis=1)
cv2.imwrite('comparison.png', combined)
print('Saved comparison.png (left = ours/red, right = official/green)')

# ---- export to ONNX ----
torch.onnx.export(
    export_model, dummy, 'rtmdet_small_raw.onnx',
    input_names=['image'], output_names=['boxes', 'scores'],
    opset_version=17, do_constant_folding=True,
)
onnx_model = onnx.load('rtmdet_small_raw.onnx')
onnx.checker.check_model(onnx_model)
print('ONNX model valid.')
print('Inputs:', [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in onnx_model.graph.input])
print('Outputs:', [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in onnx_model.graph.output])
print(f"File size: {os.path.getsize('rtmdet_small_raw.onnx') / 1e6:.1f} MB")

# ---- re-verify the actual exported ONNX file with onnxruntime ----
sess = ort.InferenceSession('rtmdet_small_raw.onnx')
ort_boxes, ort_scores = sess.run(None, {'image': tensor_in.numpy()})
diff = np.abs(ort_boxes - raw_boxes.numpy()).max()
print(f'Max diff PyTorch vs ONNX Runtime (should be ~0): {diff:.6f}')
print('DONE.')
