
import argparse
import csv
import importlib
import io
import json
import math
import random
import time
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from tqdm import tqdm
import timm

warnings.filterwarnings('ignore')


class Config:
    tn5000_root = r"<LOCAL_THYROID_ROOT>\TN5000_forReview"
    output_root = "tn5000_roi_runs_compare_5models_3seed"

    model_family = "custom"   # custom / timm
    backbone_name = "transxnet_t"
    backbone_module = "models.transxnetggg"
    backbone_func = "transxnet_t"
    backbone_out_dim = 1000

    input_size = 256
    num_classes = 2
    class_names = ('0', '1')

    use_roi_crop = True
    bbox_expand_ratio = 0.30
    min_crop_size = 64
    use_whole_image_fallback = True
    train_bbox_jitter_prob = 0.0
    train_bbox_jitter_center = 0.0
    train_bbox_jitter_scale = 0.0
    train_pred_bbox_csv = ""
    train_pred_bbox_prob = 0.0

    train_split = "train"
    val_split = "val"
    test_split = "test"

    batch_size = 8
    num_workers = 0
    num_epochs = 70
    learning_rate = 1.5e-4
    backbone_lr = 5e-5
    weight_decay = 1e-4
    dropout = 0.30
    label_smoothing = 0.00
    early_stopping_patience = 12
    early_stopping_min_delta = 5e-5
    seed = 17

    use_class_weight = True
    use_manual_class_weights = True
    manual_class_weights = (1.35, 0.85)
    save_by_metric = "auc"

    do_threshold_search = True
    threshold_start = 0.10
    threshold_end = 0.95
    threshold_step = 0.01
    threshold_selection_mode = "bal_acc"
    min_recall_1 = 0.90

    use_hflip_tta = True
    use_temperature_scaling = True
    ensemble_topk = 3
    save_all_improved_checkpoints = True

    lr_schedule = "cosine_floor"
    lr_warmup_epochs = 5
    lr_min_ratio = 0.25

    param_tag = "OURS_autoCW_exp030_s17"
    best_model_name = "best_model_tn5000_roi.pth"
    final_model_name = "last_model_tn5000_roi.pth"


VALID_IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
TRANSFORMER_SIZE_LOCKED = ("swin", "vit", "deit", "beit")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tn5000-root", default=Config.tn5000_root)
    p.add_argument("--output-root", default=Config.output_root)
    p.add_argument("--model-family", default=Config.model_family)
    p.add_argument("--backbone-name", default=Config.backbone_name)
    p.add_argument("--backbone-module", default=Config.backbone_module)
    p.add_argument("--backbone-func", default=Config.backbone_func)
    p.add_argument("--backbone-out-dim", type=int, default=Config.backbone_out_dim)
    p.add_argument("--param-tag", required=True)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--input-size", type=int, default=Config.input_size)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--num-workers", type=int, default=Config.num_workers)
    p.add_argument("--num-epochs", type=int, default=Config.num_epochs)
    p.add_argument("--head-lr", type=float, default=Config.learning_rate)
    p.add_argument("--backbone-lr", type=float, default=Config.backbone_lr)
    p.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    p.add_argument("--dropout", type=float, default=Config.dropout)
    p.add_argument("--label-smoothing", type=float, default=Config.label_smoothing)
    p.add_argument("--early-stopping-patience", type=int, default=Config.early_stopping_patience)
    p.add_argument("--bbox-expand-ratio", type=float, default=Config.bbox_expand_ratio)
    p.add_argument("--train-bbox-jitter-prob", type=float, default=Config.train_bbox_jitter_prob)
    p.add_argument("--train-bbox-jitter-center", type=float, default=Config.train_bbox_jitter_center)
    p.add_argument("--train-bbox-jitter-scale", type=float, default=Config.train_bbox_jitter_scale)
    p.add_argument("--train-pred-bbox-csv", default=Config.train_pred_bbox_csv)
    p.add_argument("--train-pred-bbox-prob", type=float, default=Config.train_pred_bbox_prob)
    p.add_argument("--save-by-metric", default=Config.save_by_metric)
    p.add_argument("--threshold-selection-mode", default=Config.threshold_selection_mode)
    p.add_argument("--lr-schedule", default=Config.lr_schedule)
    p.add_argument("--lr-warmup-epochs", type=int, default=Config.lr_warmup_epochs)
    p.add_argument("--lr-min-ratio", type=float, default=Config.lr_min_ratio)
    p.add_argument("--use-ema", type=int, default=1)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--use-class-weight", type=int, default=1)
    p.add_argument("--use-manual-class-weights", type=int, default=1)
    p.add_argument("--manual-class-weights", default="1.35,0.85")
    p.add_argument("--use-hflip-tta", type=int, default=1)
    p.add_argument("--use-temperature-scaling", type=int, default=1)
    p.add_argument("--ensemble-topk", type=int, default=3)
    return p.parse_args()


def apply_args(args):
    Config.tn5000_root = args.tn5000_root
    Config.output_root = args.output_root
    Config.model_family = args.model_family
    Config.backbone_name = args.backbone_name
    Config.backbone_module = args.backbone_module
    Config.backbone_func = args.backbone_func
    Config.backbone_out_dim = args.backbone_out_dim
    Config.param_tag = args.param_tag
    Config.seed = args.seed
    Config.input_size = args.input_size
    Config.batch_size = args.batch_size
    Config.num_workers = args.num_workers
    Config.num_epochs = args.num_epochs
    Config.learning_rate = args.head_lr
    Config.backbone_lr = args.backbone_lr
    Config.weight_decay = args.weight_decay
    Config.dropout = args.dropout
    Config.label_smoothing = args.label_smoothing
    Config.early_stopping_patience = args.early_stopping_patience
    Config.bbox_expand_ratio = args.bbox_expand_ratio
    Config.train_bbox_jitter_prob = args.train_bbox_jitter_prob
    Config.train_bbox_jitter_center = args.train_bbox_jitter_center
    Config.train_bbox_jitter_scale = args.train_bbox_jitter_scale
    Config.train_pred_bbox_csv = args.train_pred_bbox_csv
    Config.train_pred_bbox_prob = args.train_pred_bbox_prob
    Config.save_by_metric = args.save_by_metric
    Config.threshold_selection_mode = args.threshold_selection_mode
    Config.lr_schedule = args.lr_schedule
    Config.lr_warmup_epochs = args.lr_warmup_epochs
    Config.lr_min_ratio = args.lr_min_ratio
    Config.use_ema = bool(args.use_ema)
    Config.ema_decay = args.ema_decay
    Config.use_class_weight = bool(args.use_class_weight)
    Config.use_manual_class_weights = bool(args.use_manual_class_weights)
    Config.manual_class_weights = tuple(float(x) for x in str(args.manual_class_weights).split(','))
    Config.use_hflip_tta = bool(args.use_hflip_tta)
    Config.use_temperature_scaling = bool(args.use_temperature_scaling)
    Config.ensemble_topk = args.ensemble_topk


class AddGaussianNoise(nn.Module):
    def __init__(self, std=0.01):
        super().__init__()
        self.std = float(std)

    def forward(self, x):
        if self.std <= 0:
            return x
        noise = torch.randn_like(x) * self.std
        return torch.clamp(x + noise, 0.0, 1.0)


def set_seed(seed=17):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def silent_call(fn, *args, **kwargs):
    fake_out = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
        return fn(*args, **kwargs)


class TN5000ROIDataset(Dataset):
    """
    TN5000 fixed split + XML label + ROI crop
    - labels from object/name: 0/benign, 1/malignant
    - bbox from union of all object/bndbox
    """
    def __init__(self, root_dir, split="train", transform=None,
                 use_roi_crop=True, bbox_expand_ratio=0.30,
                 min_crop_size=64, use_whole_image_fallback=True,
                 bbox_jitter_prob=0.0, bbox_jitter_center=0.0, bbox_jitter_scale=0.0,
                 pred_bbox_csv="", pred_bbox_prob=0.0):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_roi_crop = use_roi_crop
        self.bbox_expand_ratio = bbox_expand_ratio
        self.min_crop_size = min_crop_size
        self.use_whole_image_fallback = use_whole_image_fallback
        self.bbox_jitter_prob = float(bbox_jitter_prob)
        self.bbox_jitter_center = float(bbox_jitter_center)
        self.bbox_jitter_scale = float(bbox_jitter_scale)
        self.pred_bbox_prob = float(pred_bbox_prob)
        self.pred_boxes = self._load_pred_boxes(pred_bbox_csv)

        self.image_dir = self.root_dir / 'JPEGImages'
        self.ann_dir = self.root_dir / 'Annotations'
        self.split_file = self.root_dir / 'ImageSets' / 'Main' / f'{split}.txt'

        if not self.image_dir.exists():
            raise FileNotFoundError("JPEGImages 不存在: %s" % self.image_dir)
        if not self.ann_dir.exists():
            raise FileNotFoundError("Annotations 不存在: %s" % self.ann_dir)
        if not self.split_file.exists():
            raise FileNotFoundError("split 文件不存在: %s" % self.split_file)

        self.samples = self._build_samples()
        self.label_counts = self._count_labels()
        if len(self.samples) == 0:
            raise RuntimeError("%s 集为空，请检查 TN5000 目录结构。" % split)
        print("[TN5000-ROI] split=%s, samples=%d, label_counts=%s" % (split, len(self.samples), self.label_counts))
        if self.bbox_jitter_prob > 0:
            print("[TN5000-ROI] bbox_jitter split=%s prob=%.3f center=%.3f scale=%.3f" %
                  (split, self.bbox_jitter_prob, self.bbox_jitter_center, self.bbox_jitter_scale))
        if self.pred_bbox_prob > 0 and self.pred_boxes:
            print("[TN5000-ROI] pred_bbox_mix split=%s prob=%.3f boxes=%d file=%s" %
                  (split, self.pred_bbox_prob, len(self.pred_boxes), pred_bbox_csv))

    def _load_pred_boxes(self, csv_path):
        csv_path = str(csv_path or "").strip()
        if not csv_path:
            return {}
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError("pred bbox csv not found: %s" % path)
        pred_boxes = {}
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                image_id = str(row.get("image_id", "")).strip()
                if not image_id:
                    continue
                if str(row.get("no_detection", "0")).strip() in ("1", "True", "true"):
                    continue
                box_text = str(row.get("pred_bbox", "")).strip()
                if box_text:
                    parts = [float(x) for x in box_text.replace(",", " ").split()]
                else:
                    keys = ("x1", "y1", "x2", "y2")
                    if not all(k in row and str(row[k]).strip() for k in keys):
                        continue
                    parts = [float(row[k]) for k in keys]
                if len(parts) != 4:
                    continue
                x1, y1, x2, y2 = parts
                if x2 > x1 and y2 > y1:
                    pred_boxes[image_id] = (x1, y1, x2, y2)
        return pred_boxes

    def _read_split_ids(self):
        ids = []
        with open(self.split_file, 'r', encoding='utf-8') as f:
            for line in f:
                x = line.strip()
                if x:
                    ids.append(x)
        return ids

    def _find_image_path(self, image_id):
        for suf in VALID_IMAGE_SUFFIXES:
            p = self.image_dir / (image_id + suf)
            if p.exists():
                return p
        raise FileNotFoundError("找不到图像文件: %s" % image_id)

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        labels = []
        boxes = []
        for obj in root.findall('object'):
            name_node = obj.find('name')
            if name_node is not None and name_node.text is not None:
                name_text = name_node.text.strip().lower()
                if name_text in ['0', 'benign']:
                    labels.append(0)
                elif name_text in ['1', 'malignant']:
                    labels.append(1)
                else:
                    raise ValueError("无法识别标签: %s, file=%s" % (name_text, xml_path))
            bnd = obj.find('bndbox')
            if bnd is not None:
                xmin = int(float(bnd.findtext('xmin', default='0')))
                ymin = int(float(bnd.findtext('ymin', default='0')))
                xmax = int(float(bnd.findtext('xmax', default='0')))
                ymax = int(float(bnd.findtext('ymax', default='0')))
                if xmax > xmin and ymax > ymin:
                    boxes.append((xmin, ymin, xmax, ymax))
        if len(labels) == 0:
            raise ValueError("XML 中没有有效标签: %s" % xml_path)
        label = max(labels)
        if len(boxes) == 0:
            return label, None
        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        return label, (min(xs1), min(ys1), max(xs2), max(ys2))

    def _build_samples(self):
        samples = []
        skipped = 0
        for image_id in self._read_split_ids():
            xml_path = self.ann_dir / (image_id + '.xml')
            if not xml_path.exists():
                skipped += 1
                continue
            try:
                img_path = self._find_image_path(image_id)
                label, bbox = self._parse_xml(xml_path)
                samples.append({
                    'image_id': image_id,
                    'image_path': str(img_path),
                    'xml_path': str(xml_path),
                    'label': int(label),
                    'bbox': bbox,
                })
            except Exception as e:
                skipped += 1
                print("[跳过] %s 解析失败: %s" % (image_id, e))
        if skipped > 0:
            print("[INFO] split=%s, skipped=%d" % (self.split, skipped))
        return samples

    def _count_labels(self):
        counter = defaultdict(int)
        for s in self.samples:
            counter[int(s['label'])] += 1
        return dict(counter)

    def __len__(self):
        return len(self.samples)

    def _expand_and_clip_box(self, box, width, height):
        x1, y1, x2, y2 = box
        bw = max(x2 - x1, self.min_crop_size)
        bh = max(y2 - y1, self.min_crop_size)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        new_w = bw * (1.0 + 2.0 * self.bbox_expand_ratio)
        new_h = bh * (1.0 + 2.0 * self.bbox_expand_ratio)
        nx1 = int(round(cx - new_w / 2.0))
        ny1 = int(round(cy - new_h / 2.0))
        nx2 = int(round(cx + new_w / 2.0))
        ny2 = int(round(cy + new_h / 2.0))
        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(width, nx2)
        ny2 = min(height, ny2)
        if nx2 <= nx1 or ny2 <= ny1:
            return None
        return (nx1, ny1, nx2, ny2)

    def _jitter_box(self, box, width, height):
        if self.bbox_jitter_prob <= 0 or random.random() >= self.bbox_jitter_prob:
            return box
        x1, y1, x2, y2 = box
        bw = max(float(x2 - x1), 1.0)
        bh = max(float(y2 - y1), 1.0)
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        if self.bbox_jitter_center > 0:
            cx += random.uniform(-self.bbox_jitter_center, self.bbox_jitter_center) * bw
            cy += random.uniform(-self.bbox_jitter_center, self.bbox_jitter_center) * bh
        if self.bbox_jitter_scale > 0:
            bw *= math.exp(random.uniform(-self.bbox_jitter_scale, self.bbox_jitter_scale))
            bh *= math.exp(random.uniform(-self.bbox_jitter_scale, self.bbox_jitter_scale))
        nx1 = max(0.0, cx - bw / 2.0)
        ny1 = max(0.0, cy - bh / 2.0)
        nx2 = min(float(width), cx + bw / 2.0)
        ny2 = min(float(height), cy + bh / 2.0)
        if nx2 <= nx1 or ny2 <= ny1:
            return box
        return (nx1, ny1, nx2, ny2)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image_path']).convert('RGB')
        if self.use_roi_crop and sample['bbox'] is not None:
            crop_source_box = sample['bbox']
            pred_box = self.pred_boxes.get(sample['image_id'])
            if pred_box is not None and self.pred_bbox_prob > 0 and random.random() < self.pred_bbox_prob:
                crop_source_box = pred_box
            crop_source_box = self._jitter_box(crop_source_box, img.width, img.height)
            crop_box = self._expand_and_clip_box(crop_source_box, img.width, img.height)
            if crop_box is not None:
                img = img.crop(crop_box)
            elif not self.use_whole_image_fallback:
                raise RuntimeError("无效裁剪框: %s" % sample['image_id'])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(sample['label']), sample['image_id'], sample['image_path']


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.input_size, Config.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(4),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.06, contrast=0.10, saturation=0.0, hue=0.0),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.20),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((Config.input_size, Config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def prepare_datasets(seed=17):
    set_seed(seed)
    train_transform, eval_transform = build_transforms()
    datasets = {
        'train': TN5000ROIDataset(Config.tn5000_root, Config.train_split, train_transform,
                                  use_roi_crop=Config.use_roi_crop,
                                  bbox_expand_ratio=Config.bbox_expand_ratio,
                                  min_crop_size=Config.min_crop_size,
                                  use_whole_image_fallback=Config.use_whole_image_fallback,
                                  bbox_jitter_prob=Config.train_bbox_jitter_prob,
                                  bbox_jitter_center=Config.train_bbox_jitter_center,
                                  bbox_jitter_scale=Config.train_bbox_jitter_scale,
                                  pred_bbox_csv=Config.train_pred_bbox_csv,
                                  pred_bbox_prob=Config.train_pred_bbox_prob),
        'val': TN5000ROIDataset(Config.tn5000_root, Config.val_split, eval_transform,
                                use_roi_crop=Config.use_roi_crop,
                                bbox_expand_ratio=Config.bbox_expand_ratio,
                                min_crop_size=Config.min_crop_size,
                                use_whole_image_fallback=Config.use_whole_image_fallback),
        'test': TN5000ROIDataset(Config.tn5000_root, Config.test_split, eval_transform,
                                 use_roi_crop=Config.use_roi_crop,
                                 bbox_expand_ratio=Config.bbox_expand_ratio,
                                 min_crop_size=Config.min_crop_size,
                                 use_whole_image_fallback=Config.use_whole_image_fallback),
    }
    return datasets


class UnifiedRoiClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model_family = str(Config.model_family).lower().strip()
        self.backbone_name = Config.backbone_name
        if self.model_family == 'timm':
            create_kwargs = dict(pretrained=True, num_classes=0, global_pool='avg')
            if any(k in Config.backbone_name.lower() for k in TRANSFORMER_SIZE_LOCKED):
                create_kwargs['img_size'] = Config.input_size
            self.backbone = timm.create_model(Config.backbone_name, **create_kwargs)
            feat_dim = self._infer_feat_dim()
            print('[INFO] inferred feature dim for %s = %d' % (Config.backbone_name, feat_dim))
        elif self.model_family == 'custom':
            module = importlib.import_module(Config.backbone_module)
            backbone_fn = getattr(module, Config.backbone_func)
            self.backbone = backbone_fn(num_classes=Config.backbone_out_dim, img_size=Config.input_size)
            feat_dim = int(Config.backbone_out_dim)
            print('[INFO] custom backbone output dim = %d' % feat_dim)
        else:
            raise ValueError('Unsupported model_family: %s' % Config.model_family)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(Config.dropout),
            nn.Linear(512, num_classes),
        )

    @torch.no_grad()
    def _infer_feat_dim(self):
        dummy = torch.zeros(1, 3, Config.input_size, Config.input_size)
        feats = self.extract_features(dummy)
        if feats.ndim != 2:
            raise RuntimeError('Unexpected feature shape: %s' % (tuple(feats.shape),))
        return int(feats.shape[1])

    def extract_features(self, x):
        if self.model_family == 'timm':
            feats = self.backbone(x)
        else:
            feats = silent_call(self.backbone, x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim == 4:
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        elif feats.ndim == 3:
            feats = feats.mean(dim=1)
        elif feats.ndim == 1:
            feats = feats.unsqueeze(0)
        return feats

    def forward(self, x):
        feats = self.extract_features(x)
        return self.head(feats)


def build_optimizer(model):
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.'):
            backbone_params.append(param)
        else:
            head_params.append(param)
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'base_lr': Config.backbone_lr, 'lr': Config.backbone_lr, 'weight_decay': Config.weight_decay, 'group_name': 'backbone'})
    if head_params:
        param_groups.append({'params': head_params, 'base_lr': Config.learning_rate, 'lr': Config.learning_rate, 'weight_decay': Config.weight_decay, 'group_name': 'head'})
    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print('[OPTIM] backbone_lr=%.2e, head_lr=%.2e' % (Config.backbone_lr, Config.learning_rate))
    print('[OPTIM] trainable backbone params=%d, head params=%d' % (n_backbone, n_head))
    return optim.AdamW(param_groups)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if k not in msd:
                continue
            src = msd[k].detach()
            if not torch.is_floating_point(v):
                v.copy_(src)
            else:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)


def create_result_dir():
    result_dir = Path(Config.output_root) / datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def build_class_weights(train_dataset, device):
    counts = [train_dataset.label_counts.get(i, 0) for i in range(Config.num_classes)]
    counts = np.array(counts, dtype=np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics_from_labels_probs(all_labels, all_probs, threshold=0.5):
    all_preds = (all_probs >= threshold).astype(int)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = np.array([0.0, 1.0]), np.array([0.0, 1.0]), float('nan')
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(Config.num_classes)))
    report = classification_report(all_labels, all_preds, target_names=Config.class_names, digits=4, zero_division=0)
    return {
        'acc': float(acc),
        'bal_acc': float(bal_acc),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'auc': float(roc_auc),
        'cm': cm,
        'report': report,
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs,
        'fpr': fpr,
        'tpr': tpr,
        'threshold': float(threshold),
    }


def scan_thresholds(y_true, y_prob, start, end, step):
    results = []
    thr = start
    while thr <= end + 1e-12:
        metrics = compute_metrics_from_labels_probs(y_true, y_prob, threshold=float(round(thr, 4)))
        cm = metrics['cm']
        tn, fp, fn, tp = cm.ravel()
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results.append({
            'threshold': float(round(thr, 4)),
            'bal_acc': float(metrics['bal_acc']),
            'f1_macro': float(metrics['f1_macro']),
            'acc': float(metrics['acc']),
            'auc': float(metrics['auc']),
            'recall_0': float(recall_0),
            'recall_1': float(recall_1),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        })
        thr += step
    return results


def choose_best_threshold(results):
    if Config.threshold_selection_mode == 'recall_constraint':
        candidates = [x for x in results if x['recall_1'] >= Config.min_recall_1]
        if candidates:
            candidates = sorted(candidates, key=lambda x: (x['bal_acc'], x['f1_macro'], -abs(x['threshold'] - 0.5)), reverse=True)
            return candidates[0]
    results = sorted(results, key=lambda x: (x['bal_acc'], x['f1_macro'], -abs(x['threshold'] - 0.5)), reverse=True)
    return results[0]


def save_threshold_scan_csv(results, out_path):
    fields = ['threshold', 'bal_acc', 'f1_macro', 'acc', 'auc', 'recall_0', 'recall_1', 'tn', 'fp', 'fn', 'tp']
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name='val', threshold=0.5):
    model.eval()
    losses, all_labels, all_probs, all_ids, all_paths = [], [], [], [], []
    with tqdm(loader, desc='Evaluating-%s' % split_name) as pbar:
        for inputs, labels, image_ids, image_paths in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            losses.append(loss.item())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_ids.extend(list(image_ids))
            all_paths.extend(list(image_paths))
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    metrics = compute_metrics_from_labels_probs(all_labels, all_probs, threshold=threshold)
    metrics['loss'] = float(np.mean(losses)) if losses else 0.0
    metrics['image_ids'] = all_ids
    metrics['image_paths'] = all_paths
    return metrics


def logits_to_probs(logits, temperature=1.0):
    x = torch.tensor(logits / max(float(temperature), 1e-6), dtype=torch.float32)
    probs = torch.softmax(x, dim=1)[:, 1].cpu().numpy()
    return probs


@torch.no_grad()
def collect_logits(model, loader, device, split_name='val', use_hflip_tta=False):
    model.eval()
    all_labels, all_logits, all_ids, all_paths = [], [], [], []
    with tqdm(loader, desc='CollectLogits-%s' % split_name) as pbar:
        for inputs, labels, image_ids, image_paths in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            if use_hflip_tta:
                logits_flip = model(torch.flip(inputs, dims=[3]))
                logits = 0.5 * (logits + logits_flip)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_ids.extend(list(image_ids))
            all_paths.extend(list(image_paths))
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_logits = torch.cat(all_logits, dim=0).numpy()
    return all_labels, all_logits, all_ids, all_paths


def collect_ensemble_logits(model, ckpt_paths, loader, device, split_name='val', use_hflip_tta=False):
    logits_sum = None
    labels_ref, ids_ref, paths_ref = None, None, None
    used_ckpts = []
    for ckpt_path in ckpt_paths:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        labels, logits, ids, paths = collect_logits(model, loader, device, split_name='%s-%s' % (split_name, ckpt_path.stem), use_hflip_tta=use_hflip_tta)
        if labels_ref is None:
            labels_ref, ids_ref, paths_ref = labels, ids, paths
        else:
            if not np.array_equal(labels_ref, labels):
                raise RuntimeError('标签顺序不一致，无法集成: %s' % ckpt_path)
            if ids_ref != ids:
                raise RuntimeError('image_id 顺序不一致，无法集成: %s' % ckpt_path)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)
        used_ckpts.append(str(ckpt_path))
    if logits_sum is None or labels_ref is None:
        raise RuntimeError('没有可用 checkpoint 用于集成推理。')
    logits_mean = logits_sum / len(used_ckpts)
    return labels_ref, logits_mean, ids_ref, paths_ref, used_ckpts


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        temperature = self.log_temperature.exp().clamp(min=1e-3, max=100.0)
        return logits / temperature


def fit_temperature_on_val(logits_np, labels_np, device):
    logits = torch.tensor(logits_np, dtype=torch.float32, device=device)
    labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    scaler = TemperatureScaler().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([scaler.log_temperature], lr=0.1, max_iter=50, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(scaler.log_temperature.exp().detach().cpu().item())
    return max(temperature, 1e-3)


def compute_nll_from_logits(logits_np, labels_np, temperature=1.0):
    logits = torch.tensor(logits_np / max(float(temperature), 1e-6), dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    return float(nn.CrossEntropyLoss()(logits, labels).item())


def metric_for_selection(metrics):
    key = Config.save_by_metric
    if key not in metrics:
        raise KeyError('save_by_metric=%s 不在 metrics 中' % key)
    return float(metrics[key])


def get_epoch_lr_for_group(param_group, epoch):
    base_lr = float(param_group.get('base_lr', Config.learning_rate))
    if Config.lr_schedule == 'cosine_floor':
        warmup = max(int(Config.lr_warmup_epochs), 0)
        total = max(Config.num_epochs, 1)
        floor = float(Config.lr_min_ratio)
        if epoch < warmup and warmup > 0:
            alpha = float(epoch + 1) / float(warmup)
            return base_lr * alpha
        remain = max(total - warmup, 1)
        t = min(max(epoch - warmup, 0), remain - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / max(remain - 1, 1)))
        ratio = floor + (1.0 - floor) * cosine
        return base_lr * ratio
    total = max(Config.num_epochs - 1, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total))


def train_model(model, dataloaders, criterion, optimizer, result_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    history = defaultdict(list)
    best_score = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_path = result_dir / Config.best_model_name
    last_path = result_dir / Config.final_model_name
    improved_ckpts = []
    ema = ModelEMA(model, decay=Config.ema_decay) if Config.use_ema else None

    for epoch in range(Config.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_epoch_lr_for_group(param_group, epoch)
        head_lr = max(pg['lr'] for pg in optimizer.param_groups) if optimizer.param_groups else Config.learning_rate

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        with tqdm(dataloaders['train'], desc='Train Epoch %d/%d' % (epoch + 1, Config.num_epochs)) as pbar:
            for inputs, labels, _, _ in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                train_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
                pbar.set_postfix({'Loss': '%.4f' % loss.item(), 'Acc': '%.4f' % (correct / max(total, 1)), 'LR': '%.2e' % head_lr})

        train_acc = correct / max(total, 1)
        train_loss_avg = train_loss / max(len(dataloaders['train']), 1)
        eval_model = ema.module if ema is not None else model
        val_metrics = evaluate(eval_model, dataloaders['val'], criterion, device, split_name='val')
        score = metric_for_selection(val_metrics)

        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_bal_acc'].append(val_metrics['bal_acc'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['val_auc'].append(val_metrics['auc'])
        history['lr'].append(head_lr)

        print("Epoch %03d | lr=%.2e train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_bal_acc=%.4f val_f1_macro=%.4f val_auc=%.4f" %
              (epoch + 1, head_lr, train_loss_avg, train_acc, val_metrics['loss'], val_metrics['acc'], val_metrics['bal_acc'], val_metrics['f1_macro'], val_metrics['auc']))

        if score > best_score + Config.early_stopping_min_delta:
            best_score = score
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(eval_model.state_dict(), best_path)
            if Config.save_all_improved_checkpoints:
                ckpt_path = result_dir / ("epoch%03d_%s_%.4f.pth" % (best_epoch, Config.save_by_metric, best_score))
                torch.save(eval_model.state_dict(), ckpt_path)
                improved_ckpts.append({'epoch': int(best_epoch), 'score': float(best_score), 'path': str(ckpt_path)})
                print('[SAVE] improved checkpoint -> %s' % ckpt_path)
            print('[SAVE] New best model saved to: %s | %s=%.4f | epoch=%d' % (best_path, Config.save_by_metric, best_score, best_epoch))
        else:
            epochs_no_improve += 1
            print('[EARLY_STOP_CHECK] no improvement for %d/%d epochs | best_%s=%.4f @ epoch %d' %
                  (epochs_no_improve, Config.early_stopping_patience, Config.save_by_metric, best_score, best_epoch))
            if epochs_no_improve >= Config.early_stopping_patience:
                print('[EARLY_STOP] Triggered at epoch %d. Best epoch=%d, best_%s=%.4f' %
                      (epoch + 1, best_epoch, Config.save_by_metric, best_score))
                break

    torch.save(model.state_dict(), last_path)
    print('[SAVE] Last model saved to: %s' % last_path)
    return history, best_path, last_path, best_epoch, best_score, improved_ckpts


@torch.no_grad()
def measure_inference_time(model, test_loader, repetitions=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    first_batch = next(iter(test_loader))
    sample_batch = first_batch[0].to(device)
    single_image = sample_batch[:1]
    for _ in range(10):
        _ = model(single_image)
        _ = model(sample_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(repetitions):
        _ = model(single_image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    single_time = (time.time() - start) / repetitions * 1000
    start = time.time()
    for _ in range(repetitions):
        _ = model(sample_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    batch_time = (time.time() - start) / repetitions * 1000
    fps = sample_batch.size(0) / (batch_time / 1000.0)
    return single_time, batch_time, fps


def save_curves(history, result_dir):
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.plot(history['val_bal_acc'], label='Val Bal Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy Curves'); plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curves'); plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(history['val_f1_macro'], label='Val Macro F1')
    plt.plot(history['val_auc'], label='Val AUC')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Validation Metrics'); plt.legend()
    plt.tight_layout()
    out = result_dir / 'training_curves.png'
    plt.savefig(out, dpi=250, bbox_inches='tight')
    plt.close(fig)
    return out


def save_roc_points(labels, probs, out_path):
    try:
        fpr, tpr, thr = roc_curve(labels, probs)
    except Exception:
        fpr, tpr, thr = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.nan, np.nan])
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['fpr', 'tpr', 'threshold'])
        for a, b, c in zip(fpr, tpr, thr):
            writer.writerow([float(a), float(b), float(c)])


def save_confusion_matrix_csv(cm, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['', Config.class_names[0], Config.class_names[1]])
        for idx, row in enumerate(cm.tolist()):
            writer.writerow([Config.class_names[idx], row[0], row[1]])


def save_confusion_matrix_png(cm, title, out_path):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(len(Config.class_names)), Config.class_names)
    plt.yticks(range(len(Config.class_names)), Config.class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def save_predictions_csv(image_ids, image_paths, labels, preds, probs, threshold, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'image_path', 'true_label', 'pred_label', 'prob_class0', 'prob_class1', 'threshold', 'is_wrong', 'wrong_conf', 'margin_from_05'])
        for image_id, image_path, y, p, prob1 in zip(image_ids, image_paths, labels, preds, probs):
            prob0 = 1.0 - float(prob1)
            wrong = int(int(y) != int(p))
            wrong_conf = float(prob1) if int(p) == 1 else float(prob0)
            margin = abs(float(prob1) - 0.5)
            writer.writerow([image_id, image_path, int(y), int(p), prob0, float(prob1), float(threshold), wrong, wrong_conf, margin])


def save_hardcases_csv(pred_csv, out_path):
    with open(pred_csv, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    hard_rows = [r for r in rows if int(r['is_wrong']) == 1]
    hard_rows = sorted(hard_rows, key=lambda r: (float(r['wrong_conf']), float(r['margin_from_05'])), reverse=True)
    if not rows:
        return
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(hard_rows)


def save_summary(history, val_metrics, test_metrics, single_ms, batch_ms, fps, result_dir, best_epoch, best_score, used_ckpts, temperature, val_nll_before, val_nll_after, best_threshold, best_thr_info):
    txt_path = result_dir / 'summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('TN5000 ROI Classification Summary\n')
        f.write('=' * 70 + '\n')
        f.write('model_family: %s\n' % Config.model_family)
        f.write('backbone_name: %s\n' % Config.backbone_name)
        f.write('backbone_module: %s\n' % Config.backbone_module)
        f.write('backbone_func: %s\n' % Config.backbone_func)
        f.write('input_size: %d\n' % Config.input_size)
        f.write('use_roi_crop: %s\n' % Config.use_roi_crop)
        f.write('bbox_expand: %.4f\n' % Config.bbox_expand_ratio)
        f.write('train_bbox_jitter_prob: %.6f\n' % Config.train_bbox_jitter_prob)
        f.write('train_bbox_jitter_center: %.6f\n' % Config.train_bbox_jitter_center)
        f.write('train_bbox_jitter_scale: %.6f\n' % Config.train_bbox_jitter_scale)
        f.write('train_pred_bbox_csv: %s\n' % Config.train_pred_bbox_csv)
        f.write('train_pred_bbox_prob: %.6f\n' % Config.train_pred_bbox_prob)
        f.write('batch_size: %d\n' % Config.batch_size)
        f.write('num_epochs: %d\n' % Config.num_epochs)
        f.write('weight_decay: %.6f\n' % Config.weight_decay)
        f.write('label_smoothing: %.6f\n' % Config.label_smoothing)
        f.write('early_stop_pat: %d\n' % Config.early_stopping_patience)
        f.write('save_by_metric: %s\n' % Config.save_by_metric)
        f.write('train_split: %s\n' % Config.train_split)
        f.write('val_split: %s\n' % Config.val_split)
        f.write('test_split: %s\n' % Config.test_split)
        f.write('seed: %d\n\n' % Config.seed)

        f.write('Validation Metrics\n')
        f.write('-' * 70 + '\n')
        f.write('loss=%.6f\n' % val_metrics['loss'])
        f.write('acc=%.6f\n' % val_metrics['acc'])
        f.write('bal_acc=%.6f\n' % val_metrics['bal_acc'])
        f.write('f1_macro=%.6f\n' % val_metrics['f1_macro'])
        f.write('auc=%.6f\n\n' % val_metrics['auc'])
        f.write(val_metrics['report'] + '\n\n')

        f.write('Test Metrics\n')
        f.write('-' * 70 + '\n')
        f.write('loss=%.6f\n' % test_metrics['loss'])
        f.write('acc=%.6f\n' % test_metrics['acc'])
        f.write('bal_acc=%.6f\n' % test_metrics['bal_acc'])
        f.write('f1_macro=%.6f\n' % test_metrics['f1_macro'])
        f.write('auc=%.6f\n\n' % test_metrics['auc'])
        f.write(test_metrics['report'] + '\n\n')

        f.write('Runtime\n')
        f.write('-' * 70 + '\n')
        f.write('single_image_ms=%.6f\n' % single_ms)
        f.write('batch_ms=%.6f\n' % batch_ms)
        f.write('fps=%.6f\n\n' % fps)

        inferred_best_epoch = int(np.argmax(history['val_auc'])) + 1 if len(history['val_auc']) > 0 else -1
        f.write('best_epoch_by_runtime=%d\n' % best_epoch)
        f.write('best_score_by_runtime=%.6f\n' % best_score)
        f.write('best_epoch_by_history_auc=%d\n' % inferred_best_epoch)
        f.write('param_tag: %s\n' % Config.param_tag)
        f.write('backbone_lr: %s\n' % Config.backbone_lr)
        f.write('head_lr: %s\n' % Config.learning_rate)
        f.write('use_class_weight: %s\n' % Config.use_class_weight)
        f.write('use_manual_class_weights: %s\n' % Config.use_manual_class_weights)
        f.write('manual_class_weights: %s\n' % list(Config.manual_class_weights))
        f.write('use_ema: %s\n' % Config.use_ema)
        f.write('ema_decay: %s\n' % Config.ema_decay)
        f.write('use_hflip_tta: %s\n' % Config.use_hflip_tta)
        f.write('use_temperature_scaling: %s\n' % Config.use_temperature_scaling)
        f.write('ensemble_topk_requested: %d\n' % Config.ensemble_topk)
        f.write('ensemble_ckpts_used: %s\n' % used_ckpts)
        f.write('temperature: %.6f\n' % temperature)
        f.write('val_nll_before_temp: %.6f\n' % val_nll_before)
        f.write('val_nll_after_temp: %.6f\n' % val_nll_after)
        f.write('\nThreshold Selection\n')
        f.write('=' * 60 + '\n')
        f.write('Selected threshold: %.4f\n' % best_threshold)
        f.write('Selection mode: %s\n' % Config.threshold_selection_mode)
        if best_thr_info is not None:
            for k, v in best_thr_info.items():
                f.write('%s: %s\n' % (k, v))
    return txt_path


def build_metadata_dict(result_dir, class_weights):
    return {
        'tn5000_root': Config.tn5000_root,
        'output_root': Config.output_root,
        'model_family': Config.model_family,
        'backbone_name': Config.backbone_name,
        'backbone_module': Config.backbone_module,
        'backbone_func': Config.backbone_func,
        'backbone_out_dim': Config.backbone_out_dim,
        'input_size': Config.input_size,
        'num_classes': Config.num_classes,
        'class_names': list(Config.class_names),
        'use_roi_crop': Config.use_roi_crop,
        'bbox_expand_ratio': Config.bbox_expand_ratio,
        'train_bbox_jitter_prob': Config.train_bbox_jitter_prob,
        'train_bbox_jitter_center': Config.train_bbox_jitter_center,
        'train_bbox_jitter_scale': Config.train_bbox_jitter_scale,
        'train_pred_bbox_csv': Config.train_pred_bbox_csv,
        'train_pred_bbox_prob': Config.train_pred_bbox_prob,
        'min_crop_size': Config.min_crop_size,
        'use_whole_image_fallback': Config.use_whole_image_fallback,
        'train_split': Config.train_split,
        'val_split': Config.val_split,
        'test_split': Config.test_split,
        'batch_size': Config.batch_size,
        'num_workers': Config.num_workers,
        'num_epochs': Config.num_epochs,
        'learning_rate': Config.learning_rate,
        'backbone_lr': Config.backbone_lr,
        'weight_decay': Config.weight_decay,
        'dropout': Config.dropout,
        'label_smoothing': Config.label_smoothing,
        'early_stopping_patience': Config.early_stopping_patience,
        'early_stopping_min_delta': Config.early_stopping_min_delta,
        'seed': Config.seed,
        'use_class_weight': Config.use_class_weight,
        'use_manual_class_weights': Config.use_manual_class_weights,
        'manual_class_weights': list(Config.manual_class_weights),
        'save_by_metric': Config.save_by_metric,
        'do_threshold_search': Config.do_threshold_search,
        'threshold_start': Config.threshold_start,
        'threshold_end': Config.threshold_end,
        'threshold_step': Config.threshold_step,
        'threshold_selection_mode': Config.threshold_selection_mode,
        'min_recall_1': Config.min_recall_1,
        'use_hflip_tta': Config.use_hflip_tta,
        'use_temperature_scaling': Config.use_temperature_scaling,
        'ensemble_topk': Config.ensemble_topk,
        'save_all_improved_checkpoints': Config.save_all_improved_checkpoints,
        'lr_schedule': Config.lr_schedule,
        'lr_warmup_epochs': Config.lr_warmup_epochs,
        'lr_min_ratio': Config.lr_min_ratio,
        'param_tag': Config.param_tag,
        'best_model_name': Config.best_model_name,
        'final_model_name': Config.final_model_name,
        'result_dir': str(result_dir),
        'class_weights': class_weights.detach().cpu().numpy().tolist() if class_weights is not None else None,
    }


def main():
    args = parse_args()
    apply_args(args)
    set_seed(Config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_dir = create_result_dir()

    print('=' * 80)
    print('TN5000 ROI Training Script (Unified 3-seed)')
    print('=' * 80)
    print('[INFO] result_dir      = %s' % result_dir)
    print('[INFO] tn5000_root     = %s' % Config.tn5000_root)
    print('[INFO] model_family    = %s' % Config.model_family)
    print('[INFO] backbone_name   = %s' % Config.backbone_name)
    print('[INFO] backbone_module = %s' % Config.backbone_module)
    print('[INFO] backbone_func   = %s' % Config.backbone_func)
    print('[INFO] use_roi_crop    = %s' % Config.use_roi_crop)
    print('[INFO] bbox_expand     = %s' % Config.bbox_expand_ratio)
    print('[INFO] train_bbox_jit = p=%.3f center=%.3f scale=%.3f' %
          (Config.train_bbox_jitter_prob, Config.train_bbox_jitter_center, Config.train_bbox_jitter_scale))
    print('[INFO] train_pred_box = p=%.3f csv=%s' % (Config.train_pred_bbox_prob, Config.train_pred_bbox_csv))
    print('[INFO] device          = %s' % device)
    print('[INFO] num_epochs      = %d' % Config.num_epochs)
    print('[INFO] head_lr         = %s' % Config.learning_rate)
    print('[INFO] backbone_lr     = %s' % Config.backbone_lr)
    print('[INFO] weight_decay    = %s' % Config.weight_decay)
    print('[INFO] label_smoothing = %s' % Config.label_smoothing)
    print('[INFO] early_stop_pat  = %d' % Config.early_stopping_patience)
    print('[INFO] save_by_metric  = %s' % Config.save_by_metric)
    print('[INFO] thr_mode        = %s' % Config.threshold_selection_mode)
    print('[INFO] lr_schedule     = %s' % Config.lr_schedule)
    print('[INFO] lr_warmup_ep    = %s' % Config.lr_warmup_epochs)
    print('[INFO] lr_min_ratio    = %s' % Config.lr_min_ratio)
    print('[INFO] param_tag       = %s' % Config.param_tag)
    print('[INFO] seed            = %s' % Config.seed)
    print('=' * 80)

    datasets = prepare_datasets(seed=Config.seed)
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=Config.batch_size, shuffle=True,
                            num_workers=Config.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True),
        'val': DataLoader(datasets['val'], batch_size=Config.batch_size, shuffle=False,
                          num_workers=Config.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False),
        'test': DataLoader(datasets['test'], batch_size=Config.batch_size, shuffle=False,
                           num_workers=Config.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False),
    }

    model = UnifiedRoiClassifier(num_classes=Config.num_classes).to(device)

    class_weights = None
    if Config.use_class_weight:
        if Config.use_manual_class_weights:
            class_weights = torch.tensor(Config.manual_class_weights, dtype=torch.float32, device=device)
        else:
            class_weights = build_class_weights(datasets['train'], device)
        print('[INFO] class_weights = %s' % class_weights.detach().cpu().numpy().tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.label_smoothing)
    optimizer = build_optimizer(model)

    single_ms, batch_ms, fps = measure_inference_time(model, dataloaders['test'], repetitions=20)
    print('[INFO] single_image_ms = %.4f' % single_ms)
    print('[INFO] batch_ms        = %.4f' % batch_ms)
    print('[INFO] fps             = %.4f' % fps)

    history, best_path, _, best_epoch, best_score, improved_ckpts = train_model(model, dataloaders, criterion, optimizer, result_dir)
    curves_path = save_curves(history, result_dir)
    print('[SAVE] curves -> %s' % curves_path)

    candidate_ckpts = sorted(improved_ckpts, key=lambda x: (x['score'], x['epoch']), reverse=True)
    candidate_ckpt_paths = [x['path'] for x in candidate_ckpts[:Config.ensemble_topk]]
    if not candidate_ckpt_paths:
        candidate_ckpt_paths = [str(best_path)]
    if len(candidate_ckpt_paths) < Config.ensemble_topk:
        print('[ENSEMBLE] 可用 improved checkpoints=%d < ensemble_topk=%d，本次将退化为单/少量 checkpoint 推理。' % (len(candidate_ckpt_paths), Config.ensemble_topk))
    print('[ENSEMBLE] checkpoint(s) used: %s' % candidate_ckpt_paths)

    val_labels, val_logits, val_ids, val_paths, used_ckpts = collect_ensemble_logits(model, candidate_ckpt_paths, dataloaders['val'], device, split_name='val', use_hflip_tta=Config.use_hflip_tta)
    test_labels, test_logits, test_ids, test_paths, _ = collect_ensemble_logits(model, used_ckpts, dataloaders['test'], device, split_name='test', use_hflip_tta=Config.use_hflip_tta)

    temperature = 1.0
    val_nll_before = compute_nll_from_logits(val_logits, val_labels, temperature=1.0)
    val_nll_after = val_nll_before
    if Config.use_temperature_scaling:
        temperature = fit_temperature_on_val(val_logits, val_labels, device)
        val_nll_after = compute_nll_from_logits(val_logits, val_labels, temperature=temperature)
        print('[TEMP] fitted temperature = %.6f | val_nll_before=%.6f | val_nll_after=%.6f' % (temperature, val_nll_before, val_nll_after))
    else:
        print('[TEMP] disabled, using temperature = %.6f' % temperature)

    val_probs = logits_to_probs(val_logits, temperature=temperature)
    test_probs = logits_to_probs(test_logits, temperature=temperature)

    best_threshold = 0.5
    best_thr_info = None
    if Config.do_threshold_search:
        threshold_results = scan_thresholds(val_labels, val_probs, start=Config.threshold_start, end=Config.threshold_end, step=Config.threshold_step)
        save_threshold_scan_csv(threshold_results, result_dir / 'val_threshold_scan.csv')
        best_thr_info = choose_best_threshold(threshold_results)
        best_threshold = float(best_thr_info['threshold'])
        print('[THRESHOLD] selected threshold = %.4f | mode=%s | info=%s' % (best_threshold, Config.threshold_selection_mode, best_thr_info))
    else:
        print('[THRESHOLD] threshold search disabled, using threshold = %.4f' % best_threshold)

    val_metrics = compute_metrics_from_labels_probs(val_labels, val_probs, threshold=best_threshold)
    test_metrics = compute_metrics_from_labels_probs(test_labels, test_probs, threshold=best_threshold)
    val_metrics['loss'] = float('nan')
    test_metrics['loss'] = float('nan')
    val_metrics['image_ids'] = val_ids
    val_metrics['image_paths'] = val_paths
    test_metrics['image_ids'] = test_ids
    test_metrics['image_paths'] = test_paths

    save_predictions_csv(val_ids, val_paths, val_metrics['labels'], val_metrics['preds'], val_metrics['probs'], best_threshold, result_dir / 'val_predictions.csv')
    save_predictions_csv(test_ids, test_paths, test_metrics['labels'], test_metrics['preds'], test_metrics['probs'], best_threshold, result_dir / 'test_predictions.csv')
    save_hardcases_csv(result_dir / 'val_predictions.csv', result_dir / 'val_hardcases.csv')
    save_hardcases_csv(result_dir / 'test_predictions.csv', result_dir / 'test_hardcases.csv')
    save_roc_points(val_metrics['labels'], val_metrics['probs'], result_dir / 'val_roc_points.csv')
    save_roc_points(test_metrics['labels'], test_metrics['probs'], result_dir / 'test_roc_points.csv')
    save_confusion_matrix_csv(val_metrics['cm'], result_dir / 'val_confusion_matrix.csv')
    save_confusion_matrix_csv(test_metrics['cm'], result_dir / 'test_confusion_matrix.csv')
    save_confusion_matrix_png(val_metrics['cm'], 'Val CM | Thr=%.2f AUC=%.4f' % (best_threshold, val_metrics["auc"]), result_dir / 'val_confusion_matrix.png')
    save_confusion_matrix_png(test_metrics['cm'], 'Test CM | Thr=%.2f AUC=%.4f' % (best_threshold, test_metrics["auc"]), result_dir / 'test_confusion_matrix.png')
    summary_path = save_summary(history, val_metrics, test_metrics, single_ms, batch_ms, fps, result_dir, best_epoch, best_score, used_ckpts, temperature, val_nll_before, val_nll_after, best_threshold, best_thr_info)

    meta = build_metadata_dict(result_dir, class_weights)
    with open(result_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 80)
    print('Final Validation Metrics')
    print('=' * 80)
    print('threshold=%.4f' % best_threshold)
    print(val_metrics['report'])
    print('Final Test Metrics')
    print('=' * 80)
    print(test_metrics['report'])
    print('[SAVE] summary -> %s' % summary_path)
    print('[DONE] all outputs saved under: %s' % result_dir)


if __name__ == '__main__':
    main()
