from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import math
import os
import random
import time
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict
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


@dataclass
class Config:
    data_root: str = r"<LOCAL_THYROID_ROOT>\aul\aul_voc_roi_v1"
    output_root: str = "aul_roi_runs_binary"

    model_family: str = "custom"   # custom / timm
    backbone_name: str = "transxnet_t"
    backbone_module: str = "models.transxnetggg"
    backbone_func: str = "transxnet_t"
    backbone_out_dim: int = 1000     # custom backbone output dim before classifier head

    input_size: int = 256
    num_classes: int = 2
    class_names: Tuple[str, str] = ('benign', 'malignant')

    use_roi_crop: bool = True
    bbox_expand_ratio: float = 0.30
    min_crop_size: int = 64
    use_whole_image_fallback: bool = True

    train_split: str = "fold0_train"
    val_split: str = "fold0_val"
    test_split: str = "test"
    fold_idx: int = 0

    batch_size: int = 8
    num_workers: int = 0
    num_epochs: int = 200
    learning_rate: float = 1.5e-4
    backbone_lr: float = 5e-5
    weight_decay: float = 1e-4
    dropout: float = 0.30
    label_smoothing: float = 0.00
    early_stopping_patience: int = 36
    early_stopping_min_delta: float = 5e-5
    seed: int = 17

    use_class_weight: bool = True
    use_manual_class_weights: bool = False
    manual_class_weights: Tuple[float, float] = (1.0, 1.0)
    save_by_metric: str = "auc"

    do_threshold_search: bool = True
    threshold_start: float = 0.10
    threshold_end: float = 0.95
    threshold_step: float = 0.01
    threshold_selection_mode: str = "bal_acc"
    min_recall_1: float = 0.90

    use_hflip_tta: bool = True
    use_temperature_scaling: bool = True
    ensemble_topk: int = 3
    save_all_improved_checkpoints: bool = True

    lr_schedule: str = "cosine_floor"
    lr_warmup_epochs: int = 5
    lr_min_ratio: float = 0.25

    param_tag: str = "AUL_BIN_autoCW_exp030_f0"
    best_model_name: str = "best_model_aul_roi_bin.pth"
    final_model_name: str = "last_model_aul_roi_bin.pth"


VALID_IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
TRANSFORMER_SIZE_LOCKED = ("swin", "vit", "deit", "beit")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=Config.data_root)
    p.add_argument("--output-root", default=Config.output_root)
    p.add_argument("--model-family", default=Config.model_family)
    p.add_argument("--backbone-name", default=Config.backbone_name)
    p.add_argument("--backbone-module", default=Config.backbone_module)
    p.add_argument("--backbone-func", default=Config.backbone_func)
    p.add_argument("--backbone-out-dim", type=int, default=Config.backbone_out_dim)
    p.add_argument("--param-tag", required=True)
    p.add_argument("--fold-idx", type=int, required=True)
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
    p.add_argument("--save-by-metric", default=Config.save_by_metric)
    p.add_argument("--threshold-selection-mode", default=Config.threshold_selection_mode)
    p.add_argument("--lr-schedule", default=Config.lr_schedule)
    p.add_argument("--lr-warmup-epochs", type=int, default=Config.lr_warmup_epochs)
    p.add_argument("--lr-min-ratio", type=float, default=Config.lr_min_ratio)
    p.add_argument("--use-ema", type=int, default=1)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--use-class-weight", type=int, default=1)
    p.add_argument("--use-manual-class-weights", type=int, default=0)
    p.add_argument("--manual-class-weights", default="1.0,1.0")
    p.add_argument("--use-hflip-tta", type=int, default=1)
    p.add_argument("--use-temperature-scaling", type=int, default=1)
    p.add_argument("--ensemble-topk", type=int, default=3)
    return p.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    Config.data_root = args.data_root
    Config.output_root = args.output_root
    Config.model_family = args.model_family
    Config.backbone_name = args.backbone_name
    Config.backbone_module = args.backbone_module
    Config.backbone_func = args.backbone_func
    Config.backbone_out_dim = args.backbone_out_dim
    Config.param_tag = args.param_tag
    Config.fold_idx = args.fold_idx
    Config.train_split = f"fold{args.fold_idx}_train"
    Config.val_split = f"fold{args.fold_idx}_val"
    Config.test_split = "test"
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


def set_seed(seed: int = 17) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def silent_call(fn, *args, **kwargs):
    fake_out = io.StringIO()
    with torch.no_grad():
        import contextlib
        with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
            return fn(*args, **kwargs)


class AULBinaryVOCRoiDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None,
                 use_roi_crop: bool = True,
                 bbox_expand_ratio: float = 0.30,
                 min_crop_size: int = 64,
                 use_whole_image_fallback: bool = True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_roi_crop = use_roi_crop
        self.bbox_expand_ratio = bbox_expand_ratio
        self.min_crop_size = min_crop_size
        self.use_whole_image_fallback = use_whole_image_fallback

        self.image_dir = self.root_dir / 'JPEGImages'
        self.ann_dir = self.root_dir / 'Annotations'
        self.split_file = self.root_dir / 'ImageSets' / 'Main' / f'{split}.txt'
        self.label_manifest = self.root_dir / 'manifests' / 'label_manifest.csv'

        if not self.image_dir.exists():
            raise FileNotFoundError(f"JPEGImages 不存在: {self.image_dir}")
        if not self.ann_dir.exists():
            raise FileNotFoundError(f"Annotations 不存在: {self.ann_dir}")
        if not self.split_file.exists():
            raise FileNotFoundError(f"split 文件不存在: {self.split_file}")
        if not self.label_manifest.exists():
            raise FileNotFoundError(f"label_manifest 不存在: {self.label_manifest}")

        self.label_map = self._load_label_manifest()
        self.samples = self._build_samples()
        self.label_counts = self._count_labels()
        if len(self.samples) == 0:
            raise RuntimeError(f"{split} 集为空，请检查数据目录。")
        print(f"[AUL-BIN]    split={split}, samples={len(self.samples)}, label_counts={self.label_counts}")

    def _load_label_manifest(self) -> Dict[str, int]:
        """Support both old VOC builders (new_stem/new_filename) and AUL builder (image_id/filename)."""
        label_map: Dict[str, int] = {}
        with open(self.label_manifest, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row['label'])
                candidates = [
                    str(row.get('new_stem', '')).strip(),
                    str(row.get('new_filename', '')).strip(),
                    str(row.get('image_id', '')).strip(),
                    str(row.get('filename', '')).strip(),
                    str(row.get('xml_filename', '')).strip(),
                ]
                for key in candidates:
                    if not key:
                        continue
                    label_map[key] = label
                    label_map[Path(key).stem] = label
        return label_map

    def _read_split_ids(self) -> List[str]:
        ids = []
        with open(self.split_file, 'r', encoding='utf-8') as f:
            for line in f:
                x = line.strip()
                if x:
                    ids.append(x)
        return ids

    def _find_image_path(self, image_id: str) -> Path:
        for suf in VALID_IMAGE_SUFFIXES:
            p = self.image_dir / f'{image_id}{suf}'
            if p.exists():
                return p
        raise FileNotFoundError(f"找不到图像文件: {image_id}")

    def _parse_xml_bbox(self, xml_path: Path) -> Optional[Tuple[int, int, int, int]]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            bnd = obj.find('bndbox')
            if bnd is None:
                continue
            xmin = int(float(bnd.findtext('xmin', default='0')))
            ymin = int(float(bnd.findtext('ymin', default='0')))
            xmax = int(float(bnd.findtext('xmax', default='0')))
            ymax = int(float(bnd.findtext('ymax', default='0')))
            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))
        if len(boxes) == 0:
            return None
        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        return (min(xs1), min(ys1), max(xs2), max(ys2))

    def _build_samples(self) -> List[Dict]:
        samples = []
        for image_id in self._read_split_ids():
            xml_path = self.ann_dir / f'{image_id}.xml'
            if not xml_path.exists():
                continue
            img_path = self._find_image_path(image_id)
            label = self.label_map.get(image_id)
            if label is None:
                label = self.label_map.get(img_path.name)
            if label is None:
                raise KeyError(f"label_manifest 中未找到标签: {image_id}")
            bbox = self._parse_xml_bbox(xml_path)
            samples.append({
                'image_id': image_id,
                'image_path': str(img_path),
                'xml_path': str(xml_path),
                'label': int(label),
                'bbox': bbox,
            })
        return samples

    def _count_labels(self) -> Dict[int, int]:
        counter = defaultdict(int)
        for s in self.samples:
            counter[int(s['label'])] += 1
        return dict(counter)

    def __len__(self):
        return len(self.samples)

    def _expand_and_clip_box(self, box, width: int, height: int):
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

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample['image_path']).convert('RGB')
        if self.use_roi_crop and sample['bbox'] is not None:
            crop_box = self._expand_and_clip_box(sample['bbox'], img.width, img.height)
            if crop_box is not None:
                img = img.crop(crop_box)
            elif not self.use_whole_image_fallback:
                raise RuntimeError(f"无效裁剪框: {sample['image_id']}")
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


def prepare_datasets(seed: int = 17):
    set_seed(seed)
    train_transform, eval_transform = build_transforms()
    datasets = {
        'train': AULBinaryVOCRoiDataset(Config.data_root, Config.train_split, train_transform,
                                   use_roi_crop=Config.use_roi_crop,
                                   bbox_expand_ratio=Config.bbox_expand_ratio,
                                   min_crop_size=Config.min_crop_size,
                                   use_whole_image_fallback=Config.use_whole_image_fallback),
        'val': AULBinaryVOCRoiDataset(Config.data_root, Config.val_split, eval_transform,
                                 use_roi_crop=Config.use_roi_crop,
                                 bbox_expand_ratio=Config.bbox_expand_ratio,
                                 min_crop_size=Config.min_crop_size,
                                 use_whole_image_fallback=Config.use_whole_image_fallback),
        'test': AULBinaryVOCRoiDataset(Config.data_root, Config.test_split, eval_transform,
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
            print(f'[INFO] inferred feature dim for {Config.backbone_name} = {feat_dim}')
        elif self.model_family == 'custom':
            module = importlib.import_module(Config.backbone_module)
            backbone_fn = getattr(module, Config.backbone_func)
            self.backbone = backbone_fn(num_classes=Config.backbone_out_dim, img_size=Config.input_size)
            feat_dim = int(Config.backbone_out_dim)
            print(f'[INFO] custom backbone output dim = {feat_dim}')
        else:
            raise ValueError(f'Unsupported model_family: {Config.model_family}')

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(Config.dropout),
            nn.Linear(512, num_classes),
        )

    @torch.no_grad()
    def _infer_feat_dim(self) -> int:
        dummy = torch.zeros(1, 3, Config.input_size, Config.input_size)
        feats = self.extract_features(dummy)
        if feats.ndim != 2:
            raise RuntimeError(f'Unexpected feature shape: {tuple(feats.shape)}')
        return int(feats.shape[1])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
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


def build_optimizer(model: nn.Module):
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
    print(f'[OPTIM] backbone_lr={Config.backbone_lr:.2e}, head_lr={Config.learning_rate:.2e}')
    print(f'[OPTIM] trainable backbone params={n_backbone}, head params={n_head}')
    return optim.AdamW(param_groups)


class ModelEMA:
    def __init__(self, model, decay: float = 0.999):
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


def create_result_dir() -> Path:
    result_dir = Path(Config.output_root) / datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def build_class_weights(train_dataset: AULBinaryVOCRoiDataset, device: torch.device):
    counts = [train_dataset.label_counts.get(i, 0) for i in range(Config.num_classes)]
    counts = np.array(counts, dtype=np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics_from_labels_probs(all_labels: np.ndarray, all_probs: np.ndarray, threshold: float = 0.5):
    all_preds = (all_probs >= threshold).astype(int)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
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


def scan_thresholds(y_true: np.ndarray, y_prob: np.ndarray, start: float, end: float, step: float):
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


def save_threshold_scan_csv(results, out_path: Path):
    fields = ['threshold', 'bal_acc', 'f1_macro', 'acc', 'auc', 'recall_0', 'recall_1', 'tn', 'fp', 'fn', 'tp']
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name: str = 'val', threshold: float = 0.5):
    model.eval()
    losses, all_labels, all_probs, all_ids, all_paths = [], [], [], [], []
    with tqdm(loader, desc=f'Evaluating-{split_name}') as pbar:
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


def logits_to_probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = torch.tensor(logits / max(float(temperature), 1e-6), dtype=torch.float32)
    probs = torch.softmax(x, dim=1)[:, 1].cpu().numpy()
    return probs


@torch.no_grad()
def collect_logits(model, loader, device, split_name: str = 'val', use_hflip_tta: bool = False):
    model.eval()
    all_labels, all_logits, all_ids, all_paths = [], [], [], []
    with tqdm(loader, desc=f'CollectLogits-{split_name}') as pbar:
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


def collect_ensemble_logits(model, ckpt_paths, loader, device, split_name: str = 'val', use_hflip_tta: bool = False):
    logits_sum = None
    labels_ref, ids_ref, paths_ref = None, None, None
    used_ckpts = []
    for ckpt_path in ckpt_paths:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        labels, logits, ids, paths = collect_logits(model, loader, device, split_name=f'{split_name}-{ckpt_path.stem}', use_hflip_tta=use_hflip_tta)
        if labels_ref is None:
            labels_ref, ids_ref, paths_ref = labels, ids, paths
        else:
            if not np.array_equal(labels_ref, labels):
                raise RuntimeError(f'标签顺序不一致，无法集成: {ckpt_path}')
            if ids_ref != ids:
                raise RuntimeError(f'image_id 顺序不一致，无法集成: {ckpt_path}')
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

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(min=1e-3, max=100.0)
        return logits / temperature


def fit_temperature_on_val(logits_np: np.ndarray, labels_np: np.ndarray, device: torch.device) -> float:
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


def compute_nll_from_logits(logits_np: np.ndarray, labels_np: np.ndarray, temperature: float = 1.0) -> float:
    logits = torch.tensor(logits_np / max(float(temperature), 1e-6), dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    return float(nn.CrossEntropyLoss()(logits, labels).item())


def metric_for_selection(metrics: Dict) -> float:
    key = Config.save_by_metric
    if key not in metrics:
        raise KeyError(f'save_by_metric={key} 不在 metrics 中')
    return float(metrics[key])


def get_epoch_lr_for_group(param_group, epoch: int) -> float:
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


def train_model(model, dataloaders, criterion, optimizer, result_dir: Path):
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
        with tqdm(dataloaders['train'], desc=f'Train Epoch {epoch + 1}/{Config.num_epochs}') as pbar:
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
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{correct / max(total, 1):.4f}', 'LR': f'{head_lr:.2e}'})

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

        print(
            f"Epoch {epoch + 1:03d} | lr={head_lr:.2e} train_loss={train_loss_avg:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_bal_acc={val_metrics['bal_acc']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

        if score > best_score + Config.early_stopping_min_delta:
            best_score = score
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(eval_model.state_dict(), best_path)
            if Config.save_all_improved_checkpoints:
                ckpt_path = result_dir / f"epoch{best_epoch:03d}_{Config.save_by_metric}_{best_score:.4f}.pth"
                torch.save(eval_model.state_dict(), ckpt_path)
                improved_ckpts.append({'epoch': int(best_epoch), 'score': float(best_score), 'path': str(ckpt_path)})
                print(f'[SAVE] improved checkpoint -> {ckpt_path}')
            print(f'[SAVE] New best model saved to: {best_path} | {Config.save_by_metric}={best_score:.4f} | epoch={best_epoch}')
        else:
            epochs_no_improve += 1
            print(f'[EARLY_STOP_CHECK] no improvement for {epochs_no_improve}/{Config.early_stopping_patience} epochs | best_{Config.save_by_metric}={best_score:.4f} @ epoch {best_epoch}')
            if epochs_no_improve >= Config.early_stopping_patience:
                print(f'[EARLY_STOP] Triggered at epoch {epoch + 1}. Best epoch={best_epoch}, best_{Config.save_by_metric}={best_score:.4f}')
                break

    torch.save(model.state_dict(), last_path)
    print(f'[SAVE] Last model saved to: {last_path}')
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


def save_curves(history: Dict, result_dir: Path):
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


def save_roc_points(labels: np.ndarray, probs: np.ndarray, out_path: Path):
    try:
        fpr, tpr, thr = roc_curve(labels, probs)
    except Exception:
        fpr, tpr, thr = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.nan, np.nan])
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['fpr', 'tpr', 'threshold'])
        for a, b, c in zip(fpr, tpr, thr):
            writer.writerow([float(a), float(b), float(c)])


def save_confusion_matrix_csv(cm: np.ndarray, out_path: Path):
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['', *Config.class_names])
        for idx, row in enumerate(cm.tolist()):
            writer.writerow([Config.class_names[idx], *row])


def save_confusion_matrix_png(cm: np.ndarray, title: str, out_path: Path):
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


def save_predictions_csv(image_ids: List[str], image_paths: List[str], labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, threshold: float, out_path: Path):
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'image_path', 'true_label', 'pred_label', 'prob_class0', 'prob_class1', 'threshold', 'is_wrong', 'wrong_conf', 'margin_from_05'])
        for image_id, image_path, y, p, prob1 in zip(image_ids, image_paths, labels, preds, probs):
            prob0 = 1.0 - float(prob1)
            wrong = int(int(y) != int(p))
            wrong_conf = float(prob1) if int(p) == 1 else float(prob0)
            margin = abs(float(prob1) - 0.5)
            writer.writerow([image_id, image_path, int(y), int(p), prob0, float(prob1), float(threshold), wrong, wrong_conf, margin])


def save_hardcases_csv(pred_csv: Path, out_path: Path):
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


def save_summary(history: Dict, val_metrics: Dict, test_metrics: Dict,
                 single_ms: float, batch_ms: float, fps: float,
                 result_dir: Path, best_epoch: int, best_score: float,
                 used_ckpts: List[str], temperature: float,
                 val_nll_before: float, val_nll_after: float,
                 best_threshold: float, best_thr_info: Optional[Dict]):
    txt_path = result_dir / 'summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('AUL ROI Binary Classification Summary\n')
        f.write('=' * 70 + '\n')
        f.write(f'model_family: {Config.model_family}\n')
        f.write(f'backbone_name: {Config.backbone_name}\n')
        f.write(f'backbone_module: {Config.backbone_module}\n')
        f.write(f'backbone_func: {Config.backbone_func}\n')
        f.write(f'input_size: {Config.input_size}\n')
        f.write(f'use_roi_crop: {Config.use_roi_crop}\n')
        f.write(f'bbox_expand: {Config.bbox_expand_ratio}\n')
        f.write(f'batch_size: {Config.batch_size}\n')
        f.write(f'num_epochs: {Config.num_epochs}\n')
        f.write(f'weight_decay: {Config.weight_decay}\n')
        f.write(f'label_smoothing: {Config.label_smoothing}\n')
        f.write(f'early_stop_pat: {Config.early_stopping_patience}\n')
        f.write(f'save_by_metric: {Config.save_by_metric}\n')
        f.write(f'train_split: {Config.train_split}\n')
        f.write(f'val_split: {Config.val_split}\n')
        f.write(f'test_split: {Config.test_split}\n')
        f.write(f'fold_idx: {Config.fold_idx}\n')
        f.write(f'seed: {Config.seed}\n\n')

        f.write('Validation Metrics\n')
        f.write('-' * 70 + '\n')
        f.write(f"loss={val_metrics['loss']:.6f}\n")
        f.write(f"acc={val_metrics['acc']:.6f}\n")
        f.write(f"bal_acc={val_metrics['bal_acc']:.6f}\n")
        f.write(f"f1_macro={val_metrics['f1_macro']:.6f}\n")
        f.write(f"auc={val_metrics['auc']:.6f}\n\n")
        f.write(val_metrics['report'] + '\n\n')

        f.write('Test Metrics\n')
        f.write('-' * 70 + '\n')
        f.write(f"loss={test_metrics['loss']:.6f}\n")
        f.write(f"acc={test_metrics['acc']:.6f}\n")
        f.write(f"bal_acc={test_metrics['bal_acc']:.6f}\n")
        f.write(f"f1_macro={test_metrics['f1_macro']:.6f}\n")
        f.write(f"auc={test_metrics['auc']:.6f}\n\n")
        f.write(test_metrics['report'] + '\n\n')

        f.write('Inference Time\n')
        f.write('-' * 70 + '\n')
        f.write(f'single_image_ms={single_ms:.4f}\n')
        f.write(f'batch_ms={batch_ms:.4f}\n')
        f.write(f'fps={fps:.4f}\n\n')

        inferred_best_epoch = int(np.argmax(history['val_auc'])) + 1 if len(history['val_auc']) > 0 else -1
        f.write(f'best_epoch_by_runtime={best_epoch}\n')
        f.write(f'best_score_by_runtime={best_score:.6f}\n')
        f.write(f'best_epoch_by_history_auc={inferred_best_epoch}\n')
        f.write(f'param_tag: {Config.param_tag}\n')
        f.write(f'backbone_lr: {Config.backbone_lr}\n')
        f.write(f'head_lr: {Config.learning_rate}\n')
        f.write(f'use_class_weight: {Config.use_class_weight}\n')
        f.write(f'use_manual_class_weights: {Config.use_manual_class_weights}\n')
        f.write(f'manual_class_weights: {list(Config.manual_class_weights)}\n')
        f.write(f'freeze_low_level: False\n')
        f.write(f'use_ema: {Config.use_ema}\n')
        f.write(f'ema_decay: {Config.ema_decay}\n')
        f.write(f'use_hflip_tta: {Config.use_hflip_tta}\n')
        f.write(f'use_temperature_scaling: {Config.use_temperature_scaling}\n')
        f.write(f'ensemble_topk_requested: {Config.ensemble_topk}\n')
        f.write(f'ensemble_ckpts_used: {used_ckpts}\n')
        f.write(f'temperature: {temperature:.6f}\n')
        f.write(f'val_nll_before_temp: {val_nll_before:.6f}\n')
        f.write(f'val_nll_after_temp: {val_nll_after:.6f}\n')
        f.write('\nThreshold Selection\n')
        f.write('=' * 60 + '\n')
        f.write(f'Selected threshold: {best_threshold:.4f}\n')
        f.write(f'Selection mode: {Config.threshold_selection_mode}\n')
        if best_thr_info is not None:
            for k, v in best_thr_info.items():
                f.write(f'{k}: {v}\n')
    return txt_path


def main():
    args = parse_args()
    apply_args(args)
    set_seed(Config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_dir = create_result_dir()
    print('=' * 80)
    print('AUL ROI Binary Training Script (Unified 5-fold)')
    print('=' * 80)
    print(f'[INFO] result_dir      = {result_dir}')
    print(f'[INFO] data_root       = {Config.data_root}')
    print(f'[INFO] model_family    = {Config.model_family}')
    print(f'[INFO] backbone_name   = {Config.backbone_name}')
    print(f'[INFO] backbone_module = {Config.backbone_module}')
    print(f'[INFO] backbone_func   = {Config.backbone_func}')
    print(f'[INFO] use_roi_crop    = {Config.use_roi_crop}')
    print(f'[INFO] bbox_expand     = {Config.bbox_expand_ratio}')
    print(f'[INFO] device          = {device}')
    print(f'[INFO] num_epochs      = {Config.num_epochs}')
    print(f'[INFO] head_lr         = {Config.learning_rate}')
    print(f'[INFO] backbone_lr     = {Config.backbone_lr}')
    print(f'[INFO] weight_decay    = {Config.weight_decay}')
    print(f'[INFO] label_smoothing = {Config.label_smoothing}')
    print(f'[INFO] early_stop_pat  = {Config.early_stopping_patience}')
    print(f'[INFO] save_by_metric  = {Config.save_by_metric}')
    print(f'[INFO] thr_mode        = {Config.threshold_selection_mode}')
    print(f'[INFO] lr_schedule     = {Config.lr_schedule}')
    print(f'[INFO] lr_warmup_ep    = {Config.lr_warmup_epochs}')
    print(f'[INFO] lr_min_ratio    = {Config.lr_min_ratio}')
    print(f'[INFO] param_tag       = {Config.param_tag}')
    print(f'[INFO] fold_idx        = {Config.fold_idx}')
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
        print(f'[INFO] class_weights = {class_weights.detach().cpu().numpy().tolist()}')

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.label_smoothing)
    optimizer = build_optimizer(model)

    single_ms, batch_ms, fps = measure_inference_time(model, dataloaders['test'], repetitions=20)
    print(f'[INFO] single_image_ms = {single_ms:.4f}')
    print(f'[INFO] batch_ms        = {batch_ms:.4f}')
    print(f'[INFO] fps             = {fps:.4f}')

    history, best_path, _, best_epoch, best_score, improved_ckpts = train_model(model, dataloaders, criterion, optimizer, result_dir)
    curves_path = save_curves(history, result_dir)
    print(f'[SAVE] curves -> {curves_path}')

    candidate_ckpts = sorted(improved_ckpts, key=lambda x: (x['score'], x['epoch']), reverse=True)
    candidate_ckpt_paths = [x['path'] for x in candidate_ckpts[:Config.ensemble_topk]]
    if not candidate_ckpt_paths:
        candidate_ckpt_paths = [str(best_path)]
    if len(candidate_ckpt_paths) < Config.ensemble_topk:
        print(f'[ENSEMBLE] 可用 improved checkpoints={len(candidate_ckpt_paths)} < ensemble_topk={Config.ensemble_topk}，本次将退化为单/少量 checkpoint 推理。')
    print(f'[ENSEMBLE] checkpoint(s) used: {candidate_ckpt_paths}')

    val_labels, val_logits, val_ids, val_paths, used_ckpts = collect_ensemble_logits(model, candidate_ckpt_paths, dataloaders['val'], device, split_name='val', use_hflip_tta=Config.use_hflip_tta)
    test_labels, test_logits, test_ids, test_paths, _ = collect_ensemble_logits(model, used_ckpts, dataloaders['test'], device, split_name='test', use_hflip_tta=Config.use_hflip_tta)

    temperature = 1.0
    val_nll_before = compute_nll_from_logits(val_logits, val_labels, temperature=1.0)
    val_nll_after = val_nll_before
    if Config.use_temperature_scaling:
        temperature = fit_temperature_on_val(val_logits, val_labels, device)
        val_nll_after = compute_nll_from_logits(val_logits, val_labels, temperature=temperature)
        print(f'[TEMP] fitted temperature = {temperature:.6f} | val_nll_before={val_nll_before:.6f} | val_nll_after={val_nll_after:.6f}')
    else:
        print(f'[TEMP] disabled, using temperature = {temperature:.6f}')

    val_probs = logits_to_probs(val_logits, temperature=temperature)
    test_probs = logits_to_probs(test_logits, temperature=temperature)

    best_threshold = 0.5
    best_thr_info = None
    if Config.do_threshold_search:
        threshold_results = scan_thresholds(val_labels, val_probs, start=Config.threshold_start, end=Config.threshold_end, step=Config.threshold_step)
        save_threshold_scan_csv(threshold_results, result_dir / 'val_threshold_scan.csv')
        best_thr_info = choose_best_threshold(threshold_results)
        best_threshold = float(best_thr_info['threshold'])
        print(f'[THRESHOLD] selected threshold = {best_threshold:.4f} | mode={Config.threshold_selection_mode} | info={best_thr_info}')
    else:
        print(f'[THRESHOLD] threshold search disabled, using threshold = {best_threshold:.4f}')

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
    save_confusion_matrix_png(val_metrics['cm'], f'Val CM | Thr={best_threshold:.2f} AUC={val_metrics["auc"]:.4f}', result_dir / 'val_confusion_matrix.png')
    save_confusion_matrix_png(test_metrics['cm'], f'Test CM | Thr={best_threshold:.2f} AUC={test_metrics["auc"]:.4f}', result_dir / 'test_confusion_matrix.png')
    summary_path = save_summary(history, val_metrics, test_metrics, single_ms, batch_ms, fps, result_dir, best_epoch, best_score, used_ckpts, temperature, val_nll_before, val_nll_after, best_threshold, best_thr_info)

    meta = asdict(Config())
    meta['result_dir'] = str(result_dir)
    meta['class_weights'] = class_weights.detach().cpu().numpy().tolist() if class_weights is not None else None
    with open(result_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 80)
    print('Final Validation Metrics')
    print('=' * 80)
    print(f'threshold={best_threshold:.4f}')
    print(val_metrics['report'])
    print('Final Test Metrics')
    print('=' * 80)
    print(test_metrics['report'])
    print(f'[SAVE] summary -> {summary_path}')
    print(f'[DONE] all outputs saved under: {result_dir}')


if __name__ == '__main__':
    main()
