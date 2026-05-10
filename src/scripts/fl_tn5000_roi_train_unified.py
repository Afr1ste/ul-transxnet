import csv
import io
import json
import math
import copy
import time
import random
import warnings
import contextlib
import multiprocessing
import importlib
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

warnings.filterwarnings('ignore')


DEFAULT_CONFIG = {
    # ===== 数据 =====
    'tn5000_root': r"<LOCAL_THYROID_ROOT>\TN5000_forReview",
    'output_root': 'tn5000_roi_unified_runs',

    # ===== 模型 =====
    'backbone_module': 'models.transxnetggg',
    'backbone_func': 'transxnet_t',
    'backbone_out_dim': 1000,
    'head_hidden_dim': 512,
    'input_size': 256,
    'num_classes': 2,
    'class_names': ['0', '1'],

    # ===== ROI 裁剪 =====
    'use_roi_crop': True,
    'bbox_expand_ratio': 0.30,
    'min_crop_size': 64,
    'use_whole_image_fallback': True,

    # ===== 训练 =====
    'batch_size': 16,
    'num_workers': 0,
    'num_epochs': 120,
    'backbone_lr': 5e-5,
    'head_lr': 1.5e-4,
    'weight_decay': 1e-4,
    'dropout': 0.30,
    'label_smoothing': 0.00,
    'early_stopping_patience': 12,
    'early_stopping_min_delta': 1e-4,
    'seed': 17,
    'save_by_metric': 'auc',           # auc / bal_acc / f1_macro / acc

    # ===== 优化细节 =====
    'optimizer_name': 'adamw',
    'use_amp': True,
    'max_grad_norm': 3.0,
    'freeze_backbone': False,
    'freeze_backbone_norm': False,

    # ===== 学习率调度 =====
    'lr_schedule': 'cosine_floor',     # cosine_floor / cosine / constant
    'lr_warmup_epochs': 5,
    'lr_min_ratio': 0.25,

    # ===== 类别权重 =====
    'use_class_weight': False,
    'use_manual_class_weights': False,
    'manual_class_weights': [1.0, 1.0],

    # ===== EMA =====
    'use_ema': True,
    'ema_decay': 0.9995,

    # ===== 阈值搜索 =====
    'do_threshold_search': True,
    'threshold_start': 0.30,
    'threshold_end': 0.70,
    'threshold_step': 0.01,
    'threshold_selection_mode': 'bal_acc',   # bal_acc / f1_macro / recall_constraint
    'min_recall_1': 0.90,

    # ===== 推理组件 =====
    'run_postprocess_variants': False,
    'use_hflip_tta': True,
    'use_temperature_scaling': True,
    'ensemble_topk': 3,
    'save_all_improved_checkpoints': True,
    'max_saved_improved_ckpts': 12,

    # ===== 分析 =====
    'measure_model_complexity': True,
    'measure_inference_time': True,
    'inference_time_repetitions': 20,

    # ===== 输出 =====
    'run_name': '',
    'best_model_name': 'best_model.pth',
    'final_model_name': 'last_model.pth',
}


class Config:
    pass


def reset_config() -> None:
    for k, v in DEFAULT_CONFIG.items():
        setattr(Config, k, copy.deepcopy(v))


reset_config()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            if not torch.is_floating_point(v):
                v.copy_(model_state[k])
            else:
                v.mul_(self.decay).add_(model_state[k].detach(), alpha=1.0 - self.decay)


class TN5000ROIDataset(Dataset):
    """
    TN5000 官方划分 + XML 标签 + ROI 裁剪版
    - 读取 ImageSets/Main/{train,val,test}.txt
    - 标签来自 XML 的 object/name: 0=benign, 1=malignant
    - 框来自 XML 的 object/bndbox
    - 多目标时：使用所有框的并集；标签按 malignant 优先（max）
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform=None,
        use_roi_crop: bool = True,
        bbox_expand_ratio: float = 0.30,
        min_crop_size: int = 64,
        use_whole_image_fallback: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_roi_crop = bool(use_roi_crop)
        self.bbox_expand_ratio = float(bbox_expand_ratio)
        self.min_crop_size = int(min_crop_size)
        self.use_whole_image_fallback = bool(use_whole_image_fallback)

        self.image_dir = self.root_dir / 'JPEGImages'
        self.ann_dir = self.root_dir / 'Annotations'
        self.split_file = self.root_dir / 'ImageSets' / 'Main' / f'{split}.txt'

        if not self.image_dir.exists():
            raise FileNotFoundError(f'JPEGImages 不存在: {self.image_dir}')
        if not self.ann_dir.exists():
            raise FileNotFoundError(f'Annotations 不存在: {self.ann_dir}')
        if not self.split_file.exists():
            raise FileNotFoundError(f'{split}.txt 不存在: {self.split_file}')

        self.samples = self._build_samples()
        self.label_counts = self._count_labels()
        if len(self.samples) == 0:
            raise RuntimeError(f'{split} 集为空，请检查 TN5000 目录结构。')

        print(f'[TN5000-ROI] split={split}, samples={len(self.samples)}, label_counts={self.label_counts}')

    def _read_split_ids(self) -> List[str]:
        ids: List[str] = []
        with open(self.split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(line)
        return ids

    def _find_image_path(self, image_id: str) -> Path:
        for suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
            p = self.image_dir / f'{image_id}{suffix}'
            if p.exists():
                return p
        raise FileNotFoundError(f'找不到图像文件: {image_id}')

    def _parse_xml(self, xml_path: Path) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        labels: List[int] = []
        boxes: List[Tuple[int, int, int, int]] = []

        for obj in root.findall('object'):
            name_node = obj.find('name')
            if name_node is not None and name_node.text is not None:
                name_text = name_node.text.strip().lower()
                if name_text in ['0', 'benign']:
                    labels.append(0)
                elif name_text in ['1', 'malignant']:
                    labels.append(1)
                else:
                    raise ValueError(f'无法识别标签: {name_text}, file={xml_path}')

            bnd = obj.find('bndbox')
            if bnd is not None:
                xmin = int(float(bnd.findtext('xmin', default='0')))
                ymin = int(float(bnd.findtext('ymin', default='0')))
                xmax = int(float(bnd.findtext('xmax', default='0')))
                ymax = int(float(bnd.findtext('ymax', default='0')))
                if xmax > xmin and ymax > ymin:
                    boxes.append((xmin, ymin, xmax, ymax))

        if not labels:
            raise ValueError(f'XML 中没有有效标签: {xml_path}')

        label = max(labels)
        if not boxes:
            return label, None

        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        union_box = (min(xs1), min(ys1), max(xs2), max(ys2))
        return label, union_box

    def _build_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        skipped = 0
        for image_id in self._read_split_ids():
            xml_path = self.ann_dir / f'{image_id}.xml'
            if not xml_path.exists():
                skipped += 1
                continue
            try:
                img_path = self._find_image_path(image_id)
                label, bbox = self._parse_xml(xml_path)
                samples.append(
                    {
                        'image_id': image_id,
                        'image_path': str(img_path),
                        'xml_path': str(xml_path),
                        'label': int(label),
                        'bbox': bbox,
                    }
                )
            except Exception as exc:
                skipped += 1
                print(f'[跳过] {image_id} 解析失败: {exc}')
        if skipped > 0:
            print(f'[INFO] split={self.split}, skipped={skipped}')
        return samples

    def _count_labels(self) -> Dict[int, int]:
        counter = defaultdict(int)
        for s in self.samples:
            counter[int(s['label'])] += 1
        return dict(counter)

    def __len__(self) -> int:
        return len(self.samples)

    def _expand_and_clip_box(self, box: Tuple[int, int, int, int], width: int, height: int):
        x1, y1, x2, y2 = box
        bw = max(x2 - x1, self.min_crop_size)
        bh = max(y2 - y1, self.min_crop_size)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        new_w = bw * (1.0 + 2.0 * self.bbox_expand_ratio)
        new_h = bh * (1.0 + 2.0 * self.bbox_expand_ratio)
        nx1 = max(0, int(round(cx - new_w / 2.0)))
        ny1 = max(0, int(round(cy - new_h / 2.0)))
        nx2 = min(width, int(round(cx + new_w / 2.0)))
        ny2 = min(height, int(round(cy + new_h / 2.0)))
        if nx2 <= nx1 or ny2 <= ny1:
            return None
        return nx1, ny1, nx2, ny2

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample['image_path']).convert('RGB')
        if self.use_roi_crop and sample['bbox'] is not None:
            crop_box = self._expand_and_clip_box(sample['bbox'], img.width, img.height)
            if crop_box is not None:
                img = img.crop(crop_box)
            elif not self.use_whole_image_fallback:
                raise RuntimeError(f'无效裁剪框: {sample["image_id"]}')
        if self.transform is not None:
            img = self.transform(img)
        return img, int(sample['label'])


class ClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        backbone_fn = load_backbone_fn(Config.backbone_module, Config.backbone_func)
        self.backbone = backbone_fn(num_classes=Config.backbone_out_dim, img_size=Config.input_size)
        self.head = nn.Sequential(
            nn.Linear(Config.backbone_out_dim, Config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.head_hidden_dim, num_classes),
        )
        if Config.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        if Config.freeze_backbone_norm:
            self._freeze_norm_layers(self.backbone)

    @staticmethod
    def _freeze_norm_layers(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm)):
                for p in m.parameters():
                    p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            raise RuntimeError('当前分类脚本不支持 fork_feat 多输出 backbone。')
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return self.head(features)


def load_backbone_fn(module_name: str, func_name: str):
    module = importlib.import_module(module_name)
    if not hasattr(module, func_name):
        raise AttributeError(f'{module_name} 中不存在 {func_name}')
    return getattr(module, func_name)


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.input_size, Config.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(3),
        transforms.ColorJitter(0.08, 0.08, 0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((Config.input_size, Config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def prepare_tn5000_roi_datasets(seed: int):
    set_seed(seed)
    train_transform, eval_transform = build_transforms()
    return {
        'train': TN5000ROIDataset(
            Config.tn5000_root,
            'train',
            train_transform,
            use_roi_crop=Config.use_roi_crop,
            bbox_expand_ratio=Config.bbox_expand_ratio,
            min_crop_size=Config.min_crop_size,
            use_whole_image_fallback=Config.use_whole_image_fallback,
        ),
        'val': TN5000ROIDataset(
            Config.tn5000_root,
            'val',
            eval_transform,
            use_roi_crop=Config.use_roi_crop,
            bbox_expand_ratio=Config.bbox_expand_ratio,
            min_crop_size=Config.min_crop_size,
            use_whole_image_fallback=Config.use_whole_image_fallback,
        ),
        'test': TN5000ROIDataset(
            Config.tn5000_root,
            'test',
            eval_transform,
            use_roi_crop=Config.use_roi_crop,
            bbox_expand_ratio=Config.bbox_expand_ratio,
            min_crop_size=Config.min_crop_size,
            use_whole_image_fallback=Config.use_whole_image_fallback,
        ),
    }


def create_result_dir() -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    leaf = f'{Config.run_name}_{ts}' if str(Config.run_name).strip() else ts
    result_dir = Path(Config.output_root) / leaf
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def build_class_weights(train_dataset: TN5000ROIDataset, device: torch.device) -> torch.Tensor:
    if Config.use_manual_class_weights:
        if len(Config.manual_class_weights) != Config.num_classes:
            raise ValueError('manual_class_weights 长度必须等于 num_classes')
        return torch.tensor(Config.manual_class_weights, dtype=torch.float32, device=device)
    counts = np.array([train_dataset.label_counts.get(i, 0) for i in range(Config.num_classes)], dtype=np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics_from_labels_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = float('nan')
    cm = confusion_matrix(y_true, y_pred, labels=list(range(Config.num_classes)))
    report = classification_report(
        y_true,
        y_pred,
        target_names=Config.class_names,
        digits=4,
        zero_division=0,
    )
    return {
        'acc': float(acc),
        'bal_acc': float(bal_acc),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'auc': float(roc_auc),
        'cm': cm,
        'report': report,
        'labels': y_true,
        'preds': y_pred,
        'probs': y_prob,
        'threshold': float(threshold),
    }


def scan_thresholds(y_true: np.ndarray, y_prob: np.ndarray, start: float, end: float, step: float) -> List[Dict]:
    rows: List[Dict] = []
    thr = start
    while thr <= end + 1e-12:
        thr = float(round(thr, 4))
        metrics = compute_metrics_from_labels_probs(y_true, y_prob, threshold=thr)
        cm = metrics['cm']
        tn, fp, fn, tp = cm.ravel()
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append({
            'threshold': thr,
            'bal_acc': float(metrics['bal_acc']),
            'f1_macro': float(metrics['f1_macro']),
            'acc': float(metrics['acc']),
            'auc': float(metrics['auc']),
            'recall_0': float(recall_0),
            'recall_1': float(recall_1),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        })
        thr += step
    return rows


def choose_best_threshold(rows: List[Dict]) -> Dict:
    mode = Config.threshold_selection_mode
    if mode == 'recall_constraint':
        candidates = [x for x in rows if x['recall_1'] >= Config.min_recall_1]
        if candidates:
            return sorted(candidates, key=lambda x: (x['bal_acc'], x['f1_macro'], -abs(x['threshold'] - 0.5)), reverse=True)[0]
    if mode == 'f1_macro':
        return sorted(rows, key=lambda x: (x['f1_macro'], x['bal_acc'], x['auc'], -abs(x['threshold'] - 0.5)), reverse=True)[0]
    return sorted(rows, key=lambda x: (x['bal_acc'], x['f1_macro'], x['auc'], -abs(x['threshold'] - 0.5)), reverse=True)[0]


def save_threshold_scan_csv(rows: List[Dict], out_path: Path) -> None:
    fieldnames = ['threshold', 'bal_acc', 'f1_macro', 'acc', 'auc', 'recall_0', 'recall_1', 'tn', 'fp', 'fn', 'tp']
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    threshold: float = 0.5,
    use_hflip_tta: bool = False,
) -> Dict:
    model.eval()
    losses: List[float] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_logits: List[np.ndarray] = []

    with tqdm(loader, desc=f'Eval-{split_name}', leave=False) as pbar:
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            if use_hflip_tta:
                logits_flip = model(torch.flip(inputs, dims=[3]))
                logits = 0.5 * (logits + logits_flip)

            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]

            losses.append(float(loss.item()))
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_logits.append(logits.detach().cpu().numpy())

    labels_np = np.array(all_labels, dtype=int)
    probs_np = np.array(all_probs, dtype=float)
    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, Config.num_classes), dtype=np.float32)

    metrics = compute_metrics_from_labels_probs(labels_np, probs_np, threshold=threshold)
    metrics['loss'] = float(np.mean(losses)) if losses else 0.0
    metrics['logits'] = logits_np
    return metrics


def metric_for_selection(metrics: Dict) -> float:
    key = Config.save_by_metric
    if key not in metrics:
        raise KeyError(f'save_by_metric={key} 不在 metrics 中')
    return float(metrics[key])


def get_lr_scale(epoch: int) -> float:
    total_epochs = max(int(Config.num_epochs), 1)
    warmup_epochs = int(Config.lr_warmup_epochs)
    min_ratio = float(Config.lr_min_ratio)
    if Config.lr_schedule == 'constant':
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0

    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)

    effective_total = max(total_epochs - warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / float(max(effective_total - 1, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    if Config.lr_schedule == 'cosine_floor':
        return min_ratio + (1.0 - min_ratio) * cosine
    return cosine


def build_optimizer(model: ClassificationModel):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': Config.backbone_lr})
    if head_params:
        param_groups.append({'params': head_params, 'lr': Config.head_lr})
    if not param_groups:
        raise RuntimeError('没有可训练参数。')
    return optim.AdamW(param_groups, weight_decay=Config.weight_decay)


def apply_epoch_lr(optimizer: optim.Optimizer, epoch: int) -> Dict[str, float]:
    scale = get_lr_scale(epoch)
    lrs = {}
    for group in optimizer.param_groups:
        base_lr = group.get('initial_lr', group['lr'])
        if 'initial_lr' not in group:
            group['initial_lr'] = group['lr']
            base_lr = group['lr']
        group['lr'] = float(base_lr) * scale
        lrs[group.get('name', f'group{len(lrs)}')] = float(group['lr'])
    return lrs


def save_checkpoint(
    model: nn.Module,
    ema: Optional[ModelEMA],
    optimizer: optim.Optimizer,
    epoch: int,
    score: float,
    out_path: Path,
) -> None:
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': int(epoch),
        'score': float(score),
        'config': export_config_dict(),
        'ema': ema.module.state_dict() if ema is not None else None,
    }
    torch.save(payload, out_path)


@torch.no_grad()
def load_checkpoint_to_model(model: nn.Module, ckpt_path: Path, device: torch.device, prefer_ema: bool = True) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('ema') if prefer_ema and ckpt.get('ema') is not None else ckpt.get('model', ckpt)
    model.load_state_dict(state_dict, strict=True)
    return ckpt


def trim_improved_ckpts(improved_ckpts: List[Dict]) -> List[Dict]:
    max_keep = int(Config.max_saved_improved_ckpts)
    if max_keep <= 0 or len(improved_ckpts) <= max_keep:
        return improved_ckpts
    improved_ckpts = sorted(improved_ckpts, key=lambda x: (x['score'], x['epoch']), reverse=True)
    for stale in improved_ckpts[max_keep:]:
        try:
            Path(stale['path']).unlink(missing_ok=True)
        except Exception:
            pass
    return improved_ckpts[:max_keep]


def train_model(
    model: ClassificationModel,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    result_dir: Path,
    device: torch.device,
):
    use_amp = bool(Config.use_amp and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ModelEMA(model, decay=Config.ema_decay) if Config.use_ema else None

    history = defaultdict(list)
    best_score = -float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    best_path = result_dir / Config.best_model_name
    last_path = result_dir / Config.final_model_name
    improved_ckpts: List[Dict] = []

    for epoch in range(Config.num_epochs):
        lr_info = apply_epoch_lr(optimizer, epoch)
        model.train()
        train_loss_sum = 0.0
        total = 0
        correct = 0

        with tqdm(dataloaders['train'], desc=f'Train {epoch + 1}/{Config.num_epochs}', leave=False) as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                if Config.max_grad_norm and Config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                if ema is not None:
                    ema.update(model)

                preds = logits.argmax(dim=1)
                batch_size = labels.size(0)
                total += batch_size
                correct += preds.eq(labels).sum().item()
                train_loss_sum += float(loss.item())

                pbar.set_postfix({
                    'loss': f'{float(loss.item()):.4f}',
                    'acc': f'{correct / max(total, 1):.4f}',
                    'bb_lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                })

        train_loss = train_loss_sum / max(len(dataloaders['train']), 1)
        train_acc = correct / max(total, 1)

        eval_model = ema.module if ema is not None else model
        val_metrics = evaluate_model(eval_model, dataloaders['val'], criterion, device, split_name='val', threshold=0.5, use_hflip_tta=False)
        score = metric_for_selection(val_metrics)

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_metrics['loss']))
        history['val_acc'].append(float(val_metrics['acc']))
        history['val_bal_acc'].append(float(val_metrics['bal_acc']))
        history['val_f1_macro'].append(float(val_metrics['f1_macro']))
        history['val_auc'].append(float(val_metrics['auc']))
        history['lr_backbone'].append(float(optimizer.param_groups[0]['lr']))
        history['lr_head'].append(float(optimizer.param_groups[-1]['lr']))

        print(
            f'Epoch {epoch + 1:03d} | '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
            f'val_loss={val_metrics["loss"]:.4f} val_acc={val_metrics["acc"]:.4f} '
            f'val_bal_acc={val_metrics["bal_acc"]:.4f} val_f1={val_metrics["f1_macro"]:.4f} '
            f'val_auc={val_metrics["auc"]:.4f} | '
            f'bb_lr={lr_info.get("group0", optimizer.param_groups[0]["lr"]):.2e} '
            f'head_lr={optimizer.param_groups[-1]["lr"]:.2e}'
        )

        if score > best_score + Config.early_stopping_min_delta:
            best_score = float(score)
            best_epoch = int(epoch + 1)
            epochs_no_improve = 0
            save_checkpoint(model, ema, optimizer, best_epoch, best_score, best_path)
            print(f'[SAVE] best -> {best_path} | {Config.save_by_metric}={best_score:.6f} @ epoch {best_epoch}')

            if Config.save_all_improved_checkpoints:
                ckpt_name = f'epoch{best_epoch:03d}_{Config.save_by_metric}_{best_score:.6f}.pth'
                ckpt_path = result_dir / ckpt_name
                save_checkpoint(model, ema, optimizer, best_epoch, best_score, ckpt_path)
                improved_ckpts.append({'epoch': best_epoch, 'score': best_score, 'path': str(ckpt_path)})
                improved_ckpts = trim_improved_ckpts(improved_ckpts)
        else:
            epochs_no_improve += 1
            print(
                f'[EARLY_STOP_CHECK] no improvement for {epochs_no_improve}/{Config.early_stopping_patience} '
                f'epochs | best_{Config.save_by_metric}={best_score:.6f} @ epoch {best_epoch}'
            )
            if epochs_no_improve >= Config.early_stopping_patience:
                print(f'[EARLY_STOP] triggered at epoch {epoch + 1}')
                break

    save_checkpoint(model, ema, optimizer, int(len(history['train_loss'])), float(best_score), last_path)
    print(f'[SAVE] last -> {last_path}')
    return history, best_path, last_path, best_epoch, best_score, improved_ckpts


def save_curves(history: Dict, result_dir: Path) -> Path:
    fig = plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.plot(history['val_bal_acc'], label='Val Bal Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['val_f1_macro'], label='Val Macro F1')
    plt.plot(history['val_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()

    plt.tight_layout()
    out_path = result_dir / 'training_curves.png'
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=Config.class_names, yticklabels=Config.class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def save_roc(labels: np.ndarray, probs: np.ndarray, out_path: Path, title: str = 'ROC Curve') -> None:
    fig = plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def save_predictions_csv(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['true_label', 'pred_label', 'prob_class1'])
        for y, p, s in zip(labels, preds, probs):
            writer.writerow([int(y), int(p), float(s)])


def save_logits_npz(labels: np.ndarray, logits: np.ndarray, out_path: Path) -> None:
    np.savez_compressed(out_path, labels=labels, logits=logits)


def logits_to_probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    x = torch.tensor(logits / temperature, dtype=torch.float32)
    return torch.softmax(x, dim=1)[:, 1].cpu().numpy()


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
    return max(float(scaler.log_temperature.exp().detach().cpu().item()), 1e-3)


def compute_nll_from_logits(logits_np: np.ndarray, labels_np: np.ndarray, temperature: float = 1.0) -> float:
    logits = torch.tensor(logits_np / max(float(temperature), 1e-6), dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    return float(nn.CrossEntropyLoss()(logits, labels).item())


def run_postprocess_variants(
    model: ClassificationModel,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
    result_dir: Path,
    improved_ckpts: List[Dict],
    best_path: Path,
) -> Dict:
    candidate_paths = [x['path'] for x in sorted(improved_ckpts, key=lambda x: (x['score'], x['epoch']), reverse=True)]
    if not candidate_paths:
        candidate_paths = [str(best_path)]

    variants = [
        {'tag': 'v1_single_raw', 'topk': 1, 'use_hflip_tta': False, 'use_temperature_scaling': False},
        {'tag': 'v2_single_tta', 'topk': 1, 'use_hflip_tta': True, 'use_temperature_scaling': False},
        {'tag': 'v3_ens_tta', 'topk': Config.ensemble_topk, 'use_hflip_tta': True, 'use_temperature_scaling': False},
        {'tag': 'v4_ens_tta_temp', 'topk': Config.ensemble_topk, 'use_hflip_tta': True, 'use_temperature_scaling': True},
    ]

    rows: List[Dict] = []
    best_variant = None

    for variant in variants:
        ckpt_paths = candidate_paths[:max(int(variant['topk']), 1)]
        val_labels = None
        val_logits_sum = None
        test_labels = None
        test_logits_sum = None
        used_ckpts = []

        for ckpt_path in ckpt_paths:
            if not Path(ckpt_path).exists():
                continue
            load_checkpoint_to_model(model, Path(ckpt_path), device, prefer_ema=True)
            val_eval = evaluate_model(model, dataloaders['val'], criterion, device, split_name=f"{variant['tag']}-val", threshold=0.5, use_hflip_tta=variant['use_hflip_tta'])
            test_eval = evaluate_model(model, dataloaders['test'], criterion, device, split_name=f"{variant['tag']}-test", threshold=0.5, use_hflip_tta=variant['use_hflip_tta'])
            if val_labels is None:
                val_labels = val_eval['labels']
                test_labels = test_eval['labels']
                val_logits_sum = val_eval['logits']
                test_logits_sum = test_eval['logits']
            else:
                if not np.array_equal(val_labels, val_eval['labels']) or not np.array_equal(test_labels, test_eval['labels']):
                    raise RuntimeError('postprocess variant 中标签顺序不一致，无法集成。')
                val_logits_sum += val_eval['logits']
                test_logits_sum += test_eval['logits']
            used_ckpts.append(str(ckpt_path))

        if not used_ckpts:
            raise RuntimeError(f'{variant["tag"]} 没有可用 checkpoint。')

        val_logits = val_logits_sum / len(used_ckpts)
        test_logits = test_logits_sum / len(used_ckpts)

        temperature = 1.0
        val_nll_before = compute_nll_from_logits(val_logits, val_labels, temperature=1.0)
        val_nll_after = val_nll_before
        if variant['use_temperature_scaling']:
            temperature = fit_temperature_on_val(val_logits, val_labels, device)
            val_nll_after = compute_nll_from_logits(val_logits, val_labels, temperature=temperature)

        val_probs = logits_to_probs(val_logits, temperature=temperature)
        test_probs = logits_to_probs(test_logits, temperature=temperature)
        threshold_rows = scan_thresholds(val_labels, val_probs, Config.threshold_start, Config.threshold_end, Config.threshold_step)
        save_threshold_scan_csv(threshold_rows, result_dir / f'{variant["tag"]}_val_threshold_scan.csv')
        best_thr = choose_best_threshold(threshold_rows)
        thr = float(best_thr['threshold'])

        val_metrics = compute_metrics_from_labels_probs(val_labels, val_probs, threshold=thr)
        test_metrics = compute_metrics_from_labels_probs(test_labels, test_probs, threshold=thr)
        val_metrics['loss'] = compute_nll_from_logits(val_logits, val_labels, temperature=temperature)
        test_metrics['loss'] = compute_nll_from_logits(test_logits, test_labels, temperature=temperature)

        save_predictions_csv(val_metrics['labels'], val_metrics['preds'], val_metrics['probs'], result_dir / f'{variant["tag"]}_val_predictions.csv')
        save_predictions_csv(test_metrics['labels'], test_metrics['preds'], test_metrics['probs'], result_dir / f'{variant["tag"]}_test_predictions.csv')
        save_logits_npz(val_labels, val_logits, result_dir / f'{variant["tag"]}_val_logits.npz')
        save_logits_npz(test_labels, test_logits, result_dir / f'{variant["tag"]}_test_logits.npz')
        save_confusion_matrix(val_metrics['cm'], f'Val CM | {variant["tag"]}', result_dir / f'{variant["tag"]}_val_confusion_matrix.png')
        save_confusion_matrix(test_metrics['cm'], f'Test CM | {variant["tag"]}', result_dir / f'{variant["tag"]}_test_confusion_matrix.png')
        save_roc(val_metrics['labels'], val_metrics['probs'], result_dir / f'{variant["tag"]}_val_roc_curve.png', title=f'Val ROC | {variant["tag"]}')
        save_roc(test_metrics['labels'], test_metrics['probs'], result_dir / f'{variant["tag"]}_test_roc_curve.png', title=f'Test ROC | {variant["tag"]}')

        tn_v, fp_v, fn_v, tp_v = val_metrics['cm'].ravel()
        tn_t, fp_t, fn_t, tp_t = test_metrics['cm'].ravel()
        row = {
            'tag': variant['tag'],
            'topk_requested': int(variant['topk']),
            'num_ckpts_used': int(len(used_ckpts)),
            'use_hflip_tta': bool(variant['use_hflip_tta']),
            'use_temperature_scaling': bool(variant['use_temperature_scaling']),
            'temperature': float(temperature),
            'selected_threshold': thr,
            'val_nll_before': float(val_nll_before),
            'val_nll_after': float(val_nll_after),
            'val_acc': float(val_metrics['acc']),
            'val_bal_acc': float(val_metrics['bal_acc']),
            'val_f1_macro': float(val_metrics['f1_macro']),
            'val_auc': float(val_metrics['auc']),
            'val_recall_0': float(tn_v / (tn_v + fp_v)) if (tn_v + fp_v) > 0 else 0.0,
            'val_recall_1': float(tp_v / (tp_v + fn_v)) if (tp_v + fn_v) > 0 else 0.0,
            'test_acc': float(test_metrics['acc']),
            'test_bal_acc': float(test_metrics['bal_acc']),
            'test_f1_macro': float(test_metrics['f1_macro']),
            'test_auc': float(test_metrics['auc']),
            'test_recall_0': float(tn_t / (tn_t + fp_t)) if (tn_t + fp_t) > 0 else 0.0,
            'test_recall_1': float(tp_t / (tp_t + fn_t)) if (tp_t + fn_t) > 0 else 0.0,
            'ckpts_used': ' | '.join(used_ckpts),
        }
        rows.append(row)

        if best_variant is None or row['test_bal_acc'] > best_variant['row']['test_bal_acc']:
            best_variant = {'row': row, 'val_metrics': val_metrics, 'test_metrics': test_metrics}

    comparison_csv = result_dir / 'postprocess_ablation_comparison.csv'
    fieldnames = list(rows[0].keys())
    with open(comparison_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        'rows': rows,
        'best_variant': best_variant,
        'comparison_csv': str(comparison_csv),
    }


@torch.no_grad()
def measure_inference_time(model: nn.Module, loader: DataLoader, repetitions: int = 20) -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    batch = next(iter(loader))[0].to(device)
    single = batch[:1]
    for _ in range(10):
        _ = model(single)
        _ = model(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(repetitions):
        _ = model(single)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    single_ms = (time.time() - start) * 1000.0 / repetitions

    start = time.time()
    for _ in range(repetitions):
        _ = model(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    batch_ms = (time.time() - start) * 1000.0 / repetitions
    fps = batch.size(0) / max(batch_ms / 1000.0, 1e-8)
    return {'single_image_ms': float(single_ms), 'batch_ms': float(batch_ms), 'fps': float(fps)}


def analyze_model_complexity(model: nn.Module) -> Dict[str, Optional[float]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy = torch.randn(1, 3, Config.input_size, Config.input_size, device=device)
    model = copy.deepcopy(model).to(device).eval()
    params = float(sum(p.numel() for p in model.parameters()))
    thop_flops = None
    fvcore_flops = None
    try:
        from thop import profile  # type: ignore
        flops, params_thop = profile(model, inputs=(dummy,), verbose=False)
        thop_flops = float(flops)
        params = float(params_thop)
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
        fvcore_flops = float(FlopCountAnalysis(model, dummy).total())
    except Exception:
        pass
    return {
        'params': params,
        'thop_flops': thop_flops,
        'fvcore_flops': fvcore_flops,
    }


def export_config_dict() -> Dict:
    out = {}
    for k in DEFAULT_CONFIG.keys():
        out[k] = copy.deepcopy(getattr(Config, k))
    return out


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_summary_json(summary: Dict, result_dir: Path) -> Path:
    out_path = result_dir / 'summary.json'
    payload = _to_jsonable(summary)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def save_summary_txt(summary: Dict, result_dir: Path) -> Path:
    out_path = result_dir / 'summary.txt'
    payload = _to_jsonable(summary)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('TN5000 Unified Experiment Summary\n')
        f.write('=' * 90 + '\n')
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    return out_path


def save_main_predictions_and_figures(tag: str, metrics: Dict, result_dir: Path) -> None:
    save_predictions_csv(metrics['labels'], metrics['preds'], metrics['probs'], result_dir / f'{tag}_predictions.csv')
    save_logits_npz(metrics['labels'], metrics['logits'], result_dir / f'{tag}_logits.npz')
    save_confusion_matrix(metrics['cm'], f'{tag.upper()} Confusion Matrix', result_dir / f'{tag}_confusion_matrix.png')
    save_roc(metrics['labels'], metrics['probs'], result_dir / f'{tag}_roc_curve.png', title=f'{tag.upper()} ROC Curve')


def main() -> Dict:
    set_seed(Config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_dir = create_result_dir()

    print('=' * 100)
    print('TN5000 Unified ROI Training Script')
    print('=' * 100)
    print(f'[INFO] result_dir      = {result_dir}')
    print(f'[INFO] backbone_module = {Config.backbone_module}')
    print(f'[INFO] backbone_func   = {Config.backbone_func}')
    print(f'[INFO] bbox_expand     = {Config.bbox_expand_ratio}')
    print(f'[INFO] save_by_metric  = {Config.save_by_metric}')
    print(f'[INFO] threshold_mode  = {Config.threshold_selection_mode}')
    print(f'[INFO] seed            = {Config.seed}')
    print(f'[INFO] device          = {device}')
    print('=' * 100)

    datasets = prepare_tn5000_roi_datasets(Config.seed)
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True),
        'val': DataLoader(datasets['val'], batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False),
        'test': DataLoader(datasets['test'], batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False),
    }

    model = ClassificationModel(num_classes=Config.num_classes).to(device)

    class_weights = None
    if Config.use_class_weight:
        class_weights = build_class_weights(datasets['train'], device)
        print(f'[INFO] class_weights = {class_weights.detach().cpu().numpy().tolist()}')

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.label_smoothing)
    optimizer = build_optimizer(model)
    complexity = analyze_model_complexity(model) if Config.measure_model_complexity else {}

    history, best_path, last_path, best_epoch, best_score, improved_ckpts = train_model(
        model, dataloaders, criterion, optimizer, result_dir, device
    )
    curves_path = save_curves(history, result_dir)
    print(f'[SAVE] curves -> {curves_path}')

    best_model = ClassificationModel(num_classes=Config.num_classes).to(device)
    load_checkpoint_to_model(best_model, best_path, device, prefer_ema=True)

    val_raw = evaluate_model(best_model, dataloaders['val'], criterion, device, split_name='val-raw', threshold=0.5, use_hflip_tta=False)
    test_raw = evaluate_model(best_model, dataloaders['test'], criterion, device, split_name='test-raw', threshold=0.5, use_hflip_tta=False)

    threshold_rows = scan_thresholds(val_raw['labels'], val_raw['probs'], Config.threshold_start, Config.threshold_end, Config.threshold_step) if Config.do_threshold_search else []
    threshold_info = choose_best_threshold(threshold_rows) if threshold_rows else {'threshold': 0.5}
    selected_threshold = float(threshold_info['threshold'])
    if threshold_rows:
        save_threshold_scan_csv(threshold_rows, result_dir / 'val_threshold_scan.csv')

    val_metrics = compute_metrics_from_labels_probs(val_raw['labels'], val_raw['probs'], threshold=selected_threshold)
    val_metrics['loss'] = float(val_raw['loss'])
    val_metrics['logits'] = val_raw['logits']
    test_metrics = compute_metrics_from_labels_probs(test_raw['labels'], test_raw['probs'], threshold=selected_threshold)
    test_metrics['loss'] = float(test_raw['loss'])
    test_metrics['logits'] = test_raw['logits']

    save_main_predictions_and_figures('val', val_metrics, result_dir)
    save_main_predictions_and_figures('test', test_metrics, result_dir)

    inference_time = measure_inference_time(best_model, dataloaders['test'], repetitions=Config.inference_time_repetitions) if Config.measure_inference_time else {}
    postprocess = None
    if Config.run_postprocess_variants:
        postprocess = run_postprocess_variants(best_model, dataloaders, criterion, device, result_dir, improved_ckpts, best_path)

    summary = {
        'config': export_config_dict(),
        'paths': {
            'result_dir': str(result_dir),
            'best_model_path': str(best_path),
            'last_model_path': str(last_path),
            'training_curves': str(curves_path),
        },
        'dataset_info': {
            'train_samples': len(datasets['train']),
            'val_samples': len(datasets['val']),
            'test_samples': len(datasets['test']),
            'train_label_counts': datasets['train'].label_counts,
            'val_label_counts': datasets['val'].label_counts,
            'test_label_counts': datasets['test'].label_counts,
        },
        'best_checkpoint': {
            'best_epoch': int(best_epoch),
            'best_score': float(best_score),
            'monitor_metric': Config.save_by_metric,
            'num_improved_ckpts': int(len(improved_ckpts)),
            'improved_ckpts': improved_ckpts,
        },
        'selected_threshold': float(selected_threshold),
        'threshold_selection_mode': Config.threshold_selection_mode,
        'val_metrics': {
            'loss': float(val_metrics['loss']),
            'acc': float(val_metrics['acc']),
            'bal_acc': float(val_metrics['bal_acc']),
            'f1_macro': float(val_metrics['f1_macro']),
            'auc': float(val_metrics['auc']),
            'threshold': float(selected_threshold),
        },
        'test_metrics': {
            'loss': float(test_metrics['loss']),
            'acc': float(test_metrics['acc']),
            'bal_acc': float(test_metrics['bal_acc']),
            'f1_macro': float(test_metrics['f1_macro']),
            'auc': float(test_metrics['auc']),
            'threshold': float(selected_threshold),
        },
        'complexity': complexity,
        'inference_time': inference_time,
        'history': {k: [float(x) for x in v] for k, v in history.items()},
        'postprocess_ablation': postprocess,
    }

    summary_json = save_summary_json(summary, result_dir)
    summary_txt = save_summary_txt(summary, result_dir)
    print(f'[SAVE] summary json -> {summary_json}')
    print(f'[SAVE] summary txt  -> {summary_txt}')
    print(f'[DONE] all outputs saved under: {result_dir}')
    return summary


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
