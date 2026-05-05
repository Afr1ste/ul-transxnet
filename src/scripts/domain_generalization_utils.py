from __future__ import annotations

import bisect
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import fl_tn5000_roi_train_unified as core


PROJECT_ROOT = Path(__file__).resolve().parent
VALID_IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


@dataclass(frozen=True)
class DomainSpec:
    name: str
    root: Path
    kind: str
    split_map: Dict[str, str]


DOMAIN_SPECS: Dict[str, DomainSpec] = {
    "tn5000": DomainSpec(
        name="tn5000",
        root=(PROJECT_ROOT / "TN5000_forReview").resolve(),
        kind="tn5000",
        split_map={
            "train": "train",
            "val": "val",
            "test": "test",
            "trainval": "trainval",
        },
    ),
    "busi": DomainSpec(
        name="busi",
        root=(PROJECT_ROOT / "busi" / "busi_voc_v3_square_consistent").resolve(),
        kind="manifest_voc",
        split_map={
            "train": "fold0_train",
            "val": "fold0_val",
            "test": "test",
            "trainval": "trainval",
        },
    ),
    "aul": DomainSpec(
        name="aul",
        root=(PROJECT_ROOT / "aul" / "aul_voc_roi_v1").resolve(),
        kind="manifest_voc",
        split_map={
            "train": "fold0_train",
            "val": "fold0_val",
            "test": "test",
            "trainval": "trainval",
        },
    ),
}


class ManifestBinaryVOCRoiDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
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

        self.image_dir = self.root_dir / "JPEGImages"
        self.ann_dir = self.root_dir / "Annotations"
        self.split_file = self.root_dir / "ImageSets" / "Main" / f"{split}.txt"
        self.label_manifest = self.root_dir / "manifests" / "label_manifest.csv"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"JPEGImages not found: {self.image_dir}")
        if not self.ann_dir.exists():
            raise FileNotFoundError(f"Annotations not found: {self.ann_dir}")
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        if not self.label_manifest.exists():
            raise FileNotFoundError(f"Label manifest not found: {self.label_manifest}")

        self.label_map = self._load_label_map()
        self.samples = self._build_samples()
        self.label_counts = self._count_labels()
        if not self.samples:
            raise RuntimeError(f"Empty dataset for split={split} under {self.root_dir}")

    def _read_split_ids(self) -> List[str]:
        ids: List[str] = []
        with open(self.split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(line)
        return ids

    def _load_label_map(self) -> Dict[str, int]:
        label_map: Dict[str, int] = {}
        with open(self.label_manifest, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                label_raw = str(row.get("label", "")).strip()
                if not label_raw:
                    continue
                try:
                    label = int(label_raw)
                except ValueError:
                    continue
                key_candidates = [
                    row.get("new_stem", ""),
                    Path(str(row.get("new_filename", "")).strip()).stem,
                    Path(str(row.get("orig_filename", "")).strip()).stem,
                    str(row.get("image_id", "")).strip(),
                    Path(str(row.get("filename", "")).strip()).stem,
                    Path(str(row.get("xml_filename", "")).strip()).stem,
                ]
                for key in key_candidates:
                    key = str(key).strip()
                    if key:
                        label_map[key] = label
        if not label_map:
            raise RuntimeError(f"No labels loaded from manifest: {self.label_manifest}")
        return label_map

    def _find_image_path(self, image_id: str) -> Path:
        for suffix in VALID_IMAGE_SUFFIXES:
            path = self.image_dir / f"{image_id}{suffix}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Image not found for id={image_id}")

    def _parse_xml_bbox(self, xml_path: Path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall("object"):
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = int(float(bnd.findtext("xmin", default="0")))
            ymin = int(float(bnd.findtext("ymin", default="0")))
            xmax = int(float(bnd.findtext("xmax", default="0")))
            ymax = int(float(bnd.findtext("ymax", default="0")))
            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))
        if not boxes:
            return None
        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        return (min(xs1), min(ys1), max(xs2), max(ys2))

    def _build_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        for image_id in self._read_split_ids():
            xml_path = self.ann_dir / f"{image_id}.xml"
            if not xml_path.exists():
                continue
            if image_id not in self.label_map:
                continue
            try:
                image_path = self._find_image_path(image_id)
            except FileNotFoundError:
                continue
            samples.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "xml_path": str(xml_path),
                    "label": int(self.label_map[image_id]),
                    "bbox": self._parse_xml_bbox(xml_path),
                }
            )
        return samples

    def _count_labels(self) -> Dict[int, int]:
        counter = defaultdict(int)
        for sample in self.samples:
            counter[int(sample["label"])] += 1
        return dict(counter)

    def _expand_and_clip_box(self, box, width: int, height: int):
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        if self.use_roi_crop and sample["bbox"] is not None:
            crop_box = self._expand_and_clip_box(sample["bbox"], img.width, img.height)
            if crop_box is not None:
                img = img.crop(crop_box)
            elif not self.use_whole_image_fallback:
                raise RuntimeError(f"Invalid ROI crop for {sample['image_id']}")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(sample["label"])


class CombinedRoiDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset], domain_names: Sequence[str]):
        if len(datasets) != len(domain_names):
            raise ValueError("datasets and domain_names must have the same length")
        self.datasets = list(datasets)
        self.domain_names = list(domain_names)
        self.cumulative_sizes = list(accumulate(len(ds) for ds in self.datasets))
        self.domain_sizes: Dict[str, int] = {}
        self.sample_domain_names: List[str] = []
        label_counts = defaultdict(int)
        for domain_name, dataset in zip(self.domain_names, self.datasets):
            size = len(dataset)
            self.domain_sizes[domain_name] = int(size)
            self.sample_domain_names.extend([domain_name] * size)
            for label, count in getattr(dataset, "label_counts", {}).items():
                label_counts[int(label)] += int(count)
        self.label_counts = dict(label_counts)

    def __len__(self) -> int:
        if not self.cumulative_sizes:
            return 0
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        dataset_start = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = idx - dataset_start
        return self.datasets[dataset_idx][sample_idx]


def configure_core_mainline(
    output_root: str | Path,
    run_name: str,
    seed: int,
    batch_size: int = 8,
    num_epochs: int = 120,
    num_workers: int = 0,
) -> None:
    core.reset_config()
    core.Config.output_root = str(output_root)
    core.Config.run_name = str(run_name)
    core.Config.seed = int(seed)
    core.Config.batch_size = int(batch_size)
    core.Config.num_epochs = int(num_epochs)
    core.Config.num_workers = int(num_workers)

    core.Config.backbone_module = "models.transxnetggg"
    core.Config.backbone_func = "transxnet_t"
    core.Config.backbone_out_dim = 1000
    core.Config.head_hidden_dim = 512
    core.Config.input_size = 256
    core.Config.num_classes = 2
    core.Config.class_names = ["0", "1"]

    core.Config.use_roi_crop = True
    core.Config.bbox_expand_ratio = 0.30
    core.Config.min_crop_size = 64
    core.Config.use_whole_image_fallback = True

    core.Config.backbone_lr = 5e-5
    core.Config.head_lr = 1.5e-4
    core.Config.weight_decay = 1e-4
    core.Config.dropout = 0.30
    core.Config.label_smoothing = 0.00
    core.Config.early_stopping_patience = 12
    core.Config.early_stopping_min_delta = 1e-4
    core.Config.save_by_metric = "auc"

    core.Config.optimizer_name = "adamw"
    core.Config.use_amp = True
    core.Config.max_grad_norm = 3.0
    core.Config.freeze_backbone = False
    core.Config.freeze_backbone_norm = False

    core.Config.lr_schedule = "cosine_floor"
    core.Config.lr_warmup_epochs = 5
    core.Config.lr_min_ratio = 0.25

    core.Config.use_class_weight = False
    core.Config.use_manual_class_weights = False
    core.Config.manual_class_weights = [1.0, 1.0]

    core.Config.use_ema = True
    core.Config.ema_decay = 0.9995

    core.Config.do_threshold_search = True
    core.Config.threshold_start = 0.10
    core.Config.threshold_end = 0.95
    core.Config.threshold_step = 0.01
    core.Config.threshold_selection_mode = "bal_acc"
    core.Config.min_recall_1 = 0.90

    core.Config.run_postprocess_variants = False
    core.Config.use_hflip_tta = True
    core.Config.use_temperature_scaling = True
    core.Config.ensemble_topk = 3
    core.Config.save_all_improved_checkpoints = True
    core.Config.max_saved_improved_ckpts = 8

    core.Config.measure_model_complexity = False
    core.Config.measure_inference_time = False
    core.Config.inference_time_repetitions = 20

    core.Config.best_model_name = "best_model.pth"
    core.Config.final_model_name = "last_model.pth"


def get_domain_spec(domain_name: str) -> DomainSpec:
    key = str(domain_name).strip().lower()
    if key not in DOMAIN_SPECS:
        raise KeyError(f"Unknown domain: {domain_name}")
    return DOMAIN_SPECS[key]


def build_domain_dataset(domain_name: str, split_key: str, transform=None) -> Dataset:
    spec = get_domain_spec(domain_name)
    if split_key not in spec.split_map:
        raise KeyError(f"Split key '{split_key}' not defined for domain '{domain_name}'")
    split_name = spec.split_map[split_key]
    if spec.kind == "tn5000":
        return core.TN5000ROIDataset(
            str(spec.root),
            split_name,
            transform,
            use_roi_crop=core.Config.use_roi_crop,
            bbox_expand_ratio=core.Config.bbox_expand_ratio,
            min_crop_size=core.Config.min_crop_size,
            use_whole_image_fallback=core.Config.use_whole_image_fallback,
        )
    if spec.kind == "manifest_voc":
        return ManifestBinaryVOCRoiDataset(
            str(spec.root),
            split_name,
            transform,
            use_roi_crop=core.Config.use_roi_crop,
            bbox_expand_ratio=core.Config.bbox_expand_ratio,
            min_crop_size=core.Config.min_crop_size,
            use_whole_image_fallback=core.Config.use_whole_image_fallback,
        )
    raise ValueError(f"Unsupported domain kind: {spec.kind}")


def build_multi_domain_dataset(domain_names: Sequence[str], split_key: str, transform=None) -> Dataset:
    names = [str(name).strip().lower() for name in domain_names]
    datasets = [build_domain_dataset(name, split_key, transform=transform) for name in names]
    if len(datasets) == 1:
        return datasets[0]
    return CombinedRoiDataset(datasets, names)


def build_domain_balanced_sampler(dataset: Dataset) -> Optional[WeightedRandomSampler]:
    if not isinstance(dataset, CombinedRoiDataset):
        return None
    weights = [
        1.0 / max(float(dataset.domain_sizes[domain_name]), 1.0)
        for domain_name in dataset.sample_domain_names
    ]
    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights), replacement=True)


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    domain_balanced_train: bool = True,
) -> Dict[str, DataLoader]:
    train_sampler = build_domain_balanced_sampler(train_dataset) if domain_balanced_train else None
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=core.Config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=core.Config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=core.Config.batch_size,
            shuffle=False,
            num_workers=core.Config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=core.Config.batch_size,
            shuffle=False,
            num_workers=core.Config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        ),
    }


def dataset_summary(dataset: Dataset) -> Dict:
    info = {
        "num_samples": int(len(dataset)),
        "label_counts": {
            str(k): int(v) for k, v in getattr(dataset, "label_counts", {}).items()
        },
    }
    if isinstance(dataset, CombinedRoiDataset):
        info["domain_sizes"] = {k: int(v) for k, v in dataset.domain_sizes.items()}
    return info


def write_json(path: str | Path, payload: Mapping) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_rows_csv(path: str | Path, rows: Sequence[Mapping]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_eval_variant_defs() -> List[Dict]:
    return [
        {
            "tag": "single_raw",
            "topk": 1,
            "use_hflip_tta": False,
            "use_temperature_scaling": False,
        },
        {
            "tag": "single_tta",
            "topk": 1,
            "use_hflip_tta": True,
            "use_temperature_scaling": False,
        },
        {
            "tag": "single_tta_temp",
            "topk": 1,
            "use_hflip_tta": True,
            "use_temperature_scaling": True,
        },
        {
            "tag": "ens_tta_temp",
            "topk": core.Config.ensemble_topk,
            "use_hflip_tta": True,
            "use_temperature_scaling": True,
        },
    ]


def _select_recommended_variant(source_rows: Sequence[Mapping]) -> Optional[str]:
    if not source_rows:
        return None
    ranked = sorted(
        source_rows,
        key=lambda row: (
            float(row["source_val_bal_acc"]),
            float(row["source_val_f1_macro"]),
            float(row["source_val_auc"]),
            -abs(float(row["selected_threshold"]) - 0.5),
        ),
        reverse=True,
    )
    return str(ranked[0]["variant"])


@torch.no_grad()
def evaluate_variants_with_source_val(
    model: nn.Module,
    candidate_ckpt_paths: Sequence[str | Path],
    source_val_loader: DataLoader,
    target_loaders: Mapping[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
    result_dir: str | Path,
) -> Dict:
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_ckpts = [str(Path(path)) for path in candidate_ckpt_paths if Path(path).exists()]
    if not ordered_ckpts:
        raise FileNotFoundError("No valid checkpoint paths were provided for evaluation.")

    target_loaders = {
        str(target_name).strip().lower(): loader for target_name, loader in target_loaders.items()
    }
    target_names = list(target_loaders.keys())
    variants = build_eval_variant_defs()

    source_variant_rows: List[Dict] = []
    target_rows: List[Dict] = []

    for variant in variants:
        requested_topk = max(int(variant["topk"]), 1)
        used_ckpts = ordered_ckpts[:requested_topk]
        if not used_ckpts:
            continue

        source_val_labels = None
        source_val_logits_sum = None
        target_labels: Dict[str, np.ndarray] = {}
        target_logits_sums: Dict[str, np.ndarray] = {}

        for ckpt_path in used_ckpts:
            core.load_checkpoint_to_model(model, Path(ckpt_path), device, prefer_ema=True)
            source_eval = core.evaluate_model(
                model,
                source_val_loader,
                criterion,
                device,
                split_name=f'{variant["tag"]}-source-val',
                threshold=0.5,
                use_hflip_tta=bool(variant["use_hflip_tta"]),
            )
            if source_val_labels is None:
                source_val_labels = source_eval["labels"]
                source_val_logits_sum = source_eval["logits"]
            else:
                if not np.array_equal(source_val_labels, source_eval["labels"]):
                    raise RuntimeError("Source validation label order changed across checkpoints.")
                source_val_logits_sum += source_eval["logits"]

            for target_name in target_names:
                target_eval = core.evaluate_model(
                    model,
                    target_loaders[target_name],
                    criterion,
                    device,
                    split_name=f'{variant["tag"]}-{target_name}',
                    threshold=0.5,
                    use_hflip_tta=bool(variant["use_hflip_tta"]),
                )
                if target_name not in target_labels:
                    target_labels[target_name] = target_eval["labels"]
                    target_logits_sums[target_name] = target_eval["logits"]
                else:
                    if not np.array_equal(target_labels[target_name], target_eval["labels"]):
                        raise RuntimeError(
                            f"Target label order changed for domain={target_name} across checkpoints."
                        )
                    target_logits_sums[target_name] += target_eval["logits"]

        source_val_logits = source_val_logits_sum / float(len(used_ckpts))
        temperature = 1.0
        source_val_nll_before = core.compute_nll_from_logits(
            source_val_logits, source_val_labels, temperature=1.0
        )
        source_val_nll_after = source_val_nll_before
        if variant["use_temperature_scaling"]:
            temperature = core.fit_temperature_on_val(source_val_logits, source_val_labels, device)
            source_val_nll_after = core.compute_nll_from_logits(
                source_val_logits, source_val_labels, temperature=temperature
            )

        source_val_probs = core.logits_to_probs(source_val_logits, temperature=temperature)
        threshold_rows = core.scan_thresholds(
            source_val_labels,
            source_val_probs,
            core.Config.threshold_start,
            core.Config.threshold_end,
            core.Config.threshold_step,
        )
        core.save_threshold_scan_csv(
            threshold_rows,
            out_dir / f'{variant["tag"]}_source_val_threshold_scan.csv',
        )
        best_threshold_info = core.choose_best_threshold(threshold_rows)
        selected_threshold = float(best_threshold_info["threshold"])
        source_metrics = core.compute_metrics_from_labels_probs(
            source_val_labels,
            source_val_probs,
            threshold=selected_threshold,
        )

        source_variant_row = {
            "variant": variant["tag"],
            "num_ckpts_used": int(len(used_ckpts)),
            "use_hflip_tta": bool(variant["use_hflip_tta"]),
            "use_temperature_scaling": bool(variant["use_temperature_scaling"]),
            "temperature": float(temperature),
            "selected_threshold": float(selected_threshold),
            "source_val_acc": float(source_metrics["acc"]),
            "source_val_bal_acc": float(source_metrics["bal_acc"]),
            "source_val_f1_macro": float(source_metrics["f1_macro"]),
            "source_val_auc": float(source_metrics["auc"]),
            "source_val_nll_before": float(source_val_nll_before),
            "source_val_nll_after": float(source_val_nll_after),
            "ckpts_used": " | ".join(used_ckpts),
        }
        source_variant_rows.append(source_variant_row)

        for target_name in target_names:
            target_logits = target_logits_sums[target_name] / float(len(used_ckpts))
            target_probs = core.logits_to_probs(target_logits, temperature=temperature)
            target_metrics = core.compute_metrics_from_labels_probs(
                target_labels[target_name],
                target_probs,
                threshold=selected_threshold,
            )
            target_row = {
                "variant": variant["tag"],
                "target_domain": target_name,
                "num_ckpts_used": int(len(used_ckpts)),
                "use_hflip_tta": bool(variant["use_hflip_tta"]),
                "use_temperature_scaling": bool(variant["use_temperature_scaling"]),
                "temperature": float(temperature),
                "selected_threshold": float(selected_threshold),
                "source_val_acc": float(source_metrics["acc"]),
                "source_val_bal_acc": float(source_metrics["bal_acc"]),
                "source_val_f1_macro": float(source_metrics["f1_macro"]),
                "source_val_auc": float(source_metrics["auc"]),
                "target_acc": float(target_metrics["acc"]),
                "target_bal_acc": float(target_metrics["bal_acc"]),
                "target_f1_macro": float(target_metrics["f1_macro"]),
                "target_auc": float(target_metrics["auc"]),
                "target_nll": float(
                    core.compute_nll_from_logits(
                        target_logits,
                        target_labels[target_name],
                        temperature=temperature,
                    )
                ),
                "ckpts_used": " | ".join(used_ckpts),
            }
            target_rows.append(target_row)

    recommended_variant = _select_recommended_variant(source_variant_rows)
    write_rows_csv(out_dir / "source_val_variant_metrics.csv", source_variant_rows)
    write_rows_csv(out_dir / "target_variant_metrics.csv", target_rows)
    write_json(
        out_dir / "evaluation_summary.json",
        {
            "recommended_variant_by_source_val": recommended_variant,
            "source_val_variants": source_variant_rows,
            "target_rows": target_rows,
        },
    )
    return {
        "recommended_variant_by_source_val": recommended_variant,
        "source_val_variants": source_variant_rows,
        "target_rows": target_rows,
    }
