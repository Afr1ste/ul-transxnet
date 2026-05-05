#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BUSI 最终论文对比实验：5 models x 5 folds 批跑脚本

特点
----
1) 统一跑 5 个模型：
   - Ours (transxnet_t)
   - ResNet50
   - EfficientNet-B0
   - MobileNetV3-Large
   - Swin-T
2) 统一 5-fold 协议，固定同一套 BUSI fold 文件；
3) 自动汇总：
   - 每个 fold 的原始指标
   - 每个模型的 mean / std
   - 每个模型的 val OOF 结果
   - 每个模型的 test 5-fold ensemble 结果
   - 5 模型 ROC 对比所需 CSV
4) 自动统计刺头样本：
   - 单 run 的 val / test hardcases 由训练脚本直接导出
   - 跨 fold、按模型聚合 test hardcases
   - 按模型导出 val OOF / test ensemble hardcases
"""

import csv
import codecs
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "fl_busi_roi_compare_5fold.py"
OUTPUT_ROOT = "busi_roi_runs_compare_5models_5fold"
LOG_ROOT = "busi_compare_5models_5fold_logs"
DATA_ROOT = PROJECT_ROOT / "busi" / "busi_voc_v3_square_consistent"

GLOBAL_SEED = 17
FOLDS = [0, 1, 2, 3, 4]
CONTINUE_ON_ERROR = False

TEST_HARDCASE_BY_CONFIG_MIN_WRONG_RUNS = 3
TEST_HARDCASE_BY_CONFIG_MIN_WRONG_RATE = 0.60
TEST_HARDCASE_OVERALL_MIN_WRONG_RUNS = 10
TEST_HARDCASE_OVERALL_MIN_WRONG_RATE = 0.40

COMMON = dict(
    input_size=256,
    backbone_lr=5e-5,
    head_lr=1.5e-4,
    bbox_expand_ratio=0.30,
    num_epochs=200,
    early_stopping_patience=36,
    weight_decay=1e-4,
    dropout=0.30,
    label_smoothing=0.00,
    save_by_metric="auc",
    threshold_selection_mode="bal_acc",
    lr_schedule="cosine_floor",
    lr_warmup_epochs=5,
    lr_min_ratio=0.25,
    use_ema=True,
    ema_decay=0.9995,
    use_class_weight=True,
    use_manual_class_weights=False,
    manual_class_weights="1.0,1.0",
    use_hflip_tta=True,
    use_temperature_scaling=True,
    ensemble_topk=3,
)

BASE_CONFIGS = [
    dict(
        name="OURS_autoCW_exp030",
        display_name="Ours",
        model_family="custom",
        backbone_name="transxnet_t",
        backbone_module="models.transxnetggg",
        backbone_func="transxnet_t",
        backbone_out_dim=1000,
        **COMMON,
    ),
    dict(
        name="RESNET50_autoCW_exp030",
        display_name="ResNet50",
        model_family="timm",
        backbone_name="resnet50",
        backbone_module="",
        backbone_func="",
        backbone_out_dim=0,
        **COMMON,
    ),
    dict(
        name="EFFB0_autoCW_exp030",
        display_name="EfficientNet-B0",
        model_family="timm",
        backbone_name="efficientnet_b0",
        backbone_module="",
        backbone_func="",
        backbone_out_dim=0,
        **COMMON,
    ),
    dict(
        name="MBV3_autoCW_exp030",
        display_name="MobileNetV3-Large",
        model_family="timm",
        backbone_name="mobilenetv3_large_100",
        backbone_module="",
        backbone_func="",
        backbone_out_dim=0,
        **COMMON,
    ),
    dict(
        name="SWINT_autoCW_exp030",
        display_name="Swin-T",
        model_family="timm",
        backbone_name="swin_tiny_patch4_window7_224",
        backbone_module="",
        backbone_func="",
        backbone_out_dim=0,
        **COMMON,
    ),
]


def build_experiments():
    exps = []
    for cfg in BASE_CONFIGS:
        for fold_idx in FOLDS:
            item = dict(cfg)
            item["name"] = f"{cfg['name']}_f{fold_idx}"
            item["base_name"] = cfg["name"]
            item["fold_idx"] = fold_idx
            item["seed"] = GLOBAL_SEED
            exps.append(item)
    return exps


EXPERIMENTS = build_experiments()


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    final_fieldnames = list(fieldnames)
    seen = set(final_fieldnames)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                final_fieldnames.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def stream_subprocess(cmd, cwd: Path, log_path: Path):
    start = time.time()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    cmd = list(cmd)
    if len(cmd) >= 1 and "-u" not in cmd[1:2]:
        cmd.insert(1, "-u")

    with log_path.open("w", encoding="utf-8", errors="replace", newline="") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            env=env,
        )
        assert proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                if proc.poll() is not None:
                    break
                continue
            text_chunk = decoder.decode(chunk)
            if text_chunk:
                sys.stderr.write(text_chunk)
                sys.stderr.flush()
                logf.write(text_chunk)
                logf.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            sys.stderr.write(tail)
            sys.stderr.flush()
            logf.write(tail)
            logf.flush()
        proc.wait()
        elapsed = time.time() - start
        return proc.returncode, elapsed


def list_subdirs(parent: Path):
    if not parent.exists():
        return []
    return [p for p in parent.iterdir() if p.is_dir()]


def detect_new_run_dir(parent: Path, before_names: set[str]):
    subdirs = list_subdirs(parent)
    new_dirs = [p for p in subdirs if p.name not in before_names]
    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)
    if subdirs:
        return max(subdirs, key=lambda p: p.stat().st_mtime)
    return None


def parse_summary(summary_path: Path):
    text = summary_path.read_text(encoding="utf-8")

    def get_float(pattern, default=np.nan):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    def get_int(pattern, default=-1):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else default

    def get_str(pattern, default=""):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    info = {
        "best_epoch_by_runtime": get_int(r"best_epoch_by_runtime=(\d+)", -1),
        "best_score_by_runtime": get_float(r"best_score_by_runtime=([0-9.]+)", np.nan),
        "selected_threshold": get_float(r"Selected threshold:\s*([0-9.]+)", np.nan),
        "temperature": get_float(r"temperature:\s*([0-9.]+)", np.nan),
        "val_nll_before_temp": get_float(r"val_nll_before_temp:\s*([0-9.]+)", np.nan),
        "val_nll_after_temp": get_float(r"val_nll_after_temp:\s*([0-9.]+)", np.nan),
        "use_hflip_tta": get_str(r"use_hflip_tta:\s*(True|False)", ""),
        "use_temperature_scaling": get_str(r"use_temperature_scaling:\s*(True|False)", ""),
        "ensemble_topk_requested": get_int(r"ensemble_topk_requested:\s*(\d+)", -1),
        "ensemble_ckpts_used_raw": get_str(r"ensemble_ckpts_used:\s*(.+)", ""),
        "param_tag": get_str(r"param_tag:\s*(.+)", ""),
        "model_family": get_str(r"model_family:\s*(.+)", ""),
        "backbone_name": get_str(r"backbone_name:\s*(.+)", ""),
        "backbone_module": get_str(r"backbone_module:\s*(.*)", ""),
        "backbone_func": get_str(r"backbone_func:\s*(.*)", ""),
        "backbone_lr": get_float(r"backbone_lr:\s*([0-9.eE+-]+)", np.nan),
        "head_lr": get_float(r"head_lr:\s*([0-9.eE+-]+)", np.nan),
        "use_ema": get_str(r"use_ema:\s*(True|False)", ""),
        "ema_decay": get_float(r"ema_decay:\s*([0-9.eE+-]+)", np.nan),
        "use_class_weight": get_str(r"use_class_weight:\s*(True|False)", ""),
    }
    info["ensemble_ckpts_used"] = info["ensemble_ckpts_used_raw"].count(".pth")
    return info


def load_predictions_csv(csv_path: Path):
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            'image_id': str(r['image_id']),
            'image_path': str(r.get('image_path', '')),
            'true_label': int(r['true_label']),
            'pred_label': int(r['pred_label']),
            'prob_class0': float(r.get('prob_class0', 1.0 - float(r['prob_class1']))),
            'prob_class1': float(r['prob_class1']),
            'threshold': float(r.get('threshold', 0.5)),
            'is_wrong': int(r.get('is_wrong', int(r['true_label']) != int(r['pred_label']))),
            'wrong_conf': float(r.get('wrong_conf', 0.0)),
            'margin_from_05': float(r.get('margin_from_05', abs(float(r['prob_class1']) - 0.5))),
        })
    return out


def compute_metrics_from_predictions(rows, threshold: Optional[float] = None):
    y_true = np.array([int(r['true_label']) for r in rows], dtype=int)
    y_prob = np.array([float(r['prob_class1']) for r in rows], dtype=float)
    thr = float(threshold) if threshold is not None else float(rows[0].get('threshold', 0.5))
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, thresholds, roc_auc = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.nan, np.nan]), np.nan
    return {
        'acc': float(acc),
        'bal_acc': float(bal_acc),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'auc': float(roc_auc),
        'recall_0': float(recall_0),
        'recall_1': float(recall_1),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'threshold': thr,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': thresholds,
    }


def scan_thresholds(rows, start=0.10, end=0.95, step=0.01):
    results = []
    thr = start
    while thr <= end + 1e-12:
        metrics = compute_metrics_from_predictions(rows, threshold=round(thr, 4))
        results.append({
            'threshold': round(thr, 4),
            'bal_acc': metrics['bal_acc'],
            'f1_macro': metrics['f1_macro'],
            'acc': metrics['acc'],
            'auc': metrics['auc'],
            'recall_0': metrics['recall_0'],
            'recall_1': metrics['recall_1'],
            'tn': metrics['tn'], 'fp': metrics['fp'], 'fn': metrics['fn'], 'tp': metrics['tp'],
        })
        thr += step
    return results


def choose_best_threshold(results):
    results = sorted(results, key=lambda x: (x['bal_acc'], x['f1_macro'], -abs(x['threshold'] - 0.5)), reverse=True)
    return results[0]


def sort_run_rows(rows):
    return sorted(rows, key=lambda x: (x.get('test_auc', -1), x.get('test_bal_acc', -1), x.get('test_f1_macro', -1), x.get('val_auc', -1)), reverse=True)


def aggregate_rows(rows):
    by_cfg = {}
    for r in rows:
        if r.get('status') != 'ok':
            continue
        by_cfg.setdefault(r['base_name'], []).append(r)

    metric_cols = [
        'val_acc', 'val_bal_acc', 'val_f1_macro', 'val_auc', 'val_recall_0', 'val_recall_1',
        'test_acc', 'test_bal_acc', 'test_f1_macro', 'test_auc', 'test_recall_0', 'test_recall_1',
        'selected_threshold', 'best_score_by_runtime',
    ]

    agg_rows = []
    for cfg_name, group in by_cfg.items():
        base = group[0]
        group = sorted(group, key=lambda g: g['fold_idx'])
        row = {
            'base_name': cfg_name,
            'display_name': base.get('display_name_cfg', ''),
            'n_folds_finished': len(group),
            'folds': ','.join(str(g['fold_idx']) for g in group),
            'global_seed': base['seed'],
            'model_family_cfg': base['model_family_cfg'],
            'backbone_name_cfg': base['backbone_name_cfg'],
            'backbone_module_cfg': base['backbone_module_cfg'],
            'backbone_func_cfg': base['backbone_func_cfg'],
            'backbone_lr': base['backbone_lr_cfg'],
            'head_lr': base['head_lr_cfg'],
            'use_ema': base['use_ema_cfg'],
            'ema_decay': base['ema_decay_cfg'],
            'use_class_weight': base['use_class_weight_cfg'],
            'use_hflip_tta': base.get('use_hflip_tta', ''),
            'use_temperature_scaling': base.get('use_temperature_scaling', ''),
            'ensemble_topk_requested': base.get('ensemble_topk_requested', -1),
        }
        for m in metric_cols:
            vals = np.array([g[m] for g in group], dtype=float)
            row[f'{m}_mean'] = float(np.mean(vals))
            row[f'{m}_std'] = float(np.std(vals, ddof=0))
        agg_rows.append(row)

    agg_rows = sorted(agg_rows, key=lambda x: (x['test_auc_mean'], x['test_bal_acc_mean'], x['test_f1_macro_mean'], x['val_auc_mean']), reverse=True)
    return agg_rows


def build_run_row(exp: dict, run_dir: Path):
    summary = parse_summary(run_dir / 'summary.txt')
    val_rows = load_predictions_csv(run_dir / 'val_predictions.csv')
    test_rows = load_predictions_csv(run_dir / 'test_predictions.csv')
    val_metrics = compute_metrics_from_predictions(val_rows)
    test_metrics = compute_metrics_from_predictions(test_rows)

    row = {
        'name': exp['name'],
        'base_name': exp['base_name'],
        'display_name_cfg': exp.get('display_name', ''),
        'fold_idx': exp['fold_idx'],
        'seed': exp['seed'],
        'run_dir': str(run_dir),
        'status': 'ok',
        'model_family_cfg': exp.get('model_family', ''),
        'backbone_name_cfg': exp.get('backbone_name', ''),
        'backbone_module_cfg': exp.get('backbone_module', ''),
        'backbone_func_cfg': exp.get('backbone_func', ''),
        'backbone_lr_cfg': exp['backbone_lr'],
        'head_lr_cfg': exp['head_lr'],
        'use_ema_cfg': exp['use_ema'],
        'ema_decay_cfg': exp['ema_decay'],
        'use_class_weight_cfg': exp['use_class_weight'],
        'param_tag': summary['param_tag'],
        'model_family': summary['model_family'],
        'backbone_name': summary['backbone_name'],
        'backbone_module': summary['backbone_module'],
        'backbone_func': summary['backbone_func'],
        'backbone_lr': summary['backbone_lr'],
        'head_lr': summary['head_lr'],
        'use_ema': summary['use_ema'],
        'ema_decay': summary['ema_decay'],
        'use_hflip_tta': summary['use_hflip_tta'],
        'use_temperature_scaling': summary['use_temperature_scaling'],
        'ensemble_topk_requested': summary['ensemble_topk_requested'],
        'ensemble_ckpts_used': summary['ensemble_ckpts_used'],
        'best_epoch_by_runtime': summary['best_epoch_by_runtime'],
        'best_score_by_runtime': summary['best_score_by_runtime'],
        'selected_threshold': summary['selected_threshold'],
        'val_acc': val_metrics['acc'],
        'val_bal_acc': val_metrics['bal_acc'],
        'val_f1_macro': val_metrics['f1_macro'],
        'val_auc': val_metrics['auc'],
        'val_recall_0': val_metrics['recall_0'],
        'val_recall_1': val_metrics['recall_1'],
        'val_tn': val_metrics['tn'], 'val_fp': val_metrics['fp'], 'val_fn': val_metrics['fn'], 'val_tp': val_metrics['tp'],
        'test_acc': test_metrics['acc'],
        'test_bal_acc': test_metrics['bal_acc'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_auc': test_metrics['auc'],
        'test_recall_0': test_metrics['recall_0'],
        'test_recall_1': test_metrics['recall_1'],
        'test_tn': test_metrics['tn'], 'test_fp': test_metrics['fp'], 'test_fn': test_metrics['fn'], 'test_tp': test_metrics['tp'],
    }
    return row


def save_incremental_outputs(log_dir: Path, statuses, run_rows):
    write_csv(log_dir / 'batch_status.csv', statuses, fieldnames=['name', 'base_name', 'fold_idx', 'seed', 'return_code', 'elapsed_sec', 'log_file', 'run_dir', 'status'])
    if run_rows:
        fieldnames = list(run_rows[0].keys())
        write_csv(log_dir / 'all_runs_metrics.csv', run_rows, fieldnames)
        write_csv(log_dir / 'all_runs_metrics_sorted.csv', sort_run_rows(run_rows), fieldnames)
        agg_rows = aggregate_rows(run_rows)
        if agg_rows:
            write_csv(log_dir / 'aggregate_by_config.csv', agg_rows, list(agg_rows[0].keys()))
            write_paper_main_table(log_dir / 'paper_main_table.csv', agg_rows)


def write_paper_main_table(path: Path, agg_rows: List[Dict]):
    rows = []
    for r in agg_rows:
        rows.append({
            'Model': r['display_name'] or r['base_name'],
            'AUC': f"{r['test_auc_mean']:.4f} ± {r['test_auc_std']:.4f}",
            'BalAcc': f"{r['test_bal_acc_mean']:.4f} ± {r['test_bal_acc_std']:.4f}",
            'F1_macro': f"{r['test_f1_macro_mean']:.4f} ± {r['test_f1_macro_std']:.4f}",
            'Acc': f"{r['test_acc_mean']:.4f} ± {r['test_acc_std']:.4f}",
        })
    write_csv(path, rows, fieldnames=['Model', 'AUC', 'BalAcc', 'F1_macro', 'Acc'])


def write_rows_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def aggregate_one_scope(records, key_name: str, min_wrong_runs: int, min_wrong_rate: float):
    merged = {}
    for rec in records:
        key = rec[key_name]
        x = merged.get(key)
        if x is None:
            x = {
                key_name: key,
                'true_label': int(rec['true_label']),
                'seen_runs': 0,
                'wrong_runs': 0,
                'correct_runs': 0,
                'avg_prob1_sum': 0.0,
                'avg_wrong_conf_sum': 0.0,
                'wrong_conf_max': 0.0,
                'wrong_run_names': [],
                'correct_run_names': [],
                'base_names': set(),
                'folds': set(),
            }
            merged[key] = x
        x['seen_runs'] += 1
        x['wrong_runs'] += int(rec['is_wrong'])
        x['correct_runs'] += 1 - int(rec['is_wrong'])
        x['avg_prob1_sum'] += float(rec['prob_class1'])
        x['base_names'].add(str(rec['base_name']))
        x['folds'].add(str(rec['fold_idx']))
        if int(rec['is_wrong']):
            x['avg_wrong_conf_sum'] += float(rec['wrong_conf'])
            x['wrong_conf_max'] = max(x['wrong_conf_max'], float(rec['wrong_conf']))
            x['wrong_run_names'].append(str(rec['run_name']))
        else:
            x['correct_run_names'].append(str(rec['run_name']))

    rows = []
    for x in merged.values():
        seen = max(int(x['seen_runs']), 1)
        wrong = int(x['wrong_runs'])
        rows.append({
            key_name: x[key_name],
            'true_label': int(x['true_label']),
            'seen_runs': seen,
            'wrong_runs': wrong,
            'correct_runs': int(x['correct_runs']),
            'wrong_rate': wrong / seen,
            'always_wrong': int(wrong == seen),
            'avg_prob1': x['avg_prob1_sum'] / seen,
            'avg_wrong_conf': (x['avg_wrong_conf_sum'] / wrong) if wrong > 0 else np.nan,
            'wrong_conf_max': float(x['wrong_conf_max']),
            'base_names': '|'.join(sorted(x['base_names'])),
            'folds': '|'.join(sorted(x['folds'])),
            'wrong_run_names': '|'.join(x['wrong_run_names']),
            'correct_run_names': '|'.join(x['correct_run_names']),
        })
    rows = sorted(rows, key=lambda r: (r['always_wrong'], r['wrong_rate'], r['wrong_runs'], r['wrong_conf_max']), reverse=True)
    hard_rows = [r for r in rows if int(r['wrong_runs']) >= int(min_wrong_runs) and float(r['wrong_rate']) >= float(min_wrong_rate)]
    return rows, hard_rows


def aggregate_test_hardcases(log_dir: Path, run_rows: List[Dict]):
    if not run_rows:
        return
    overall_records = []
    by_config_records = []
    for row in run_rows:
        pred_csv = Path(row['run_dir']) / 'test_predictions.csv'
        if not pred_csv.exists():
            continue
        detailed = load_predictions_csv(pred_csv)
        for rec in detailed:
            item = dict(rec)
            item['run_name'] = row['name']
            item['base_name'] = row['base_name']
            item['fold_idx'] = row['fold_idx']
            overall_records.append(item)
            cfg_item = dict(item)
            cfg_item['image_key'] = f"{row['base_name']}::{item['image_id']}"
            by_config_records.append(cfg_item)
    if not overall_records:
        return

    overall_fields = ['image_id', 'true_label', 'seen_runs', 'wrong_runs', 'correct_runs', 'wrong_rate', 'always_wrong', 'avg_prob1', 'avg_wrong_conf', 'wrong_conf_max', 'base_names', 'folds', 'wrong_run_names', 'correct_run_names']
    cfg_fields = ['base_name', 'image_id', 'true_label', 'seen_runs', 'wrong_runs', 'correct_runs', 'wrong_rate', 'always_wrong', 'avg_prob1', 'avg_wrong_conf', 'wrong_conf_max', 'base_names', 'folds', 'wrong_run_names', 'correct_run_names']

    overall_rows, overall_hard = aggregate_one_scope(overall_records, 'image_id', TEST_HARDCASE_OVERALL_MIN_WRONG_RUNS, TEST_HARDCASE_OVERALL_MIN_WRONG_RATE)
    write_rows_csv(log_dir / 'test_hardcases_overall_all.csv', overall_rows, overall_fields)
    write_rows_csv(log_dir / 'test_hardcases_overall_hard.csv', overall_hard, overall_fields)

    cfg_rows, cfg_hard = aggregate_one_scope(by_config_records, 'image_key', TEST_HARDCASE_BY_CONFIG_MIN_WRONG_RUNS, TEST_HARDCASE_BY_CONFIG_MIN_WRONG_RATE)
    def _expand_cfg_rows(rows):
        out = []
        for r in rows:
            rr = dict(r)
            base_name, image_id = str(rr.pop('image_key')).split('::', 1)
            rr['base_name'] = base_name
            rr['image_id'] = image_id
            out.append(rr)
        return out
    cfg_rows_out = _expand_cfg_rows(cfg_rows)
    cfg_hard_out = _expand_cfg_rows(cfg_hard)
    write_rows_csv(log_dir / 'test_hardcases_by_config_all.csv', cfg_rows_out, cfg_fields)
    write_rows_csv(log_dir / 'test_hardcases_by_config_hard.csv', cfg_hard_out, cfg_fields)


def save_threshold_scan_csv(results, out_path: Path):
    fields = ['threshold', 'bal_acc', 'f1_macro', 'acc', 'auc', 'recall_0', 'recall_1', 'tn', 'fp', 'fn', 'tp']
    write_rows_csv(out_path, results, fields)


def save_predictions_with_metrics(rows: List[Dict], threshold: float, out_path: Path):
    out_rows = []
    for r in rows:
        prob1 = float(r['prob_class1'])
        pred = int(prob1 >= threshold)
        out_rows.append({
            'image_id': r['image_id'],
            'image_path': r.get('image_path', ''),
            'true_label': int(r['true_label']),
            'pred_label': pred,
            'prob_class0': 1.0 - prob1,
            'prob_class1': prob1,
            'threshold': threshold,
            'is_wrong': int(int(r['true_label']) != pred),
            'wrong_conf': prob1 if pred == 1 else (1.0 - prob1),
            'margin_from_05': abs(prob1 - 0.5),
        })
    write_csv(out_path, out_rows, fieldnames=list(out_rows[0].keys()) if out_rows else ['image_id'])


def save_roc_compare_rows(path: Path, roc_rows: List[Dict]):
    write_csv(path, roc_rows, fieldnames=['model', 'split', 'fpr', 'tpr', 'threshold'])


def plot_roc_compare(roc_rows: List[Dict], split_name: str, out_path: Path):
    fig = plt.figure(figsize=(6.5, 5.5))
    models = sorted({r['model'] for r in roc_rows if r['split'] == split_name})
    for model in models:
        xs = [float(r['fpr']) for r in roc_rows if r['split'] == split_name and r['model'] == model]
        ys = [float(r['tpr']) for r in roc_rows if r['split'] == split_name and r['model'] == model]
        plt.plot(xs, ys, label=model)
    plt.plot([0, 1], [0, 1], '--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Compare ({split_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def build_model_level_artifacts(log_dir: Path, run_rows: List[Dict]):
    if not run_rows:
        return
    model_level_root = log_dir / 'model_level'
    model_level_root.mkdir(parents=True, exist_ok=True)
    roc_compare_rows = []
    summary_rows = []

    by_cfg = {}
    for r in run_rows:
        if r.get('status') == 'ok':
            by_cfg.setdefault(r['base_name'], []).append(r)

    for cfg_name, group in by_cfg.items():
        group = sorted(group, key=lambda x: x['fold_idx'])
        cfg_dir = model_level_root / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        display_name = group[0].get('display_name_cfg', cfg_name)
        fold_thresholds = sorted(
            float(row['selected_threshold'])
            for row in group
            if str(row.get('selected_threshold', '')).strip()
        )

        # val OOF: 每个 image_id 只出现一次（来自其所属 fold 的 val）
        val_oof_map = {}
        for row in group:
            pred_csv = Path(row['run_dir']) / 'val_predictions.csv'
            for rec in load_predictions_csv(pred_csv):
                val_oof_map[rec['image_id']] = rec
        val_oof_rows = list(val_oof_map.values())
        val_scan = scan_thresholds(val_oof_rows)
        save_threshold_scan_csv(val_scan, cfg_dir / 'val_oof_threshold_scan.csv')
        best_thr_info = choose_best_threshold(val_scan)
        raw_best_thr = float(best_thr_info['threshold'])
        best_thr = raw_best_thr
        threshold_source = 'val_oof_scan'
        fold_thr_median = float(np.median(fold_thresholds)) if fold_thresholds else raw_best_thr
        fold_thr_q1 = float(np.percentile(fold_thresholds, 25)) if len(fold_thresholds) >= 2 else fold_thr_median
        fold_thr_q3 = float(np.percentile(fold_thresholds, 75)) if len(fold_thresholds) >= 2 else fold_thr_median

        # Test ensemble averages probabilities from all fold models. If the pooled OOF optimum
        # falls outside the fold-threshold interquartile range, it is usually not transferable.
        if len(fold_thresholds) >= 3 and (raw_best_thr < fold_thr_q1 or raw_best_thr > fold_thr_q3):
            best_thr = fold_thr_median
            threshold_source = 'median_fold_threshold_fallback'

        save_predictions_with_metrics(val_oof_rows, best_thr, cfg_dir / 'val_oof_predictions.csv')
        val_oof_loaded = load_predictions_csv(cfg_dir / 'val_oof_predictions.csv')
        val_oof_metrics = compute_metrics_from_predictions(val_oof_loaded, threshold=best_thr)
        write_rows_csv(cfg_dir / 'val_oof_metrics.csv', [{k: v for k, v in val_oof_metrics.items() if k not in ('fpr', 'tpr', 'roc_thresholds')}], fieldnames=[k for k in val_oof_metrics.keys() if k not in ('fpr', 'tpr', 'roc_thresholds')])
        save_hardcases_file(cfg_dir / 'val_oof_predictions.csv', cfg_dir / 'val_oof_hardcases.csv')

        # test ensemble: 固定 test 集，按 image_id 平均 5 fold prob
        test_accum = {}
        for row in group:
            pred_csv = Path(row['run_dir']) / 'test_predictions.csv'
            for rec in load_predictions_csv(pred_csv):
                x = test_accum.get(rec['image_id'])
                if x is None:
                    x = {
                        'image_id': rec['image_id'],
                        'image_path': rec.get('image_path', ''),
                        'true_label': int(rec['true_label']),
                        'prob_sum': 0.0,
                        'seen': 0,
                    }
                    test_accum[rec['image_id']] = x
                x['prob_sum'] += float(rec['prob_class1'])
                x['seen'] += 1
        test_ensemble_rows = []
        for x in test_accum.values():
            test_ensemble_rows.append({
                'image_id': x['image_id'],
                'image_path': x['image_path'],
                'true_label': x['true_label'],
                'prob_class1': x['prob_sum'] / max(x['seen'], 1),
            })
        test_ensemble_rows = sorted(test_ensemble_rows, key=lambda r: r['image_id'])
        save_predictions_with_metrics(test_ensemble_rows, best_thr, cfg_dir / 'test_ensemble_predictions.csv')
        test_ensemble_loaded = load_predictions_csv(cfg_dir / 'test_ensemble_predictions.csv')
        test_ensemble_metrics = compute_metrics_from_predictions(test_ensemble_loaded, threshold=best_thr)
        write_rows_csv(cfg_dir / 'test_ensemble_metrics.csv', [{k: v for k, v in test_ensemble_metrics.items() if k not in ('fpr', 'tpr', 'roc_thresholds')}], fieldnames=[k for k in test_ensemble_metrics.keys() if k not in ('fpr', 'tpr', 'roc_thresholds')])
        save_hardcases_file(cfg_dir / 'test_ensemble_predictions.csv', cfg_dir / 'test_ensemble_hardcases.csv')

        # ROC compare rows
        for fpr, tpr, thr in zip(val_oof_metrics['fpr'], val_oof_metrics['tpr'], val_oof_metrics['roc_thresholds']):
            roc_compare_rows.append({'model': display_name, 'split': 'val_oof', 'fpr': float(fpr), 'tpr': float(tpr), 'threshold': float(thr)})
        for fpr, tpr, thr in zip(test_ensemble_metrics['fpr'], test_ensemble_metrics['tpr'], test_ensemble_metrics['roc_thresholds']):
            roc_compare_rows.append({'model': display_name, 'split': 'test_ensemble', 'fpr': float(fpr), 'tpr': float(tpr), 'threshold': float(thr)})

        summary_rows.append({
            'base_name': cfg_name,
            'display_name': display_name,
            'val_oof_threshold': best_thr,
            'val_oof_threshold_raw': raw_best_thr,
            'val_oof_threshold_source': threshold_source,
            'fold_threshold_median': fold_thr_median,
            'fold_threshold_q1': fold_thr_q1,
            'fold_threshold_q3': fold_thr_q3,
            'val_oof_auc': val_oof_metrics['auc'],
            'val_oof_bal_acc': val_oof_metrics['bal_acc'],
            'val_oof_f1_macro': val_oof_metrics['f1_macro'],
            'val_oof_acc': val_oof_metrics['acc'],
            'test_ensemble_auc': test_ensemble_metrics['auc'],
            'test_ensemble_bal_acc': test_ensemble_metrics['bal_acc'],
            'test_ensemble_f1_macro': test_ensemble_metrics['f1_macro'],
            'test_ensemble_acc': test_ensemble_metrics['acc'],
        })

    if roc_compare_rows:
        save_roc_compare_rows(log_dir / 'roc_compare_model_level.csv', roc_compare_rows)
        plot_roc_compare(roc_compare_rows, 'test_ensemble', log_dir / 'roc_compare_test_ensemble.png')
        plot_roc_compare(roc_compare_rows, 'val_oof', log_dir / 'roc_compare_val_oof.png')
    if summary_rows:
        write_csv(log_dir / 'model_level_summary.csv', summary_rows, fieldnames=list(summary_rows[0].keys()))
        merge_model_level_into_paper_table(log_dir)


def merge_model_level_into_paper_table(log_dir: Path):
    agg_path = log_dir / 'aggregate_by_config.csv'
    model_level_path = log_dir / 'model_level_summary.csv'
    if not (agg_path.exists() and model_level_path.exists()):
        return
    with open(agg_path, 'r', encoding='utf-8-sig', newline='') as f:
        agg_rows = list(csv.DictReader(f))
    with open(model_level_path, 'r', encoding='utf-8-sig', newline='') as f:
        level_rows = {r['base_name']: r for r in csv.DictReader(f)}
    out = []
    for r in agg_rows:
        x = level_rows.get(r['base_name'], {})
        out.append({
            'Model': r['display_name'] or r['base_name'],
            'CV_AUC_mean_std': f"{float(r['test_auc_mean']):.4f} ± {float(r['test_auc_std']):.4f}",
            'CV_BalAcc_mean_std': f"{float(r['test_bal_acc_mean']):.4f} ± {float(r['test_bal_acc_std']):.4f}",
            'CV_F1_mean_std': f"{float(r['test_f1_macro_mean']):.4f} ± {float(r['test_f1_macro_std']):.4f}",
            'CV_Acc_mean_std': f"{float(r['test_acc_mean']):.4f} ± {float(r['test_acc_std']):.4f}",
            'TestEnsemble_AUC': x.get('test_ensemble_auc', ''),
            'TestEnsemble_BalAcc': x.get('test_ensemble_bal_acc', ''),
            'TestEnsemble_F1': x.get('test_ensemble_f1_macro', ''),
            'TestEnsemble_Acc': x.get('test_ensemble_acc', ''),
        })
    write_csv(log_dir / 'paper_main_table_with_ensemble.csv', out, fieldnames=list(out[0].keys()) if out else ['Model'])


def save_hardcases_file(pred_csv: Path, out_path: Path):
    rows = load_predictions_csv(pred_csv)
    hard_rows = [r for r in rows if int(r['is_wrong']) == 1]
    hard_rows = sorted(hard_rows, key=lambda r: (float(r['wrong_conf']), float(r['margin_from_05'])), reverse=True)
    if not rows:
        return
    write_csv(out_path, hard_rows, fieldnames=list(rows[0].keys()))


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = PROJECT_ROOT / LOG_ROOT / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    output_root_path = PROJECT_ROOT / OUTPUT_ROOT
    output_root_path.mkdir(parents=True, exist_ok=True)

    print('=' * 100)
    print('BUSI 最终论文对比实验：5 models x 5 folds 批跑脚本')
    print('=' * 100)
    print(f'[INFO] project_root  = {PROJECT_ROOT}')
    print(f'[INFO] train_script  = {TRAIN_SCRIPT.name}')
    print(f'[INFO] data_root     = {DATA_ROOT}')
    print(f'[INFO] output_root   = {output_root_path}')
    print(f'[INFO] log_dir       = {log_dir}')
    print(f'[INFO] num_runs      = {len(EXPERIMENTS)}')
    print()

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f'缺少训练脚本: {TRAIN_SCRIPT}')
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f'缺少数据目录: {DATA_ROOT}')

    write_csv(
        log_dir / 'experiment_plan.csv',
        EXPERIMENTS,
        fieldnames=['name', 'base_name', 'display_name', 'fold_idx', 'seed', 'model_family', 'backbone_name', 'backbone_module', 'backbone_func', 'backbone_out_dim', 'backbone_lr', 'head_lr']
    )

    statuses = []
    run_rows = []
    python_exe = sys.executable

    for i, exp in enumerate(EXPERIMENTS, start=1):
        print('=' * 100)
        print(f'[{i}/{len(EXPERIMENTS)}] {exp["name"]}')
        print('=' * 100)

        before_names = {p.name for p in list_subdirs(output_root_path)}
        log_file = log_dir / f'{exp["name"]}.log'
        cmd = [
            python_exe, str(TRAIN_SCRIPT),
            '--data-root', str(DATA_ROOT),
            '--output-root', OUTPUT_ROOT,
            '--model-family', str(exp['model_family']),
            '--backbone-name', str(exp['backbone_name']),
            '--backbone-module', str(exp['backbone_module']),
            '--backbone-func', str(exp['backbone_func']),
            '--backbone-out-dim', str(exp.get('backbone_out_dim', 0)),
            '--param-tag', str(exp['name']),
            '--fold-idx', str(exp['fold_idx']),
            '--seed', str(exp['seed']),
            '--input-size', str(exp.get('input_size', 256)),
            '--batch-size', '8',
            '--num-workers', '0',
            '--num-epochs', str(exp['num_epochs']),
            '--head-lr', str(exp['head_lr']),
            '--backbone-lr', str(exp['backbone_lr']),
            '--weight-decay', str(exp['weight_decay']),
            '--dropout', str(exp['dropout']),
            '--label-smoothing', str(exp['label_smoothing']),
            '--early-stopping-patience', str(exp['early_stopping_patience']),
            '--bbox-expand-ratio', str(exp['bbox_expand_ratio']),
            '--save-by-metric', str(exp['save_by_metric']),
            '--threshold-selection-mode', str(exp['threshold_selection_mode']),
            '--lr-schedule', str(exp['lr_schedule']),
            '--lr-warmup-epochs', str(exp['lr_warmup_epochs']),
            '--lr-min-ratio', str(exp['lr_min_ratio']),
            '--use-ema', '1' if exp['use_ema'] else '0',
            '--ema-decay', str(exp['ema_decay']),
            '--use-class-weight', '1' if exp['use_class_weight'] else '0',
            '--use-manual-class-weights', '1' if exp['use_manual_class_weights'] else '0',
            '--manual-class-weights', str(exp['manual_class_weights']),
            '--use-hflip-tta', '1' if exp['use_hflip_tta'] else '0',
            '--use-temperature-scaling', '1' if exp['use_temperature_scaling'] else '0',
            '--ensemble-topk', str(exp['ensemble_topk']),
        ]
        return_code, elapsed = stream_subprocess(cmd, PROJECT_ROOT, log_file)
        run_dir = detect_new_run_dir(output_root_path, before_names)

        status_row = {
            'name': exp['name'], 'base_name': exp['base_name'], 'fold_idx': exp['fold_idx'], 'seed': exp['seed'],
            'return_code': return_code, 'elapsed_sec': f'{elapsed:.1f}', 'log_file': str(log_file),
            'run_dir': str(run_dir) if run_dir is not None else '', 'status': 'ok' if return_code == 0 else 'failed',
        }
        statuses.append(status_row)

        print(f'    -> return_code={return_code} | elapsed={elapsed:.1f}s')
        print(f'    -> log_file={log_file}')
        print(f'    -> run_dir={run_dir}')

        if return_code == 0 and run_dir is not None:
            try:
                row = build_run_row(exp, run_dir)
                run_rows.append(row)
                print(
                    '    -> metrics: '
                    f"test_bal_acc={row['test_bal_acc']:.6f}, "
                    f"test_f1_macro={row['test_f1_macro']:.6f}, "
                    f"test_auc={row['test_auc']:.6f}, "
                    f"thr={row['selected_threshold']:.4f}"
                )
            except Exception as e:
                print(f'[WARN] {exp["name"]} 训练完成，但汇总解析失败：{repr(e)}')
        else:
            print(f'[WARN] {exp["name"]} 非零退出。')

        save_incremental_outputs(log_dir, statuses, run_rows)

        if return_code != 0 and not CONTINUE_ON_ERROR:
            print('[STOP] CONTINUE_ON_ERROR=False，停止后续批跑。')
            break

    had_failure = any(s.get('status') != 'ok' for s in statuses)

    aggregate_test_hardcases(log_dir, run_rows)
    build_model_level_artifacts(log_dir, run_rows)

    readme = log_dir / 'README_BUSI_FINAL_5MODELS_5FOLD.txt'
    readme.write_text(
        '\n'.join([
            'BUSI final paper compare: 5 models x 5 folds',
            f'train_script = {TRAIN_SCRIPT}',
            f'output_root = {output_root_path}',
            f'log_dir = {log_dir}',
            '',
            'Models:',
            *[
                f"- {cfg['name']} ({cfg['display_name']}): family={cfg['model_family']}, name={cfg['backbone_name']}, module={cfg['backbone_module']}, func={cfg['backbone_func']}"
                for cfg in BASE_CONFIGS
            ],
            '',
            f'Global seed: {GLOBAL_SEED}',
            f'Folds: {FOLDS}',
            '',
            'Key outputs:',
            '- all_runs_metrics.csv: per-fold raw metrics',
            '- aggregate_by_config.csv: fold mean/std by model',
            '- paper_main_table.csv: plot/table-friendly mean±std',
            '- model_level_summary.csv: val OOF + test ensemble metrics by model',
            '- paper_main_table_with_ensemble.csv: mean/std plus test-ensemble metrics',
            '- roc_compare_model_level.csv: plot-ready ROC points for all models',
            '- test_hardcases_by_config_hard.csv: per-model hard test samples across folds',
        ]),
        encoding='utf-8'
    )

    print('\n' + '=' * 100)
    print('BUSI 最终论文对比实验批跑完成')
    print('=' * 100)
    print(f'[DONE] log_dir = {log_dir}')
    if run_rows:
        print(f"[DONE] all_runs_metrics.csv             = {log_dir / 'all_runs_metrics.csv'}")
        print(f"[DONE] aggregate_by_config.csv         = {log_dir / 'aggregate_by_config.csv'}")
        print(f"[DONE] paper_main_table.csv            = {log_dir / 'paper_main_table.csv'}")
        print(f"[DONE] model_level_summary.csv         = {log_dir / 'model_level_summary.csv'}")
        print(f"[DONE] paper_main_table_with_ensemble  = {log_dir / 'paper_main_table_with_ensemble.csv'}")
        print(f"[DONE] roc_compare_model_level.csv     = {log_dir / 'roc_compare_model_level.csv'}")
        print(f"[DONE] test_hardcases_by_config_hard   = {log_dir / 'test_hardcases_by_config_hard.csv'}")

    return 1 if had_failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
