from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(r'<LOCAL_THYROID_ROOT>')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_paper_gradcam_sanity_figure as grad_fig

OUT_DIR = Path(r'<LOCAL_MANUSCRIPT_ROOT>\figures\candidate_previews')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def compactness(cam: np.ndarray, top_frac: float = 0.12) -> float:
    flat = np.asarray(cam, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(1, int(round(flat.size * top_frac)))
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].sum() / (flat.sum() + 1e-8))


def dark_overlap(rgb: np.ndarray, cam: np.ndarray) -> float:
    gray = rgb.mean(axis=2)
    # Avoid ultrasound margins; focus only on central tissue field.
    h, w = gray.shape
    y0, y1 = int(h * 0.08), int(h * 0.92)
    x0, x1 = int(w * 0.08), int(w * 0.92)
    tissue = gray[y0:y1, x0:x1]
    cam_c = cam[y0:y1, x0:x1]
    if tissue.size == 0 or cam_c.sum() <= 1e-8:
        return 0.0
    thr = float(np.quantile(tissue, 0.42))
    mask = tissue <= thr
    return float(cam_c[mask].sum() / (cam_c.sum() + 1e-8))


def border_energy(cam: np.ndarray, border: float = 0.10) -> float:
    h, w = cam.shape
    by, bx = int(round(h * border)), int(round(w * border))
    mask = np.zeros_like(cam, dtype=bool)
    mask[:by, :] = True
    mask[-by:, :] = True
    mask[:, :bx] = True
    mask[:, -bx:] = True
    return float(cam[mask].sum() / (cam.sum() + 1e-8))


def candidate_rows(case: grad_fig.GradCamCase, limit: int = 260):
    rows = grad_fig.read_predictions(case.run_dir / 'test_predictions.csv')
    rows = [
        r for r in rows
        if r.get('true_label') == '1'
        and r.get('pred_label') == '1'
        and r.get('is_wrong') == '0'
        and safe_float(r.get('prob_class1')) >= safe_float(r.get('threshold'), 0.5)
    ]
    # Include confident cases and a deterministic middle-confidence tail for visual diversity.
    rows_sorted = sorted(rows, key=lambda r: safe_float(r.get('prob_class1')), reverse=True)
    top = rows_sorted[:180]
    tail = rows_sorted[180::4][:80]
    pool = top + tail
    seen = set()
    out = []
    for r in pool:
        iid = r.get('image_id')
        if iid in seen:
            continue
        seen.add(iid)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def main():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
    })
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    case = next(c for c in grad_fig.CASES if c.dataset == 'TN5000')
    case = grad_fig.GradCamCase(
        dataset=case.dataset,
        module_name=case.module_name,
        dataset_class=case.dataset_class,
        root_attr=case.root_attr,
        split=case.split,
        bbox_expand=case.bbox_expand,
        run_dir=case.run_dir,
        checkpoint_name=case.checkpoint_name,
        preferred_image_id=None,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        module = grad_fig.import_train_module(case.module_name)
        dataset = grad_fig.make_dataset(module, case)
        _train_transform, eval_transform = module.build_transforms()
        model = grad_fig.instantiate_model(module, case.run_dir / case.checkpoint_name, device)
    target_layer = grad_fig.get_target_layer(model)
    gradcam = grad_fig.GradCAM(model, target_layer)
    items = []
    try:
        rows = candidate_rows(case)
        for i, pred in enumerate(rows, 1):
            try:
                roi_img, label, image_id = grad_fig.get_roi_by_id(dataset, pred['image_id'])
            except KeyError:
                continue
            tensor = eval_transform(roi_img).unsqueeze(0).to(device)
            target_class = int(label)
            cam, probs = gradcam(tensor, target_class)
            rgb = grad_fig.resize_for_display(roi_img, size=256)
            overlay = grad_fig.cam_overlay(rgb, cam, alpha=0.42)
            center58 = grad_fig.center_energy(cam, frac=0.58)
            center68 = grad_fig.center_energy(cam, frac=0.68)
            comp = compactness(cam)
            dark = dark_overlap(rgb, cam)
            edge = border_energy(cam)
            peak = float(np.percentile(cam, 97) - np.mean(cam))
            white = float((rgb.mean(axis=2) > 0.90).mean())
            conf_true = float(probs[target_class])
            # Manual-screening score: compact, central, dark-region-overlapping CAM; penalize borders.
            score = 0.23 * center58 + 0.18 * center68 + 0.23 * comp + 0.17 * dark + 0.12 * peak + 0.07 * conf_true - 0.18 * edge - 0.55 * white
            items.append({
                'image_id': image_id,
                'true_label': int(label),
                'pred_label': int(pred['pred_label']),
                'prob_class1': safe_float(pred.get('prob_class1')),
                'threshold': safe_float(pred.get('threshold'), 0.5),
                'conf_true_single_ckpt': conf_true,
                'center58': center58,
                'center68': center68,
                'compact': comp,
                'dark_overlap': dark,
                'border_energy': edge,
                'peakiness': peak,
                'white_fraction': white,
                'score': score,
                'rgb': rgb,
                'overlay': overlay,
            })
    finally:
        gradcam.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranked = sorted(items, key=lambda x: x['score'], reverse=True)[:16]
    fig, axes = plt.subplots(4, 4, figsize=(11.0, 10.2), dpi=220)
    for idx, item in enumerate(ranked):
        ax = axes[idx // 4, idx % 4]
        combo = np.concatenate([item['rgb'], item['overlay']], axis=0)
        ax.imshow(combo)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{chr(65+idx)} {item['image_id']} | c={item['center58']:.2f} k={item['compact']:.2f}\n"
            f"dark={item['dark_overlap']:.2f} edge={item['border_energy']:.2f} score={item['score']:.2f}",
            fontsize=7.4,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor('#202020')
    fig.suptitle('TN5000 Grad-CAM candidates v2: ROI input over Grad-CAM overlay', fontsize=12, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    png = OUT_DIR / 'fig5_gradcam_candidates_tn5000_v2.png'
    fig.savefig(png, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    serializable = [{k: v for k, v in item.items() if k not in {'rgb', 'overlay'}} for item in ranked]
    json_path = OUT_DIR / 'fig5_gradcam_candidates_tn5000_v2.json'
    txt_path = OUT_DIR / 'fig5_gradcam_candidates_tn5000_v2.txt'
    json_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [f'device={device}', f'evaluated={len(items)}', f'png={png}']
    for i, item in enumerate(serializable):
        lines.append(
            f"{chr(65+i)} id={item['image_id']} p1={item['prob_class1']:.4f} conf={item['conf_true_single_ckpt']:.4f} "
            f"center58={item['center58']:.4f} compact={item['compact']:.4f} dark={item['dark_overlap']:.4f} "
            f"edge={item['border_energy']:.4f} score={item['score']:.4f}"
        )
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(txt_path)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
