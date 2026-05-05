from __future__ import annotations
import contextlib, io, json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT=Path(r'C:\Users\Afr1ste\PycharmProjects\Thyroid')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import generate_paper_gradcam_sanity_figure as grad_fig

OUT_DIR=Path(r'C:\Users\Afr1ste\OneDrive\My Notes\tex\pr_ultrasound_lesion_classification\figures\candidate_previews')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(x, default=0.0):
    try: return float(x)
    except Exception: return default


def compactness(cam, top_frac=0.10):
    flat=np.asarray(cam,dtype=np.float32).ravel()
    k=max(1,int(round(flat.size*top_frac)))
    idx=np.argpartition(flat,-k)[-k:]
    return float(flat[idx].sum()/(flat.sum()+1e-8))


def border_energy(cam,border=0.12):
    h,w=cam.shape; by=max(1,int(h*border)); bx=max(1,int(w*border))
    mask=np.zeros_like(cam,dtype=bool)
    mask[:by,:]=mask[-by:,:]=True; mask[:,:bx]=mask[:,-bx:]=True
    return float(cam[mask].sum()/(cam.sum()+1e-8))


def dark_overlap(rgb,cam):
    gray=rgb.mean(axis=2); h,w=gray.shape
    y0,y1=int(h*.08),int(h*.92); x0,x1=int(w*.08),int(w*.92)
    g=gray[y0:y1,x0:x1]; c=cam[y0:y1,x0:x1]
    if c.sum()<=1e-8: return 0.0
    thr=float(np.quantile(g,.45))
    mask=g<=thr
    return float(c[mask].sum()/(c.sum()+1e-8))


def colorfulness(rgb):
    # Source images should be grayscale ultrasound; colored marks imply visual contamination.
    return float(np.mean(np.abs(rgb[...,0]-rgb[...,1]) + np.abs(rgb[...,1]-rgb[...,2]) + np.abs(rgb[...,0]-rgb[...,2])))


def central_peak_distance(cam):
    h,w=cam.shape
    y,x=np.unravel_index(int(np.argmax(cam)),cam.shape)
    dy=(y-(h-1)/2)/((h-1)/2); dx=(x-(w-1)/2)/((w-1)/2)
    return float((dx*dx+dy*dy)**0.5)


def get_layer(model, layer_key):
    net=model.backbone.network
    if layer_key=='stage3_h16': return net[4][-1]
    if layer_key=='stage4_h32': return net[6][-1]
    if layer_key=='stage2_h8': return net[2][-1]
    raise KeyError(layer_key)


def prediction_pool(case, limit=None):
    rows=grad_fig.read_predictions(case.run_dir/'test_predictions.csv')
    # Correct cases only. Include both classes because the figure is a sanity check, not class-specific evidence.
    correct=[r for r in rows if r.get('is_wrong')=='0' and r.get('true_label')==r.get('pred_label')]
    # Prefer confident but keep enough visual diversity.
    correct=sorted(correct,key=lambda r:max(safe_float(r.get('prob_class1')),1-safe_float(r.get('prob_class1'))),reverse=True)
    if limit: correct=correct[:limit]
    return correct


def evaluate_dataset(case, layer_key, device):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        module=grad_fig.import_train_module(case.module_name)
        dataset=grad_fig.make_dataset(module,case)
        _tr,eval_transform=module.build_transforms()
        model=grad_fig.instantiate_model(module,case.run_dir/case.checkpoint_name,device)
    layer=get_layer(model,layer_key)
    gradcam=grad_fig.GradCAM(model,layer)
    items=[]
    try:
        for pred in prediction_pool(case):
            try:
                roi_img,label,iid=grad_fig.get_roi_by_id(dataset,pred['image_id'])
            except KeyError:
                continue
            tensor=eval_transform(roi_img).unsqueeze(0).to(device)
            target=int(label)
            try:
                cam,probs=gradcam(tensor,target)
            except Exception:
                continue
            rgb=grad_fig.resize_for_display(roi_img,256)
            overlay=grad_fig.cam_overlay(rgb,cam,alpha=.42)
            center58=grad_fig.center_energy(cam,.58)
            center72=grad_fig.center_energy(cam,.72)
            comp=compactness(cam)
            edge=border_energy(cam)
            dark=dark_overlap(rgb,cam)
            color=colorfulness(rgb)
            peakdist=central_peak_distance(cam)
            white=float((rgb.mean(axis=2)>0.90).mean())
            conf=float(probs[target])
            # High score means: centered/compact, dark-lesion-overlapping, confident, not border/color contaminated.
            score=(0.18*center58+0.12*center72+0.22*comp+0.18*dark+0.12*conf
                   -0.18*edge-0.10*peakdist-0.65*white-1.8*color)
            items.append({
                'dataset':case.dataset,'layer_key':layer_key,'image_id':iid,
                'true_label':target,'pred_label':int(pred['pred_label']),
                'prob_class1':safe_float(pred.get('prob_class1')),
                'threshold':safe_float(pred.get('threshold'),.5),
                'conf_true_single_ckpt':conf,'center58':center58,'center72':center72,
                'compact':comp,'dark_overlap':dark,'border_energy':edge,
                'peak_dist':peakdist,'white_fraction':white,'colorfulness':color,
                'score':score,'rgb':rgb,'overlay':overlay,
            })
    finally:
        gradcam.close(); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return sorted(items,key=lambda x:x['score'],reverse=True)


def save_sheet(dataset, layer_key, ranked, n=16):
    ranked=ranked[:n]
    cols=4; rows=int(np.ceil(n/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(11.4,rows*3.15),dpi=220)
    axes=np.asarray(axes).reshape(rows,cols)
    for idx,ax in enumerate(axes.ravel()):
        if idx>=len(ranked): ax.axis('off'); continue
        it=ranked[idx]
        combo=np.concatenate([it['rgb'],it['overlay']],axis=0)
        ax.imshow(combo); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"{chr(65+idx)} {it['image_id']} y={it['true_label']} p1={it['prob_class1']:.2f}\n"
            f"c={it['center58']:.2f} k={it['compact']:.2f} dark={it['dark_overlap']:.2f} edge={it['border_energy']:.2f}",
            fontsize=7.2)
        for sp in ax.spines.values(): sp.set_linewidth(.8); sp.set_color('#222')
    fig.suptitle(f'{dataset} Grad-CAM candidates v3 - {layer_key}: ROI input over overlay',fontsize=12,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.975])
    out=OUT_DIR/f'fig5_gradcam_candidates_{dataset.lower()}_v3_{layer_key}.png'
    fig.savefig(out,bbox_inches='tight',pad_inches=.05); plt.close(fig)
    txt=OUT_DIR/f'fig5_gradcam_candidates_{dataset.lower()}_v3_{layer_key}.txt'
    lines=[f'png={out}',f'n_evaluated={len(ranked)}']
    for i,it in enumerate(ranked[:n]):
        lines.append(f"{chr(65+i)} id={it['image_id']} y={it['true_label']} pred={it['pred_label']} p1={it['prob_class1']:.4f} conf={it['conf_true_single_ckpt']:.4f} center={it['center58']:.4f} compact={it['compact']:.4f} dark={it['dark_overlap']:.4f} edge={it['border_energy']:.4f} color={it['colorfulness']:.5f} score={it['score']:.4f}")
    txt.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return out,txt


def main():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif','STIXGeneral'],'mathtext.fontset':'stix'})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cases=[c for c in grad_fig.CASES if c.dataset in ('BUSI','AUL')]
    all_summary=[]
    for case in cases:
        # Override preferred_image_id during candidate search.
        case=grad_fig.GradCamCase(case.dataset,case.module_name,case.dataset_class,case.root_attr,case.split,case.bbox_expand,case.run_dir,case.checkpoint_name,None)
        for layer_key in ['stage3_h16','stage4_h32']:
            ranked=evaluate_dataset(case,layer_key,device)
            out,txt=save_sheet(case.dataset,layer_key,ranked,n=16)
            all_summary.append(str(txt))
            print(txt)
            print('\n'.join(txt.read_text(encoding='utf-8').splitlines()[:8]))
    (OUT_DIR/'fig5_gradcam_busi_aul_v3_summary.txt').write_text('\n'.join(all_summary)+'\n',encoding='utf-8')

if __name__=='__main__': main()
