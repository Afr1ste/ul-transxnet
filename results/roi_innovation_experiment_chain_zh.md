# ROI 创新实验链条设计

更新时间：2026-05-02

## 1. 目标定位

当前稿件的主实验使用 annotation-derived ROI crop。这个协议适合做 backbone 和分类器公平比较，但临床完整性会被质疑：真实部署时病灶框从哪里来？

建议把 ROI 相关实验组织成一条独立、可防御的实验链条：

> 从人工标注 ROI 分类，扩展为“诊断无关的一类病灶定位 + ROI 裁剪 + 轻量分类器”的全图输入诊断流程，并量化自动 ROI 与人工 ROI 的性能差距、失败来源和部署代价。

这里不要声称提出了新的检测器架构。真正的贡献应表述为：

- 建立了超声 ROI 分类模型从 annotation-dependent protocol 到 automatic ROI protocol 的闭环验证。
- 检测器只学习 lesion localization，不使用 benign/malignant 诊断标签，避免把分类监督泄漏进 ROI 生成阶段。
- 同时报告 oracle ROI、auto ROI、full image 三个输入协议，证明 ROI 不是 cosmetic preprocessing，而是当前超声分类链条中的必要环节。
- 通过 box IoU、IoU 分层分类误差、扩张比例敏感性和部署延迟，把 ROI 质量与最终诊断性能联系起来。

## 2. 当前已经完成的证据

### 2.1 TN5000 YOLO 一类病灶检测数据

已构建 YOLO 格式检测数据集：

- 数据集目录：`C:\Users\Afr1ste\PycharmProjects\Thyroid\detector_datasets\tn5000_yolo_lesion_v1`
- 构建脚本：`C:\Users\Afr1ste\PycharmProjects\Thyroid\build_tn5000_yolo_detection_dataset.py`
- 标注来源：TN5000 VOC XML bbox
- 类别：单类 `lesion`
- split：train 3500, val 500, test 1000
- 异常：missing/corrupt 为 0

### 2.2 TN5000 YOLO11n smoke detector

已完成一轮 YOLO11n smoke 训练：

- 训练目录：`C:\Users\Afr1ste\PycharmProjects\Thyroid\tn5000_roi_detector_runs\yolo11n_tn5000_lesion_smoke_20260502_123522`
- best weights：`C:\Users\Afr1ste\PycharmProjects\Thyroid\tn5000_roi_detector_runs\yolo11n_tn5000_lesion_smoke_20260502_123522\weights\best.pt`
- test validation 目录：`C:\Users\Afr1ste\PycharmProjects\Thyroid\tn5000_roi_detector_runs\yolo11n_tn5000_lesion_smoke_20260502_123522_testval`

Ultralytics test split 指标：

| split | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val | 0.897 | 0.950 | 0.959 | 0.652 |
| test | 0.906 | 0.937 | 0.956 | 0.647 |

Top-1 predicted box 与 VOC box 的质量：

| split | no-box rate | mean IoU | median IoU | recall@0.50 | recall@0.75 |
|---|---:|---:|---:|---:|---:|
| val | 0.000 | 0.7827 | 0.8451 | 0.926 | 0.788 |
| test | 0.000 | 0.7875 | 0.8432 | 0.937 | 0.798 |

### 2.3 TN5000 自动 ROI 闭环分类 smoke

已完成 oracle ROI / auto ROI / full image 三协议闭环评估：

- 评估脚本：`C:\Users\Afr1ste\PycharmProjects\Thyroid\eval_tn5000_auto_roi_pipeline.py`
- 输出目录：`C:\Users\Afr1ste\PycharmProjects\Thyroid\eval_reports\tn5000_auto_roi_pipeline_yolo11n_20260502_1245`
- 指标 CSV：`C:\Users\Afr1ste\PycharmProjects\Thyroid\eval_reports\tn5000_auto_roi_pipeline_yolo11n_20260502_1245\auto_roi_classification_metrics.csv`
- 分类器：当前 TN5000 GGG-withMCA 3-seed checkpoint ensemble，共 9 个 checkpoint
- 校准：每种输入模式在 val 上单独做 temperature scaling 和 threshold selection
- 当前裁剪规则：`bbox_expand_ratio=0.30`, `square_crop=false`

重要说明：`square_crop=false` 是为了匹配当前 TN5000 分类训练脚本 `fl_tn5000_roi_compare_multimodel.py` 的真实代码路径。该脚本目前是按宽高分别扩张 bbox 后 resize，不是强制正方形裁剪。论文图里的 “square crop” 若保留，需要用独立实验支持，或者改成 “expanded ROI crop”。

闭环分类结果：

| mode | split | AUC | BalAcc | F1-macro | Acc | threshold |
|---|---|---:|---:|---:|---:|---:|
| oracle ROI | val | 0.9701 | 0.9106 | 0.9064 | 0.9280 | 0.71 |
| oracle ROI | test | 0.9542 | 0.8809 | 0.8708 | 0.8990 | 0.71 |
| auto ROI | val | 0.9362 | 0.8864 | 0.8655 | 0.8920 | 0.82 |
| auto ROI | test | 0.9442 | 0.8602 | 0.8242 | 0.8530 | 0.82 |
| full image | val | 0.6749 | 0.6211 | 0.5470 | 0.5620 | 0.69 |
| full image | test | 0.6609 | 0.6248 | 0.5452 | 0.5580 | 0.69 |

当前结论：

- 自动 ROI 可行：test mAP50=0.956，no-box rate=0。
- auto ROI 相比 oracle ROI 的 AUC 下降约 1.0 pp：0.9542 -> 0.9442。
- operating point 下降更明显：BalAcc 下降约 2.1 pp，F1-macro 下降约 4.7 pp。
- full image 分类显著失败：AUC=0.6609，说明 ROI 定位不是装饰性预处理，而是当前分类器成立的关键条件。

## 3. 建议补齐的实验链条

下面按优先级分组。P0-P3 是论文主线最低要求；P4-P6 会显著增强审稿防御；P7 是部署增强项；P8 是 revision reserve。

### P0. 协议审计与术语修正

状态：基本完成，但需要在论文文字和图中同步。

目的：

- 明确 ROI 来源、裁剪规则和诊断标签使用边界。
- 避免论文图说 “square crop”，代码实际不是 square crop 的不一致。

必须写清楚：

- Oracle ROI：使用数据集提供的 lesion bbox，只用于裁剪，不使用 diagnosis label。
- Auto ROI：YOLO 一类 detector 预测 lesion bbox，再按同样 expand rule 裁剪。
- Full image：不裁剪，直接 resize 原图，作为 negative control。
- 当前主分类训练协议是 expanded rectangular ROI crop，不是 forced square crop。

建议改动：

- 如果不补 square crop 实验，论文图和正文统一改为 “expanded ROI crop”。
- 如果希望保留 “square crop”，需要执行 P4 的 square-vs-rectangular 实验。

### P1. TN5000 自动 ROI detector 主实验

状态：smoke 已完成；建议补 3-seed 或至少补一次正式记录。

目标：

- 让 detector 本身的稳定性有统计支撑。

推荐设置：

- 模型：YOLO11n
- seeds：17, 27, 37
- train/val/test：TN5000 fixed split
- 标签：单类 lesion
- 训练 epoch：以 smoke 收敛曲线为准，可设 100 epoch + early stop

报告指标：

- P, R, mAP50, mAP50-95
- no-box rate
- mean IoU, median IoU
- recall@IoU 0.50 / 0.75 / 0.90

建议输出表：

| Detector | Params | mAP50 | mAP50-95 | mean IoU | recall@0.50 | recall@0.75 | no-box |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | ... | mean±std | mean±std | mean±std | mean±std | mean±std | mean±std |

### P2. TN5000 闭环诊断主实验

状态：smoke 已完成；建议补正式版本。

目标：

- 证明自动 ROI 可以把模型从 “需要人工框” 推到 “全图输入后自动裁剪再诊断”。

输入协议：

1. Oracle ROI + UL-TransXNet：上界。
2. Auto ROI + UL-TransXNet：自动 ROI 闭环。
3. Full image + UL-TransXNet：负对照。

建议同时报告两种 threshold 策略：

- Deployment-calibrated：每种输入模式在 val 上独立校准 threshold。当前 smoke 使用的是这一种。
- Oracle-threshold transfer：使用 oracle ROI val 上得到的 threshold，直接迁移到 auto ROI test。这个更严格，能回答“自动 ROI 是否改变 score calibration”。

报告指标：

- AUC
- balanced accuracy
- macro F1
- benign recall / malignant recall
- accuracy
- threshold
- optional：ECE, Brier score

主表建议：

| Input protocol | Detector | AUC | BalAcc | F1-macro | Benign recall | Malignant recall |
|---|---|---:|---:|---:|---:|---:|
| Full image | none | ... | ... | ... | ... | ... |
| Oracle ROI | annotation | ... | ... | ... | ... | ... |
| Auto ROI | YOLO11n | ... | ... | ... | ... | ... |

论文可支持的主张：

- “Automatic ROI narrows most of the gap between full-image inference and annotation-derived ROI inference.”
- 不要写成 “solves clinical deployment”，除非后面补了临床级检测器泛化和移动端验证。

### P3. ROI 质量到诊断性能的归因分析

状态：未完成，强烈建议补。

目标：

- 审稿人会问：auto ROI 掉点是 detector 定位差导致，还是分类器本身对 crop distribution shift 敏感？
- IoU 分层可以直接回答。

做法：

对 test set 的 auto ROI 预测按 IoU 分桶：

- no detection
- IoU < 0.50
- 0.50 <= IoU < 0.75
- 0.75 <= IoU < 0.90
- IoU >= 0.90

每个桶报告：

- 样本数
- AUC 或正负类 recall
- accuracy / error rate
- benign error rate
- malignant error rate

建议图：

- x 轴：IoU bucket
- y 轴：classification accuracy 或 error rate
- 另画一张 benign/malignant recall grouped bars

建议结论模板：

- 如果高 IoU 桶接近 oracle ROI，说明主要误差来自定位。
- 如果高 IoU 桶仍明显低于 oracle ROI，说明 detector crop 的边界/上下文分布与人工框仍有 shift，需要调整 crop expansion 或训练时加入 predicted-box augmentation。

### P4. ROI crop rule 敏感性实验

状态：未完成，优先级高。它直接关系到图和方法描述是否站得住。

目标：

- 确认 “expand ratio=0.30” 是否合理。
- 确认 rectangular expanded crop 与 square crop 哪个更优。
- 给 ROI 预处理图的数学描述提供依据。

建议 sweep：

1. Expand ratio:

   - 0.00
   - 0.10
   - 0.20
   - 0.30
   - 0.40
   - 0.50

2. Crop geometry:

   - rectangular expanded crop：当前真实代码路径
   - square expanded crop：以 lesion center 为中心，side=max(w,h)，再按 ratio 扩张

最低成本版本：

- 不重训分类器，只对已有 checkpoint ensemble 做 inference sweep。
- 对 oracle ROI 和 auto ROI 都 sweep。

更严格版本：

- 对最优 crop rule 重训 UL-TransXNet。
- 成本较高，可作为 revision reserve。

建议表：

| ROI source | Geometry | Expand | AUC | BalAcc | F1-macro |
|---|---|---:|---:|---:|---:|
| Oracle | rectangular | 0.30 | ... | ... | ... |
| Auto | rectangular | 0.30 | ... | ... | ... |
| Auto | square | 0.30 | ... | ... | ... |

### P5. Predicted-box augmentation

状态：未完成，可作为增强项。

目的：

- 当前分类器训练时看到的是 annotation-derived crop，测试 auto ROI 时看到的是 detector-derived crop。二者存在 crop distribution shift。
- 可以在训练阶段模拟 detector 噪声，让分类器更适应 auto ROI。

方案：

- 在训练集中对 GT bbox 加随机扰动：
  - center jitter：±5% / ±10%
  - scale jitter：0.9-1.2
  - aspect jitter：可选
- 或先用 detector 对 train split 生成 predicted boxes，再用 mixed crop training：
  - 50% oracle crop
  - 50% detector crop

对比：

| Training crop | Test crop | AUC | BalAcc | F1-macro |
|---|---|---:|---:|---:|
| Oracle only | Oracle | ... | ... | ... |
| Oracle only | Auto | ... | ... | ... |
| Oracle + jitter | Auto | ... | ... | ... |
| Oracle + detector crop | Auto | ... | ... | ... |

如果有效，这会成为 ROI 创新链条里最有方法味的一环。

### P6. 可视化和失败案例

状态：未完成，建议补。

目的：

- 医学图像论文需要让读者直观看到 detector 是否真的框住 lesion。
- failure cases 能降低审稿人对 cherry-pick 的质疑。

建议 4 类示例，每类 4 张：

1. good detection + correct diagnosis
2. good detection + wrong diagnosis
3. poor detection + wrong diagnosis
4. full image wrong but auto ROI correct

每张显示：

- 原图
- GT bbox
- predicted bbox
- IoU
- classifier probability
- true label / predicted label

输出：

- 一张主文小图：4 个代表例子
- appendix 大图：16 个例子

### P7. 自动 ROI 端到端部署代价

状态：未完成，可作为部署增强项。

目标：

- 现在已有 Android 分类器延迟表。如果加入 detector，需要报告 detector + crop + classifier 的总延迟。

报告：

- detector latency
- ROI crop latency
- classifier latency
- total latency
- memory footprint
- model size

对比：

| Protocol | Detector | Classifier | AUC | BalAcc | Latency/image |
|---|---|---|---:|---:|---:|
| Oracle ROI | none | UL-TransXNet | ... | ... | current classifier latency |
| Auto ROI | YOLO11n | UL-TransXNet | ... | ... | detector + classifier |
| Full image | none | UL-TransXNet | ... | ... | classifier only |

注意：

- Oracle ROI 没有真实部署意义，只是上界。
- Auto ROI 的 latency 才是端到端临床流程的主要数字。

### P8. BUSI/AUL 自动 ROI 泛化

状态：未完成，暂不建议作为当前主线强行补。

原因：

- BUSI/AUL 样本小，detector 训练稳定性可能不足。
- 当前主稿已经承载了大量结果，再加入两个小数据集 detector 可能增加不稳定点。

可以作为 revision reserve：

- 为 BUSI/AUL 各构建一类 YOLO dataset。
- 只跑 YOLO11n 5-fold 或 train/val/test fixed protocol。
- 报告 auto ROI 是否也接近 oracle ROI。

如果时间紧，当前稿件可把自动 ROI 只作为 TN5000 上的“large-scale feasibility analysis”，不要声称三数据集均完成自动 ROI。

## 4. 最推荐的最小可发表链条

如果目标是在不显著拖慢投稿的前提下补强 ROI 弱点，建议最低完成：

1. P0：统一术语，确认 manuscript 中不再把当前协议误写为 square crop。
2. P1：TN5000 YOLO11n detector 3-seed。
3. P2：TN5000 closed-loop oracle/auto/full 三协议正式表。
4. P3：IoU bucket vs diagnosis error。
5. P4：expand ratio + square/rectangular inference sweep。
6. P6：代表性成功/失败图。

这条链条完成后，ROI 叙事可以从“我们依赖人工 ROI”升级为：

> The primary benchmark uses annotation-derived ROI crops to isolate classifier comparison. To assess deployment feasibility, we additionally trained a diagnosis-agnostic one-class lesion localizer and evaluated a closed-loop automatic ROI pipeline. Automatic ROI preserved most of the oracle-ROI AUC while substantially outperforming full-image inference, and the remaining gap was analyzed through localization-quality stratification.

中文含义：

> 主实验使用标注 ROI 是为了公平比较分类器；另外补充自动 ROI 闭环实验，证明该分类器可以接入一个不使用诊断标签的病灶定位器，从全图生成 ROI 后完成诊断。自动 ROI 保留了大部分人工 ROI 的 AUC，并显著优于直接全图分类；剩余差距通过定位质量分层解释。

## 5. 可直接生成的论文表图

### 主文或补充表 1：Automatic ROI detection quality

内容：

- Detector model
- Params
- mAP50
- mAP50-95
- mean IoU
- recall@0.50
- recall@0.75
- no-box rate

### 主文或补充表 2：Closed-loop diagnosis

内容：

- Full image
- Oracle ROI
- Auto ROI
- AUC / BalAcc / F1 / benign recall / malignant recall

### 主文或补充图 1：Automatic ROI workflow

内容：

Original image -> lesion detector -> expanded ROI crop -> UL-TransXNet classifier -> diagnosis

注意：

- 图中不要暗示 detector 使用 benign/malignant 标签。
- 如果没有完成 square crop 实验，图中写 expanded ROI crop。

### 主文或补充图 2：Localization quality versus diagnosis error

内容：

- IoU bucket on x-axis
- classification error or accuracy on y-axis
- 可选分 benign/malignant 两条曲线或柱状图

### Appendix 图：代表性案例

内容：

- GT bbox 和 predicted bbox 叠加。
- 显示 IoU、probability、true/pred label。

## 6. 需要补写或改造的脚本

### 6.1 正式 detector 3-seed runner

建议新增：

`C:\Users\Afr1ste\PycharmProjects\Thyroid\run_tn5000_yolo_detector_3seed.py`

功能：

- seeds: 17, 27, 37
- 每个 seed 单独输出 run dir
- 汇总 `detector_model_level_summary.csv`
- 写 `latest.status.json`

### 6.2 IoU 分层诊断分析

建议新增：

`C:\Users\Afr1ste\PycharmProjects\Thyroid\analyze_tn5000_auto_roi_by_iou.py`

输入：

- detector predictions CSV
- classifier prediction CSV
- ground truth labels

输出：

- `auto_roi_iou_bucket_metrics.csv`
- `auto_roi_iou_bucket_metrics.tex`
- `fig_auto_roi_iou_error.png`

### 6.3 ROI crop rule sweep

建议新增：

`C:\Users\Afr1ste\PycharmProjects\Thyroid\run_tn5000_roi_crop_rule_sweep.py`

功能：

- reuse existing classifier checkpoints
- ROI source: oracle / auto
- geometry: rectangular / square
- expand ratios: 0.00, 0.10, 0.20, 0.30, 0.40, 0.50
- 输出 `roi_crop_rule_sweep.csv`

### 6.4 失败案例导出

建议新增：

`C:\Users\Afr1ste\PycharmProjects\Thyroid\export_tn5000_auto_roi_case_grid.py`

输出：

- `fig_auto_roi_cases.png`
- `appendix_auto_roi_cases.pdf`
- 每张图有 GT/pred bbox overlay

## 7. 风险与边界

### 7.1 不应过度声称

不建议写：

- “fully automatic clinical diagnosis system”
- “solves ROI annotation requirement”
- “novel lesion detector”

建议写：

- “automatic ROI feasibility analysis”
- “diagnosis-agnostic lesion localization front-end”
- “closed-loop full-image-to-ROI-to-diagnosis evaluation”

### 7.2 需要防止数据泄漏质疑

必须强调：

- detector 的 train/val/test 与 classifier split 一致。
- detector 只用 bbox，不用 benign/malignant diagnosis label。
- threshold 和 temperature 只在 val 上拟合。
- test split 不参与 detector 或 classifier 的任何选择。

### 7.3 当前图文一致性问题

当前分类代码真实逻辑是 expanded rectangular bbox crop。论文图如果继续画 square crop，会被审稿人追问。建议二选一：

1. 快速修稿：把图和文字改成 expanded ROI crop。
2. 补 P4 实验：如果 square crop 确实更好，再把 square crop 写入方法。

## 8. 执行顺序建议

最实际的执行顺序：

1. 新增 detector 3-seed runner，先跑 TN5000 YOLO11n 3 seeds。
2. 扩展 `eval_tn5000_auto_roi_pipeline.py`，支持 strict oracle-threshold transfer。
3. 写 IoU bucket 分析脚本，先复用 smoke 输出验证。
4. 写 crop rule sweep 脚本，先 inference-only sweep，不重训。
5. 导出案例图。
6. 只有当 auto ROI drop 过大时，再考虑 predicted-box augmentation 训练。

建议 stop condition：

- 如果 auto ROI AUC 相对 oracle ROI 下降 <= 1.5 pp，BalAcc 下降 <= 3 pp，可以作为较强补充实验。
- 如果 auto ROI AUC 下降 1.5-3 pp，但 full image 明显更差，仍可作为 feasibility analysis。
- 如果 auto ROI AUC 下降 > 5 pp，则不要放主文，只放 appendix，并把 ROI detector 作为未来工作。

当前 smoke 结果满足第一档：AUC 下降约 1.0 pp，BalAcc 下降约 2.1 pp。

