# IMS 光谱判别分析 

用离子迁移谱（Ion Mobility Spectrometry, IMS）谱线区分五组人群：

- **RA** — 类风湿关节炎（Rheumatoid Arthritis）
- **OA** — 骨关节炎（Osteoarthritis）
- **HC** — 健康对照（Healthy Control）
- **SLE** — 系统性红斑狼疮（Systemic Lupus Erythematosus）
- **SS** — 干燥综合征（Sjögren's Syndrome）

本仓库提供从 **PNG 光谱图像** 一路跑到 **判别结果与图表** 的完整、可复现代码。

---

## 数据要求

数据目录内放置光谱 PNG，命名规则 `组名-编号.png`，例如：

```
RA-1.png ... RA-100.png
OA-1.png ... OA-100.png
HC-1.png ... HC-100.png
SLE-1.png ... SLE-10.png
SS-1.png ... SS-11.png
```

图像为白色背景上的深色光谱曲线，横轴漂移时间 0–22 ms、纵轴强度。
脚本会自动处理两种尺寸：RA/OA/HC 为 945px 宽（用固定绘图区），SLE/SS 为 866px 宽（自动检测绘图区边界）。

---

## 环境安装

```bash
pip install -r requirements.txt
```

Python ≥ 3.8。核心依赖：numpy / scipy / scikit-learn / matplotlib / pillow；
`scikit-posthocs` 用于 Dunn's 事后检验（若缺失，脚本会自动跳过事后检验、仅保留 K-W 检验）。

---

## 运行

```bash
python ims_pipeline.py --data_dir <PNG所在目录> --output_dir <输出目录> [--groups 3|5]
```

- `--groups 5`（默认）：五组分析 RA/OA/HC/SLE/SS
- `--groups 3`：三组分析 RA/OA/HC
- 也可自定义：`--groups RA,OA,HC`

例如：

```bash
# 五组
python ims_pipeline.py --data_dir "D:\IMS Figure" --output_dir results

# 三组
python ims_pipeline.py --data_dir "D:\IMS Figure" --output_dir results3 --groups 3
```

输出目录会得到：

```
boxplots_5groups.png          五组箱线图（K-W + Dunn 事后检验）
pca_score_plot.png            PCA 得分图（top-12 判别特征 + 95% 置信椭圆）
roc_RA_vs_*.png / dca_RA_vs_*.png   每对二分类的 ROC 与决策曲线
logreg_coefficients.png       逻辑回归系数热图（RA/OA/HC one-vs-rest）
discriminant_metrics.txt      LDA 判别指标汇总（repeated CV, mean±SD）
```

---

## 分析流程与方法选择

| 步骤 | 方法 | 为什么选它 |
| --- | --- | --- |
| 图像数字化 | 像素灰度 → 光谱曲线 | 原始数据只有图片，需反推数值 |
| 特征提取 | P1-P20 固定特征峰强度 | 预先筛选出的关键漂移时间 |
| 组间差异 | Kruskal-Wallis 检验 | 强度偏态/异方差，非参数更稳健（不假设正态） |
| 两两比较 | Dunn's test + Bonferroni | 与 KW 自洽，并控制多重比较假阳性 |
| 降维可视化 | PCA（top-12 KW 特征） | 无监督找主要变异方向，不人为制造分离 |
| 判别分析 | LDA + repeated 5-fold CV | 小样本下稳定；重复 CV 报告 mean±SD |
| 可解释性 | 逻辑回归 one-vs-rest | 系数大小直接说明"哪个峰最重要" |
| 临床效用 | DCA（Net Benefit） | 从统计准确率落到临床净收益 |

## 复现性说明

- 随机种子固定（`np.random.seed(42)`，CV `random_state=42..51`），结果可精确复现。
- repeated CV 的 `N_REPEATS` 默认为 10，可在脚本顶部修改。
- 小样本组（SLE=10、SS=11）的 per-fold 指标波动大，对应 SD 偏大，属数据本身特性。
