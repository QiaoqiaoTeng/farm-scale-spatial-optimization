import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
# =====================================================================
# 1. 全局配置与参数变量区
# =====================================================================
# ---------------------------------------------------------
# 💥 核心算法与模型参数提取
# ---------------------------------------------------------
# 多目标优化 (NSGA-II) 参数
OPT_N_GEN = 200               # 进化代数 (原120)
OPT_POP_SIZE = 100            # 种群大小 (原100)
OPT_SEED = 42                 # 算法随机种子

# 机器学习代理模型 (Random Forest) 参数
RF_N_ESTIMATORS_SHAP = 100    # SHAP归因分析时的森林树木数量 (原100)
RF_N_ESTIMATORS_REG = 50      # 散点图回归验证时的森林数量 (原50)
RF_CV_FOLDS = 5               # 散点图交叉验证折数 (原5)
RF_RANDOM_STATE = 42          # 随机森林随机种子

# Mock数据生成参数 (当缺少csv时触发)
MOCK_SAMPLES_PER_CROP = 300   # 每个作物的模拟样本量 (原300)
MOCK_SEED = 42                # 模拟数据生成随机种子
# ---------------------------------------------------------

# 颜色主题
NPG_COLORS = {'Maize': '#E64B35', 'Soybean': '#4DBBD5', 'Rice': '#00A087', 'Dark': '#3C5488', 'Gray': '#7E6148'}

# 字体大小配置 (全部放大)
FONT_TITLE = 18  # 标题字体大小 (原16)
FONT_LABEL = 16  # 坐标轴标签字体大小 (原14)
FONT_TICK = 14  # 刻度线字体大小 (原12)
FONT_ANNOTATION = 18  # A, B 等学术标注的字体大小 (原18)
FONT_LEGEND = 14  # 图例字体大小 (原12)

# 图片尺寸配置
GLOBAL_DPI = 1200
FIG_SIZE_ALGO = (18, 8)  # 算法图尺寸
FIG_SIZE_SHAP = (20, 9)  # SHAP 图尺寸

# Matplotlib 全局环境配置
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.weight': 'bold',  # 💥 核心修改：全局字体加粗 (涵盖刻度、图例等所有文本)
    'font.size': FONT_TICK,
    'axes.labelsize': FONT_LABEL,
    'axes.titlesize': FONT_TITLE,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.5,  # 💥 配合加粗字体，边框线宽也从 1.2 加粗到了 1.5
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.fontsize': FONT_LEGEND,
    'legend.frameon': False,
    'figure.dpi': GLOBAL_DPI,
    'savefig.bbox': 'tight'
})

# 屏蔽 pymoo 的 C++ 加速警告 (让终端输出更干净)
from pymoo.config import Config

Config.warnings['not_compiled'] = False

print("🚀 正在启动全量可视化分析流水线...")

# =====================================================================
# 2. 统一的数据加载与预处理
# =====================================================================
print("📂 [1/8] 正在加载数据...")
# 2.1 加载机器学习训练数据 (带降级 Mock 方案)
try:
    df_ml = pd.read_csv("ML_Training_Data.csv")
    crops = ["Maize", "Soybean", "Rice"]
    print("✅ 成功加载 ML_Training_Data.csv。")
except Exception as e:
    print("⚠️ 未找到 ML_Training_Data.csv，正在生成带噪声的模拟训练集用于演示...")
    np.random.seed(MOCK_SEED)
    mock_ml = []
    for crop in ["Maize", "Soybean", "Rice"]:
        for _ in range(MOCK_SAMPLES_PER_CROP):
            dens = np.random.uniform(15, 40) if crop == 'Rice' else (
                np.random.uniform(3, 18) if crop == 'Maize' else np.random.uniform(10, 35))
            fert = np.random.uniform(280, 500) if crop == 'Rice' else (
                np.random.uniform(140, 400) if crop == 'Maize' else np.random.uniform(40, 300))
            base_y = 9000 if crop == 'Rice' else (10500 if crop == 'Maize' else 3000)
            yield_val = base_y + 15 * fert - 0.02 * fert ** 2 + 50 * dens - 0.5 * dens ** 2 + np.random.normal(0,
                                                                                                               base_y * 0.05)
            et_val = 400 + 0.1 * fert + 2 * dens + np.random.normal(0, 20)
            mock_ml.append(
                {'Crop': crop, 'Density_plants_m2': dens, 'Fert_Weight_kg_ha': fert, 'Yield_kg_ha': yield_val,
                 'ET_mm': et_val})
    df_ml = pd.DataFrame(mock_ml)
    crops = ["Maize", "Soybean", "Rice"]

# 2.2 加载优化结果数据
try:
    df_pareto = pd.read_csv("Pareto_Front_Data.csv")
    df_plan = pd.read_csv("Optimized_Farm_Plan_Continuous.csv")
    df_metrics = pd.read_csv("All_Metrics_Data_Continuous.csv")
    print("✅ 成功加载帕累托与农场计划数据。")
except Exception as e:
    raise RuntimeError(f"未找到 CSV 优化结果数据，请确保相关数据文件在当前目录下。错误详情: {e}")

# 2.3 字段兼容性统一 (统一重命名为 N_Fert_kg_ha)
if 'Fert_Weight_kg_ha' in df_plan.columns:
    df_plan.rename(columns={'Fert_Weight_kg_ha': 'N_Fert_kg_ha'}, inplace=True)
if 'Fert_Weight_kg_ha' in df_ml.columns:
    df_ml.rename(columns={'Fert_Weight_kg_ha': 'N_Fert_kg_ha'}, inplace=True)

# =====================================================================
# 3. 算法诊断图 (1行2列并排)
# =====================================================================
print("📊 [2/8] 正在生成算法诊断与演化图...")

# 构造极简测试目标追踪演化轨迹
class QuickFarmOpt(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=2, xl=np.zeros(10), xu=np.ones(10) * 5)

    def _evaluate(self, x, out, *args, **kwargs):
        profit = np.sum(x) * 10000 - np.sum(x ** 2) * 500
        carbon = np.sum(x) * 500
        out["F"] = [-profit, carbon]

res = minimize(QuickFarmOpt(), NSGA2(pop_size=OPT_POP_SIZE), ('n_gen', OPT_N_GEN), seed=OPT_SEED, save_history=True, verbose=False)

gen_numbers, max_profits, min_carbons, hist_F = [], [], [], []
for algo in res.history:
    gen_numbers.append(algo.n_gen)
    F = algo.pop.get("F")
    max_profits.append(-np.min(F[:, 0]))
    min_carbons.append(np.min(F[:, 1]))
    hist_F.append(F)

# 创建 1行2列 布局画布
fig_diag = plt.figure(figsize=FIG_SIZE_ALGO, dpi=GLOBAL_DPI)
gs_diag = GridSpec(1, 2, figure=fig_diag, wspace=0.3)

ax_conv1 = fig_diag.add_subplot(gs_diag[0, 0])
ax_pareto = fig_diag.add_subplot(gs_diag[0, 1])

# [图 A] 绘制算法收敛曲线 (Convergence)
ax_conv2 = ax_conv1.twinx()
l1 = ax_conv1.plot(gen_numbers, max_profits, color=NPG_COLORS['Maize'], linewidth=2.5, label='Max Profit Discovery')
ax_conv1.set_xlabel('Evolution Generations')
ax_conv1.set_ylabel('Discovered Max Profit', color=NPG_COLORS['Maize'])
ax_conv1.tick_params(axis='y', labelcolor=NPG_COLORS['Maize'])

l2 = ax_conv2.plot(gen_numbers, min_carbons, color=NPG_COLORS['Soybean'], linewidth=2.5, linestyle='-',
                   label='Min Carbon Discovery')
ax_conv2.set_ylabel('Discovered Min Carbon', color=NPG_COLORS['Soybean'], labelpad=15)
ax_conv2.tick_params(axis='y', labelcolor=NPG_COLORS['Soybean'])

ax_conv1.legend(l1 + l2, [l.get_label() for l in (l1 + l2)], loc='center right', frameon=False)
ax_conv2.spines['top'].set_visible(False)
ax_conv1.set_title("Algorithm Convergence Dynamics", pad=15)
ax_conv1.text(-0.12, 1.05, 'A', transform=ax_conv1.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')

# [图 B] 绘制帕累托前沿动态推进 (Pareto Dynamics) 自适应修改参数
last_gen_idx = OPT_N_GEN - 1
mid_gen_idx = min(29, max(0, last_gen_idx - 1))
target_gens = [0, mid_gen_idx, last_gen_idx]
colors_gen = ['#CCCCCC', '#88CCEE', NPG_COLORS['Dark']]
labels_gen = ['Generation 1', f'Generation {mid_gen_idx + 1}', f'Generation {OPT_N_GEN} (Final Front)']

for g_idx, color, label in zip(target_gens, colors_gen, labels_gen):
    F_gen = hist_F[g_idx]
    profit_gen = -F_gen[:, 0]
    carbon_gen = F_gen[:, 1]

    ax_pareto.scatter(profit_gen, carbon_gen, color=color, s=(80 if g_idx == last_gen_idx else 40),
                      alpha=(0.9 if g_idx == last_gen_idx else 0.5), label=label, edgecolor='white')
    if g_idx == last_gen_idx:
        ax_pareto.plot(profit_gen[np.argsort(profit_gen)], carbon_gen[np.argsort(profit_gen)],
                       color=NPG_COLORS['Dark'], linewidth=1.5, linestyle='--')

start_pt = (np.mean(-hist_F[0][:, 0]), np.mean(hist_F[0][:, 1]))
end_pt = (np.mean(-hist_F[-1][:, 0]), np.mean(hist_F[-1][:, 1]))
ax_pareto.annotate('', xy=end_pt, xytext=start_pt,
                   arrowprops=dict(facecolor=NPG_COLORS['Maize'], shrink=0.05, width=2, headwidth=10, alpha=0.7))
ax_pareto.text((start_pt[0] + end_pt[0]) / 2, (start_pt[1] + end_pt[1]) / 2 + 200, 'Evolution\nDirection',
               color=NPG_COLORS['Maize'], fontweight='bold', ha='center', fontsize=FONT_TICK)

ax_pareto.set_title("Pareto Front Evolution", pad=15)
ax_pareto.set_xlabel('Profit Objective (Maximize)')
ax_pareto.set_ylabel('Carbon Objective (Minimize)')
ax_pareto.legend(loc='upper right', frameon=False)
ax_pareto.text(-0.1, 1.05, 'B', transform=ax_pareto.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')

plt.tight_layout()
plt.savefig('Fig_Algorithm_Diagnostics.png', facecolor='white', bbox_inches='tight')
plt.close()

# =====================================================================
# 4. 帕累托前沿深度分析图 (Fig 13 & Fig 14 Pareto Projection)
# =====================================================================
print("🔍 [3/8] 正在生成帕累托前沿分析图 (Fig 13, Fig 14)...")
df_p_plot = df_pareto.copy()
df_p_plot['Profit (10k CNY)'] = df_p_plot['Profit'] / 10000
df_p_plot['Carbon (t CO2e)'] = df_p_plot['Carbon'] / 1000
df_p_plot = df_p_plot[['Profit (10k CNY)', 'WUE', 'Carbon (t CO2e)']]

g = sns.PairGrid(df_p_plot, diag_sharey=False, corner=False, height=2.5)
g.map_lower(sns.scatterplot, alpha=0.7, color=NPG_COLORS['Dark'], edgecolor='white')
g.map_upper(sns.kdeplot, fill=True, cmap='Blues', alpha=0.7)
g.map_diag(sns.histplot, kde=True, color=NPG_COLORS['Soybean'])
plt.savefig('Fig 13 Pairwise Trade-off Matrix.png')
plt.close()

fig14, ax14 = plt.subplots(figsize=(9, 6), dpi=GLOBAL_DPI)
sizes = (df_p_plot['WUE'] - df_p_plot['WUE'].min()) / (df_p_plot['WUE'].max() - df_p_plot['WUE'].min())
bubble_sizes = sizes * 400 + 50
best_p_idx = df_p_plot['Profit (10k CNY)'].idxmax()

scatter14 = ax14.scatter(df_p_plot['Profit (10k CNY)'], df_p_plot['Carbon (t CO2e)'], s=bubble_sizes,
                         c=df_p_plot['WUE'], cmap='coolwarm', alpha=0.8, edgecolors='black', linewidth=0.5)
ax14.scatter(df_p_plot.loc[best_p_idx, 'Profit (10k CNY)'], df_p_plot.loc[best_p_idx, 'Carbon (t CO2e)'], color='red',
             marker='*', s=400, edgecolor='black', linewidth=1.5, zorder=10, label='Max Profit Strategy')
cbar14 = fig14.colorbar(scatter14, ax=ax14, pad=0.02)
cbar14.set_label('System WUE (kg/m³)')
ax14.set_xlabel('Total Profit (10,000 CNY)')
ax14.set_ylabel('Total Carbon Footprint (Tons CO2e)')
ax14.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Fig 14_Pareto_Projection_Bubble.png')
plt.close()

# =====================================================================
# 5. 经济分析图 (Fig 10)
# =====================================================================
print("📈 [4/8] 正在渲染经济学分析 (Fig 10)...")
df_eco = df_metrics.copy()
crop_col = 'Crop' if 'Crop' in df_eco.columns else 'Optimized_Crop'
df_eco['Total_Profit_10k'] = (df_eco['Profit_ha'] * df_eco['Area_ha']) / 10000

fig_eco = plt.figure(figsize=(20, 6), dpi=GLOBAL_DPI)
gs_eco = GridSpec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.25)

# Panel A
axA = fig_eco.add_subplot(gs_eco[0])
df_sorted_profit = df_eco.sort_values(by='Profit_ha', ascending=False).reset_index(drop=True)
colors_A = [NPG_COLORS[c] for c in df_sorted_profit[crop_col]]
axA.bar(range(len(df_sorted_profit)), df_sorted_profit['Profit_ha'], color=colors_A, width=0.8, edgecolor='black',
        linewidth=0.5)

axA.text(-0.05, 1.05, 'A', transform=axA.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
axA.set_ylabel('Net Profit (CNY/ha)')
axA.set_xlabel('54 Fields (Ranked by Profit Margin)')
axA.set_xticks([])
axA.axhline(df_eco['Profit_ha'].mean(), color='#555555', linestyle='--', linewidth=1.5, zorder=0)

legend_handles = [mpatches.Patch(color=NPG_COLORS[c], label=c) for c in ['Maize', 'Soybean', 'Rice']]
axA.legend(handles=legend_handles, title='Optimized Crop', loc='upper right', frameon=False)

# Panel B
axB = fig_eco.add_subplot(gs_eco[1])
for c in ['Maize', 'Soybean', 'Rice']:
    sub = df_eco[df_eco[crop_col] == c]
    if not sub.empty:
        axB.scatter(sub['Area_ha'], sub['Profit_ha'], s=sub['Total_Profit_10k'] * 3 + 30, color=NPG_COLORS[c],
                    alpha=0.75, edgecolor='white', linewidth=1, label=c)
axB.text(-0.05, 1.05, 'B', transform=axB.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
axB.axvline(df_eco['Area_ha'].median(), color='#999999', linestyle=':', zorder=0)
axB.axhline(df_eco['Profit_ha'].median(), color='#999999', linestyle=':', zorder=0)
axB.set_xlabel('Field Area (ha)')
axB.set_ylabel('Net Profit (CNY/ha)')

# Panel C
axC = fig_eco.add_subplot(gs_eco[2])
df_lorenz = df_eco.sort_values('Profit_ha', ascending=False)
cum_area = np.insert((df_lorenz['Area_ha'].cumsum() / df_lorenz['Area_ha'].sum() * 100).values, 0, 0)
cum_profit = np.insert((df_lorenz['Total_Profit_10k'].cumsum() / df_lorenz['Total_Profit_10k'].sum() * 100).values, 0,
                       0)
axC.text(-0.05, 1.05, 'C', transform=axC.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
axC.plot(cum_area, cum_profit, color=NPG_COLORS['Dark'], linewidth=2.5, label='Actual Profit Curve')
axC.plot([0, 100], [0, 100], color='gray', linestyle='--', linewidth=1.5)
axC.fill_between(cum_area, cum_area, cum_profit, color=NPG_COLORS['Soybean'], alpha=0.15)
axC.set_xlabel('Cumulative Area (%)')
axC.set_ylabel('Cumulative Total Profit (%)')
axC.set_xlim(0, 100)
axC.set_ylim(0, 100)
plt.savefig('Fig 10 Economic Benefit Analysis.png', facecolor='white')
plt.close()

# =====================================================================
# 6. 三品种农艺响应空间热力图 (Fig 09)
# =====================================================================
print("🌡️ [5/8] 正在生成三品种农艺响应空间热力图 (Fig 09)...")
fig09, axes09 = plt.subplots(1, 3, figsize=(18, 5), dpi=GLOBAL_DPI)
cmap_dict = {'Soybean': 'Blues', 'Rice': 'Greens', 'Maize': 'Reds'}
target_z = 'Yield_kg_ha' if 'Yield_kg_ha' in df_ml.columns else 'Profit_ha'

for i, (ax, crop) in enumerate(zip(axes09, crops)):
    sub_df = df_ml[df_ml['Crop'] == crop]
    if len(sub_df) > 5:
        x, y, z = sub_df['Density_plants_m2'], sub_df['N_Fert_kg_ha'], sub_df[target_z]
        cntr = ax.tricontourf(x, y, z, levels=20, cmap=cmap_dict[crop], alpha=0.9)
        ax.tricontour(x, y, z, levels=20, colors='black', linewidths=0.3, alpha=0.5)
        cbar = fig09.colorbar(cntr, ax=ax, pad=0.02)
        cbar.set_label(f'{target_z}')
        ax.set_title(f"{crop}: Density vs Fertilizer", color=NPG_COLORS[crop])
        ax.set_xlabel("Density (plants/m²)")
        ax.set_ylabel("Fertilizer (kg/ha)")
    # 新增编号 A, B, C
    ax.text(-0.1, 1.05, chr(65 + i), transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')

plt.tight_layout()
plt.savefig('Fig09_Agronomic_Response_Space.png', bbox_inches='tight', facecolor='white')
plt.close()
# =====================================================================
# 7. SHAP 机器学习特征归因分析 (Fig 16 - 修复重叠问题的 2x2 网格)
# =====================================================================
print("🧠 [6/8] 正在生成 SHAP 特征归因分析图 (Fig 16)...")
try:
    df_ml_model = df_ml.copy()
    target_col = 'Profit_ha' if 'Profit_ha' in df_ml_model.columns else 'Yield_kg_ha'
    base_inputs = ['N_Fert_kg_ha', 'P_Fert_kg_ha', 'K_Fert_kg_ha', 'Density_plants_m2', 'Irrigation_mm']

    # 💥 调整尺寸为 20x14，给水平特征名和垂直分布留出足够空间
    fig_shap, axes_shap = plt.subplots(2, 2, figsize=(20, 14), dpi=GLOBAL_DPI)
    axes_flat = axes_shap.flatten()

    features_cols_all = [col for col in base_inputs + ['Crop'] if col in df_ml_model.columns]
    X_raw_all = df_ml_model[features_cols_all]
    X_all = pd.get_dummies(X_raw_all, drop_first=True)
    y_all = df_ml_model[target_col]

    rf_all = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS_SHAP, random_state=RF_RANDOM_STATE, n_jobs=-1)
    rf_all.fit(X_all, y_all)
    explainer_all = shap.TreeExplainer(rf_all)
    shap_values_all = explainer_all.shap_values(X_all)

    # ------------------
    # 【图A】Overall System (左上角)
    # ------------------
    plt.sca(axes_flat[0])
    # 💥 关键修复：必须加上 plot_size=None，否则 SHAP 会强制修改整个画布的高度导致挤压
    shap.summary_plot(shap_values_all, X_all, show=False, cmap='coolwarm', max_display=10, color_bar=True,
                      plot_size=None)
    axes_flat[0].set_title("Overall System", pad=15)
    axes_flat[0].set_xlabel("SHAP Value (Impact)")
    # 调整角标位置，避免和长特征名重叠
    axes_flat[0].text(-0.25, 1.05, 'A', transform=axes_flat[0].transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')

    target_crops = ['Maize', 'Soybean', 'Rice']
    for i, crop_name in enumerate(target_crops):
        ax = axes_flat[i + 1]
        df_crop = df_ml_model[df_ml_model['Crop'] == crop_name]
        features_cols_crop = [col for col in base_inputs if col in df_crop.columns]
        X_crop = df_crop[features_cols_crop]
        y_crop = df_crop[target_col]

        rf_crop = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS_SHAP, random_state=RF_RANDOM_STATE, n_jobs=-1)
        rf_crop.fit(X_crop, y_crop)
        explainer_crop = shap.TreeExplainer(rf_crop)
        shap_values_crop = explainer_crop.shap_values(X_crop)

        plt.sca(ax)
        # 💥 关键修复：同样加上 plot_size=None
        shap.summary_plot(shap_values_crop, X_crop, show=False, cmap='coolwarm', max_display=10, color_bar=True,
                          plot_size=None)

        ax.set_title(f"{crop_name} Only", pad=15, color=NPG_COLORS[crop_name])
        ax.set_xlabel("SHAP Value (Impact)")

        # 调整角标位置
        ax.text(-0.25, 1.05, chr(66 + i), transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')

    # 清理边框样式并调整 Y 轴标签字体大小，防止拥挤
    for ax in axes_flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0, labelsize=16)  # 稍微缩小特征名的字号

    # 💥 增大水平和垂直的间距，确保文字、色条、散点互不干扰
    plt.tight_layout(pad=3.0, w_pad=6.0, h_pad=5.0)
    plt.savefig('Fig16_SHAP_2x2_Grid_Beeswarm.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Fig 16 SHAP 优化版（修复重叠问题的 2x2 网格）生成成功！")

except Exception as e:
    print(f"⚠️ SHAP 图表生成失败，错误信息: {e}")
# =====================================================================
# 8. 农艺处方小提琴图 (Fig 14 - Agronomic Prescriptions Violin)
# =====================================================================
print("🌾 [7/8] 正在生成农艺处方小提琴图 (Agronomic Prescriptions Violin)...")
try:
    fig_presc, axes_presc = plt.subplots(1, 3, figsize=(18, 6), dpi=GLOBAL_DPI)
    target_z = 'Yield_kg_ha' if 'Yield_kg_ha' in df_ml.columns else 'Profit_ha'

    crops_order = ['Maize', 'Soybean', 'Rice']
    metrics = ['Density_plants_m2', 'N_Fert_kg_ha', target_z]
    y_labels = ['Density Prescriptions (plants/m²)', 'Fertilizer Prescriptions (kg/ha)', f'{target_z} Distribution']
    titles = ['Density Allocation', 'Fertilizer Allocation', 'Outcome Distribution']

    for i, (ax, metric, ylabel, title) in enumerate(zip(axes_presc, metrics, y_labels, titles)):
        sns.violinplot(x='Crop', y=metric, data=df_ml, order=crops_order, palette=NPG_COLORS, ax=ax, inner="quartile",
                       alpha=0.8, linewidth=1.5)
        sns.stripplot(x='Crop', y=metric, data=df_ml, order=crops_order, color='black', alpha=0.15, size=2.5, ax=ax,
                      jitter=True)

        ax.set_title(title, pad=15)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # 新增编号 A, B, C
        ax.text(-0.15, 1.05, chr(65 + i), transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Fig14_Agronomic_Prescriptions_Violin.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Fig 14 农艺处方小提琴图生成成功！")

except Exception as e:
    print(f"⚠️ 农艺处方小提琴图表生成失败，错误信息: {e}")

# =====================================================================
# 9. 代理模型回归散点图 (Yield & ET) - Predicted vs Observed
# =====================================================================
print("📉 [8/8] 正在生成代理模型回归散点图 (Yield & ET)...")
try:
    fig_scatter, axes_scatter = plt.subplots(2, 3, figsize=(16, 10), dpi=GLOBAL_DPI)
    targets = [('Yield_kg_ha', 'Yield (kg/ha)'), ('ET_mm', 'ET (mm)')]

    for row_idx, (target_col, target_label) in enumerate(targets):
        if target_col not in df_ml.columns:
            print(f"⚠️ 未在数据集中找到 {target_col} 列，跳过该指标。")
            continue

        for col_idx, crop in enumerate(crops):
            ax = axes_scatter[row_idx, col_idx]
            df_c = df_ml[df_ml['Crop'] == crop].dropna(subset=['Density_plants_m2', 'N_Fert_kg_ha', target_col])

            if len(df_c) < 5:
                continue

            X = df_c[['Density_plants_m2', 'N_Fert_kg_ha']]
            y_true = df_c[target_col]

            # 使用交叉验证避免过拟合并直接获取拟合预测结果
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS_REG, random_state=RF_RANDOM_STATE)
            y_pred = cross_val_predict(rf, X, y_true, cv=RF_CV_FOLDS)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            # 绘制带有置信区间的回归散点
            sns.regplot(x=y_true, y=y_pred, ax=ax, color=NPG_COLORS[crop],
                        scatter_kws={'alpha': 0.6, 'edgecolor': 'white', 's': 50},
                        line_kws={'color': '#555555', 'linestyle': '--', 'linewidth': 2})

            # 绘制 1:1 参考线
            min_val = min(y_true.min(), y_pred.min()) * 0.95
            max_val = max(y_true.max(), y_pred.max()) * 1.05
            ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1.5, alpha=0.3, label='1:1 Reference')

            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

            # 设置标题（仅第一行需要）
            if row_idx == 0:
                ax.set_title(f'{crop}', color=NPG_COLORS[crop], pad=10)

            ax.set_xlabel(f'DSSAT Simulated {target_label}')
            if col_idx == 0:
                ax.set_ylabel(f'RF Predicted {target_label}')

            # 添加 R2 和 RMSE 文本框
            ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}\n$RMSE$ = {rmse:.1f}', transform=ax.transAxes, fontsize=FONT_TICK,
                    fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    # 添加 A-F 编号 (这里原来已经写得很好了，继续保留)
    labels_scatter = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, ax in enumerate(axes_scatter.flat):
        ax.text(-0.15, 1.05, labels_scatter[i], transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')

    plt.tight_layout(pad=2.0)
    plt.savefig('Fig_Surrogate_Regression_Scatter.png', facecolor='white', bbox_inches='tight')
    plt.close()
    print("✅ 代理模型回归散点图生成成功！('Fig_Surrogate_Regression_Scatter.png')")
except Exception as e:
    print(f"⚠️ 代理模型回归散点图生成失败，错误信息: {e}")

print("🎉 全部可视化脚本执行完毕！请查看当前目录下的 png 图片。")