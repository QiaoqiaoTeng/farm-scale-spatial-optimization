import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import warnings
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# =====================================================================
# 1. 顶级期刊全局配置 (Nature/Science Style)
# =====================================================================
NPG_COLORS = {'Maize': '#E64B35', 'Soybean': '#4DBBD5', 'Rice': '#00A087', 'Dark': '#3C5488', 'Gray': '#7E6148'}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.linewidth': 1.2, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'legend.frameon': False, 'figure.dpi': 900, 'savefig.bbox': 'tight'
})


# =====================================================================
# 2. 数据准备与模型训练模拟
# =====================================================================
# 读取机器学习数据
ml_data = pd.read_csv("ML_Training_Data.csv")
crops = ["Maize", "Soybean", "Rice"]
print("✅ 成功加载 ML_Training_Data.csv。")



# 设定一个轻量级的目标函数以追踪进化历史
class QuickFarmOpt(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=2, xl=np.zeros(10), xu=np.ones(10) * 5)

    def _evaluate(self, x, out, *args, **kwargs):
        profit = np.sum(x) * 10000 - np.sum(x ** 2) * 500
        carbon = np.sum(x) * 500
        out["F"] = [-profit, carbon]


print("🧬 正在追踪 NSGA-II 多目标优化的代际进化轨迹...")
res = minimize(QuickFarmOpt(), NSGA2(pop_size=100), ('n_gen', 120), seed=42, save_history=True, verbose=False)

# 提取每一代的数据
gen_numbers, max_profits, min_carbons, hist_F = [], [], [], []
for algo in res.history:
    gen_numbers.append(algo.n_gen)
    F = algo.pop.get("F")
    max_profits.append(-np.min(F[:, 0]))
    min_carbons.append(np.min(F[:, 1]))
    hist_F.append(F)

# =====================================================================
# 3. 核心布局：GridSpec 划分布局 (三张放上面，两张放下面)
# =====================================================================
fig = plt.figure(figsize=(18, 11), dpi=900)
# 设定一个 2行6列 的网格，通过跨列来实现 3 vs 2 的排版
gs = GridSpec(1, 2, figure=fig, hspace=0.35, wspace=1.0)


# 下排 2 张图 (每张占 3 列)
ax_conv1 = fig.add_subplot(gs[0, 0])
ax_pareto = fig.add_subplot(gs[0, 1])


# =====================================================================
# 5. 绘制下排左侧 - 算法收敛曲线 (Convergence)
# =====================================================================
ax_conv2 = ax_conv1.twinx()

l1 = ax_conv1.plot(gen_numbers, max_profits, color=NPG_COLORS['Maize'], linewidth=2.5, label='Max Profit Discovery')
ax_conv1.set_xlabel('Evolution Generations', fontweight='bold')
ax_conv1.set_ylabel('Discovered Max Profit', fontweight='bold', color=NPG_COLORS['Maize'])
ax_conv1.tick_params(axis='y', labelcolor=NPG_COLORS['Maize'])

l2 = ax_conv2.plot(gen_numbers, min_carbons, color=NPG_COLORS['Soybean'], linewidth=2.5, linestyle='-',
                   label='Min Carbon Discovery')
ax_conv2.set_ylabel('Discovered Min Carbon', fontweight='bold', color=NPG_COLORS['Soybean'])
ax_conv2.tick_params(axis='y', labelcolor=NPG_COLORS['Soybean'])

lns = l1 + l2
labs = [l.get_label() for l in lns]
ax_conv1.legend(lns, labs, loc='center right', frameon=False)

ax_conv1.spines['top'].set_visible(False)
ax_conv2.spines['top'].set_visible(False)
ax_conv1.set_title("Algorithm Convergence Dynamics", fontweight="bold", pad=15)

# 添加左上角学术标注 D
ax_conv1.text(-0.12, 1.05, 'A', transform=ax_conv1.transAxes, fontsize=18, fontweight='bold')

# =====================================================================
# 6. 绘制下排右侧 - 帕累托前沿的动态推进 (Pareto Dynamics)
# =====================================================================
target_gens = [0, 29, 119]
colors_gen = ['#CCCCCC', '#88CCEE', NPG_COLORS['Dark']]
labels_gen = ['Generation 1', 'Generation 30', 'Generation 120 (Final Front)']

for g_idx, color, label in zip(target_gens, colors_gen, labels_gen):
    F_gen = hist_F[g_idx]
    profit_gen = -F_gen[:, 0]
    carbon_gen = F_gen[:, 1]

    size = 60 if g_idx == 119 else 30
    alpha = 0.9 if g_idx == 119 else 0.5
    ax_pareto.scatter(profit_gen, carbon_gen, color=color, s=size, alpha=alpha, label=label, edgecolor='white')

    if g_idx == 119:
        sort_idx = np.argsort(profit_gen)
        ax_pareto.plot(profit_gen[sort_idx], carbon_gen[sort_idx], color=NPG_COLORS['Dark'], linewidth=1.5,
                       linestyle='--')

start_pt = (np.mean(-hist_F[0][:, 0]), np.mean(hist_F[0][:, 1]))
end_pt = (np.mean(-hist_F[-1][:, 0]), np.mean(hist_F[-1][:, 1]))
ax_pareto.annotate('', xy=end_pt, xytext=start_pt,
                   arrowprops=dict(facecolor=NPG_COLORS['Maize'], shrink=0.05, width=2, headwidth=10, alpha=0.7))
ax_pareto.text((start_pt[0] + end_pt[0]) / 2, (start_pt[1] + end_pt[1]) / 2 + 200, 'Evolution\nDirection',
               color=NPG_COLORS['Maize'], fontweight='bold', ha='center')

ax_pareto.set_title("Pareto Front Evolution", fontweight="bold", pad=15)
ax_pareto.set_xlabel('Profit Objective (Maximize)', fontweight='bold')
ax_pareto.set_ylabel('Carbon Objective (Minimize)', fontweight='bold')
ax_pareto.spines['top'].set_visible(False)
ax_pareto.spines['right'].set_visible(False)
ax_pareto.legend(loc='upper right', frameon=False)

# 添加左上角学术标注 E
ax_pareto.text(-0.1, 1.05, 'B', transform=ax_pareto.transAxes, fontsize=18, fontweight='bold')

# =====================================================================
# 7. 渲染并保存组合大图
# =====================================================================
plt.tight_layout()
plt.savefig('Fig11_12_Combined_Algorithm_Diagnostics.png', facecolor='white', bbox_inches='tight')
plt.close()
print("✅ 完美组合大图已生成：'Fig11_12_Combined_Algorithm_Diagnostics.png'")