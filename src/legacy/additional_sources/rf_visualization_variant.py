import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import shap
from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# 1. 全局顶级期刊配置 (Nature/Science 风格)
# =====================================================================
NPG_COLORS = {'Maize': '#E64B35', 'Soybean': '#4DBBD5', 'Rice': '#00A087', 'Dark': '#3C5488', 'Gray': '#7E6148'}
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'axes.linewidth': 1.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False, 'figure.dpi': 600,
    'savefig.bbox': 'tight'
})

# =====================================================================
# 2. 加载新版数据并进行兼容性对齐
# =====================================================================
print("📂 正在加载数据...")
try:
    df_pareto = pd.read_csv("Pareto_Front_Data.csv")
    df_plan = pd.read_csv("Optimized_Farm_Plan_Continuous.csv")
    df_metrics = pd.read_csv("All_Metrics_Data_Continuous.csv")
    df_ml = pd.read_csv("ML_Training_Data.csv")
except Exception as e:
    raise RuntimeError(f"未找到 CSV 数据，请确保相关数据文件在当前目录下。错误详情: {e}")

if 'Fert_Weight_kg_ha' in df_plan.columns:
    df_plan.rename(columns={'Fert_Weight_kg_ha': 'N_Fert_kg_ha'}, inplace=True)
if 'Fert_Weight_kg_ha' in df_ml.columns:
    df_ml.rename(columns={'Fert_Weight_kg_ha': 'N_Fert_kg_ha'}, inplace=True)

# ================= 3. Pareto Front 深度分析图 =================
print("🔍 正在生成帕累托前沿分析图 (Fig 13, Fig 14)...")

df_p_plot = df_pareto.copy()
df_p_plot['Profit (10k CNY)'] = df_p_plot['Profit'] / 10000
df_p_plot['Carbon (t CO2e)'] = df_p_plot['Carbon'] / 1000
df_p_plot = df_p_plot[['Profit (10k CNY)', 'WUE', 'Carbon (t CO2e)']]

# --- Fig 13 ---
g = sns.PairGrid(df_p_plot, diag_sharey=False, corner=False, height=2.5)
g.map_lower(sns.scatterplot, alpha=0.7, color=NPG_COLORS['Dark'], edgecolor='white')
g.map_upper(sns.kdeplot, fill=True, cmap='Blues', alpha=0.7)
g.map_diag(sns.histplot, kde=True, color=NPG_COLORS['Soybean'])
plt.savefig('Fig 13 Pairwise Trade-off Matrix.png')
plt.close()

# --- Fig 14 ---
fig14, ax14 = plt.subplots(figsize=(9, 6), dpi=600)
sizes = (df_p_plot['WUE'] - df_p_plot['WUE'].min()) / (df_p_plot['WUE'].max() - df_p_plot['WUE'].min())
bubble_sizes = sizes * 400 + 50
best_p_idx = df_p_plot['Profit (10k CNY)'].idxmax()

scatter14 = ax14.scatter(df_p_plot['Profit (10k CNY)'], df_p_plot['Carbon (t CO2e)'],
                         s=bubble_sizes, c=df_p_plot['WUE'], cmap='coolwarm',
                         alpha=0.8, edgecolors='black', linewidth=0.5)
ax14.scatter(df_p_plot.loc[best_p_idx, 'Profit (10k CNY)'], df_p_plot.loc[best_p_idx, 'Carbon (t CO2e)'],
             color='red', marker='*', s=400, edgecolor='black', linewidth=1.5, zorder=10, label='Max Profit Strategy')
cbar14 = fig14.colorbar(scatter14, ax=ax14, pad=0.02)
cbar14.set_label('System WUE (kg/m³)', fontweight='bold')
ax14.set_xlabel('Total Profit (10,000 CNY)', fontweight='bold')
ax14.set_ylabel('Total Carbon Footprint (Tons CO2e)', fontweight='bold')
ax14.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Fig 14 2D Pareto Projection Bubble.png')
plt.close()

# ================= 4. 经济分析图 (Fig 10) =================
print("📊 正在渲染经济学分析 (Fig 10)...")
df_eco = df_metrics.copy()
crop_col = 'Crop' if 'Crop' in df_eco.columns else 'Optimized_Crop'
df_eco['Total_Profit_10k'] = (df_eco['Profit_ha'] * df_eco['Area_ha']) / 10000

fig_eco = plt.figure(figsize=(20, 6), dpi=600)
gs_eco = GridSpec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.25)

# Panel A
axA = fig_eco.add_subplot(gs_eco[0])
df_sorted_profit = df_eco.sort_values('Profit_ha', ascending=False).reset_index(drop=True)
colors_A = [NPG_COLORS[c] for c in df_sorted_profit[crop_col]]
axA.bar(df_sorted_profit.index, df_sorted_profit['Profit_ha'], color=colors_A, width=0.8, edgecolor='black',
        linewidth=0.5)
axA.text(-0.05, 1.05, 'A', transform=axA.transAxes, fontsize=16, fontweight='bold')
axA.set_ylabel('Net Profit (CNY/ha)', fontweight='bold')
axA.set_xlabel('54 Fields (Ranked by Profit Margin)', fontweight='bold')
axA.set_xticks([])
mean_profit = df_eco['Profit_ha'].mean()
axA.axhline(mean_profit, color='#555555', linestyle='--', linewidth=1.5, zorder=0)
axA.legend(handles=[mpatches.Patch(color=NPG_COLORS[c], label=c) for c in ['Maize', 'Soybean', 'Rice']],
           title='Optimized Crop', loc='upper right', frameon=False)

# Panel B
axB = fig_eco.add_subplot(gs_eco[1])
for c in ['Maize', 'Soybean', 'Rice']:
    sub = df_eco[df_eco[crop_col] == c]
    if not sub.empty:
        axB.scatter(sub['Area_ha'], sub['Profit_ha'], s=sub['Total_Profit_10k'] * 3 + 30, color=NPG_COLORS[c],
                    alpha=0.75, edgecolor='white', linewidth=1, label=c)
axB.text(-0.05, 1.05, 'B', transform=axB.transAxes, fontsize=16, fontweight='bold')
axB.axvline(df_eco['Area_ha'].median(), color='#999999', linestyle=':', zorder=0)
axB.axhline(df_eco['Profit_ha'].median(), color='#999999', linestyle=':', zorder=0)
axB.set_xlabel('Field Area (ha)', fontweight='bold');
axB.set_ylabel('Net Profit (CNY/ha)', fontweight='bold')

# Panel C
axC = fig_eco.add_subplot(gs_eco[2])
df_lorenz = df_eco.sort_values('Profit_ha', ascending=False)
cum_area = np.insert((df_lorenz['Area_ha'].cumsum() / df_lorenz['Area_ha'].sum() * 100).values, 0, 0)
cum_profit = np.insert((df_lorenz['Total_Profit_10k'].cumsum() / df_lorenz['Total_Profit_10k'].sum() * 100).values, 0,
                       0)
axC.text(-0.05, 1.05, 'C', transform=axC.transAxes, fontsize=16, fontweight='bold')
axC.plot(cum_area, cum_profit, color=NPG_COLORS['Dark'], linewidth=2.5, label='Actual Profit Curve')
axC.plot([0, 100], [0, 100], color='gray', linestyle='--', linewidth=1.5)
axC.fill_between(cum_area, cum_area, cum_profit, color=NPG_COLORS['Soybean'], alpha=0.15)
axC.set_xlabel('Cumulative Area (%)', fontweight='bold');
axC.set_ylabel('Cumulative Total Profit (%)', fontweight='bold')
axC.set_xlim(0, 100);
axC.set_ylim(0, 100)
plt.savefig('Fig 10 Economic Benefit Analysis.png', facecolor='white')
plt.close()

# ================= 5. 三品种农艺响应空间热力图 (Fig 09) =================
print("🌡️ 正在生成三品种农艺响应空间热力图 (Fig 09)...")
fig09, axes09 = plt.subplots(1, 3, figsize=(18, 5), dpi=600)
crops = ['Soybean', 'Rice', 'Maize']

# 优先使用训练集数据绘制连续的响应面
df_heat = df_ml.copy()
target_z = 'Yield_kg_ha' if 'Yield_kg_ha' in df_heat.columns else 'Profit_ha'

# 【核心修复】：将热力图色带与 NPG_COLORS 全局配色保持绝对一致
cmap_dict = {'Soybean': 'Blues', 'Rice': 'Greens', 'Maize': 'Reds'}

for ax, crop in zip(axes09, crops):
    sub_df = df_heat[df_heat['Crop'] == crop]
    if len(sub_df) > 5:  # 确保有足够的数据点插值
        x = sub_df['Density_plants_m2']
        y = sub_df['N_Fert_kg_ha']
        z = sub_df[target_z]

        # 使用 tricontourf 绘制不规则散点的平滑等高线热力图
        cntr = ax.tricontourf(x, y, z, levels=20, cmap=cmap_dict[crop], alpha=0.9)
        ax.tricontour(x, y, z, levels=20, colors='black', linewidths=0.3, alpha=0.5)

        cbar = fig09.colorbar(cntr, ax=ax, pad=0.02)
        cbar.set_label(f'{target_z}', fontweight='bold')

        # 标题颜色同样使用 NPG_COLORS，两者完美匹配
        ax.set_title(f"{crop}: Density vs Nitrogen", fontweight='bold', fontsize=14, color=NPG_COLORS[crop])
        ax.set_xlabel("Density (plants/m²)", fontweight='bold')
        ax.set_ylabel("Nitrogen Fertilizer (kg/ha)", fontweight='bold')

plt.tight_layout()
plt.savefig('Fig09_Agronomic_Response_Space.png', bbox_inches='tight', facecolor='white')
plt.close()

# ================= 6. 修复后的 SHAP 机器学习特征归因分析 (Fig 16) =================
print("🧠 正在生成修复后的 SHAP 特征归因分析图 (Fig 16)...")
try:
    df_ml_model = df_ml.copy()
    target_col = 'Profit_ha' if 'Profit_ha' in df_ml_model.columns else 'Yield_kg_ha'

    # 严禁引入 Yield, Leaching, Carbon 等结果变量作为特征！
    # 仅保留农场主能够真实干预的“决策变量”（Decision Variables）
    allowed_inputs = ['N_Fert_kg_ha', 'P_Fert_kg_ha', 'K_Fert_kg_ha', 'Density_plants_m2', 'Irrigation_mm', 'Crop']

    # 提取存在的合法特征
    features_cols = [col for col in allowed_inputs if col in df_ml_model.columns]
    X_raw = df_ml_model[features_cols]

    # 独热编码作物类型
    X = pd.get_dummies(X_raw, drop_first=True)
    y = df_ml_model[target_col]

    # 训练模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # 计算 SHAP
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # 绘制
    plt.figure(figsize=(10, 6), dpi=600)
    shap.summary_plot(shap_values, X, show=False, cmap='coolwarm', max_display=10)

    ax = plt.gca()
    ax.set_title(f"SHAP Feature Importance (Strictly Input Variables) -> {target_col}", fontweight='bold', fontsize=14,
                 pad=20)
    ax.set_xlabel("SHAP Value (Impact on Output)", fontweight='bold', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Fig16_SHAP_Advanced_Beeswarm.png', bbox_inches='tight', facecolor='white')
    plt.close()

    print("✅ Fig 16 SHAP 高级蜂群图生成成功！(已修复变量选择问题)")

except Exception as e:
    print(f"⚠️ SHAP 图表生成失败，错误信息: {e}")

print("🎉 所有图表已成功更新！")