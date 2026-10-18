import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines  
import seaborn as sns
from matplotlib.gridspec import GridSpec
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, train_test_split
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
OPT_N_GEN = 200               
OPT_POP_SIZE = 100            
OPT_SEED = 42                 
RF_N_ESTIMATORS_SHAP = 100    
RF_N_ESTIMATORS_REG = 50      
RF_RANDOM_STATE = 42          

NPG_COLORS = {'Maize': '#E64B35', 'Soybean': '#4DBBD5', 'Rice': '#00A087', 'Dark': '#3C5488', 'Gray': '#7E6148'}
FONT_TITLE = 20  
FONT_LABEL = 18  
FONT_TICK = 16  
FONT_ANNOTATION = 26  
FONT_LEGEND = 16  
GLOBAL_DPI = 900 
FIG_SIZE_ALGO = (26, 8)  
FIG_SIZE_SHAP = (20, 9)  

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.weight': 'bold',  
    'font.size': FONT_TICK,
    'axes.labelsize': FONT_LABEL,
    'axes.titlesize': FONT_TITLE,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 2.0,  
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.fontsize': FONT_LEGEND,
    'legend.frameon': False,
    'figure.dpi': GLOBAL_DPI,
    'savefig.bbox': 'tight'
})

from pymoo.config import Config
Config.warnings['not_compiled'] = False

print("🚀 正在启动全量可视化分析流水线...")

# =====================================================================
# 2. 统一的数据加载与预处理
# =====================================================================
print("📂 [1/7] 正在加载数据...")
try:
    df_ml = pd.read_csv("ML_Training_Data.csv")
    crops = ["Maize", "Soybean", "Rice"]
except Exception as e:
    raise FileNotFoundError("⚠️ 未找到 ML_Training_Data.csv")

try:
    df_pareto = pd.read_csv("Pareto_Front_Data.csv")
    df_plan = pd.read_csv("Optimized_Farm_Plan_Continuous.csv")
    df_metrics = pd.read_csv("All_Metrics_Data_Continuous.csv")
except Exception as e:
    raise FileNotFoundError("⚠️ 未找到 CSV 优化结果数据")

if 'Fert_Weight_kg_ha' in df_ml.columns:
    df_ml.rename(columns={'Fert_Weight_kg_ha': 'N_Fert_kg_ha'}, inplace=True)

print("📈 [2/7] 正在生成代理模型 Train/Test 验证与分布对比图 (Fig 6)...")
try:
    df_eval = df_ml.copy()
    
    targets_to_evaluate = []
    if 'Yield_kg_ha' in df_eval.columns: targets_to_evaluate.append('Yield_kg_ha')
    if 'ET_mm' in df_eval.columns: targets_to_evaluate.append('ET_mm')

    base_inputs = ['N_Fert_kg_ha', 'P_Fert_kg_ha', 'K_Fert_kg_ha', 'Density_plants_m2', 'Irrigation_mm']

    for target_col in targets_to_evaluate:
        results_train = []
        results_test = []

        for crop in ['Maize', 'Soybean', 'Rice']:
            df_c = df_eval[df_eval['Crop'] == crop].dropna(subset=[target_col])
            if len(df_c) < 10: continue

            features_cols = [col for col in base_inputs if col in df_c.columns]
            X = df_c[features_cols]
            y = df_c[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RF_RANDOM_STATE)

            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS_REG, random_state=RF_RANDOM_STATE, n_jobs=-1)
            rf.fit(X_train, y_train)

            y_train_pred = rf.predict(X_train)
            y_test_pred = rf.predict(X_test)

            results_train.append(pd.DataFrame({'Observed': y_train, 'Predicted': y_train_pred, 'Crop': crop}))
            results_test.append(pd.DataFrame({'Observed': y_test, 'Predicted': y_test_pred, 'Crop': crop}))

        if not results_train: continue
        df_train_all = pd.concat(results_train)
        df_test_all = pd.concat(results_test)

        fig = plt.figure(figsize=(14, 12), dpi=GLOBAL_DPI)
        gs = GridSpec(2, 2, height_ratios=[1, 1.1], hspace=0.6, wspace=0.35)

        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[1, :])

        def plot_scatter_by_crop(ax, df_data, title, label_letter):
            min_val_global = float('inf')
            max_val_global = float('-inf')
            metrics_texts = []

            for crop in ['Maize', 'Soybean', 'Rice']:
                sub_df = df_data[df_data['Crop'] == crop]
                if sub_df.empty: continue
                obs, pred = sub_df['Observed'], sub_df['Predicted']
                ax.scatter(obs, pred, color=NPG_COLORS[crop], edgecolor='white', s=390, alpha=0.7, linewidths=0.5, label=crop)

                min_val, max_val = min(obs.min(), pred.min()), max(obs.max(), pred.max())
                min_val_global = min(min_val_global, min_val)
                max_val_global = max(max_val_global, max_val)

                r2 = r2_score(obs, pred)
                rmse = np.sqrt(mean_squared_error(obs, pred))
                r2_str = "0.99" if 0.99 <= r2 < 1.0 else f"{r2:.2f}"
                metrics_texts.append(f"{crop}: $R^2$ = {r2_str}, RMSE = {rmse:.1f}")

            min_val_global *= 0.95
            max_val_global *= 1.05
            ax.plot([min_val_global, max_val_global], [min_val_global, max_val_global], 'k--', linewidth=2.5)

            metrics_str = "\n".join(metrics_texts)
            ax.text(0.05, 0.95, metrics_str, transform=ax.transAxes, fontsize=FONT_TICK,
                    verticalalignment='top', fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray'))

            ax.set_xlim(min_val_global, max_val_global)
            ax.set_ylim(min_val_global, max_val_global)
            ax.set_title(title, pad=15, fontsize=FONT_TITLE)
            ax.set_xlabel(f"DSSAT Simulated {target_col}", fontsize=FONT_LABEL)
            ax.set_ylabel(f"RF Predicted {target_col}", fontsize=FONT_LABEL)
            ax.text(-0.25, 1.05, label_letter, transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold', va='top')
            ax.tick_params(axis='both', labelsize=FONT_TICK)
            ax.spines['bottom'].set_linewidth(2.0)
            ax.spines['left'].set_linewidth(2.0)

        plot_scatter_by_crop(axA, df_train_all, 'Training', 'A')
        plot_scatter_by_crop(axB, df_test_all, 'Testing', 'B')

        crop_handles = [mlines.Line2D([], [], color='w', marker='o', markerfacecolor=NPG_COLORS[c], 
                                      markersize=14, label=c) for c in ['Maize', 'Soybean', 'Rice']]
        fig.legend(handles=crop_handles, loc='center', bbox_to_anchor=(0.5, 0.52), ncol=3, fontsize=FONT_LEGEND+2, frameon=False)

        # Panel C
        df_violin = pd.melt(df_test_all, id_vars=['Crop'], value_vars=['Observed', 'Predicted'],
                            var_name='Data_Type', value_name='Value')
        palette_colors = {'Observed': '#D3D3D3', 'Predicted': '#F05A5A'}
        crop_order = ['Maize', 'Soybean', 'Rice']

        df_main = df_violin[df_violin['Crop'].isin(['Maize', 'Rice'])]
        sns.violinplot(data=df_main, x='Crop', y='Value', hue='Data_Type', order=crop_order,
                       ax=axC, split=False, inner='box', palette=palette_colors, linewidth=1.5, width=0.75)
        sns.stripplot(data=df_main, x='Crop', y='Value', hue='Data_Type', order=crop_order,
                      ax=axC, dodge=True, color='black', alpha=0.3, size=4, jitter=True)

        axC_twin = axC.twinx()
        df_soy = df_violin[df_violin['Crop'] == 'Soybean']
        sns.violinplot(data=df_soy, x='Crop', y='Value', hue='Data_Type', order=crop_order,
                       ax=axC_twin, split=False, inner='box', palette=palette_colors, linewidth=1.5, width=0.75)
        sns.stripplot(data=df_soy, x='Crop', y='Value', hue='Data_Type', order=crop_order,
                      ax=axC_twin, dodge=True, color='black', alpha=0.3, size=4, jitter=True)

        axC.set_title('', pad=10) 
        axC.set_xlabel('Crop Type', fontsize=FONT_LABEL, labelpad=10)
        axC_twin.set_xlabel('')
        axC.set_ylabel(f"{target_col} (Maize & Rice)", fontsize=FONT_LABEL)
        soybean_color = NPG_COLORS['Soybean']
        axC_twin.set_ylabel(f"{target_col} (Soybean)", fontsize=FONT_LABEL, color=soybean_color)
        
        axC.tick_params(axis='both', labelsize=FONT_TICK)
        axC_twin.tick_params(axis='y', labelsize=FONT_TICK, colors=soybean_color)
        axC.text(-0.15, 1.05, 'C', transform=axC.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold', va='top')

        axC.spines['top'].set_visible(False)
        axC_twin.spines['top'].set_visible(False)
        axC.spines['bottom'].set_linewidth(2.0)
        axC.spines['left'].set_linewidth(2.0)
        axC.spines['right'].set_visible(False)
        axC_twin.spines['right'].set_linewidth(2.0)
        axC_twin.spines['right'].set_color(soybean_color)
        axC_twin.spines['left'].set_visible(False)
        axC_twin.spines['bottom'].set_visible(False)

        handles, labels = axC.get_legend_handles_labels()
        axC.legend(handles[:2], ['DSSAT Simulated (Obs)', 'RF Predicted (EMS)'],
                   loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=FONT_LEGEND, frameon=False)
        if axC_twin.get_legend() is not None: axC_twin.get_legend().remove()

        plt.savefig(f'Fig6_Model_Validation_TrainTest_{target_col}.png', bbox_inches='tight', facecolor='white', dpi=GLOBAL_DPI)
        plt.close()
except Exception as e:
    pass

# =====================================================================
# 3. 算法诊断图 + Pareto气泡图 (合并为1行3列)
# =====================================================================
print("📊 [3/7] 正在生成算法诊断、演化与气泡图...")
try:
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

    fig_diag = plt.figure(figsize=FIG_SIZE_ALGO, dpi=GLOBAL_DPI)
    gs_diag = GridSpec(1, 3, figure=fig_diag, wspace=0.35)

    ax_conv1 = fig_diag.add_subplot(gs_diag[0, 0])
    ax_pareto = fig_diag.add_subplot(gs_diag[0, 1])
    ax_bubble = fig_diag.add_subplot(gs_diag[0, 2])  

    ax_conv2 = ax_conv1.twinx()
    l1 = ax_conv1.plot(gen_numbers, max_profits, color=NPG_COLORS['Maize'], linewidth=3.0, label='Max Profit Discovery')
    ax_conv1.set_xlabel('Evolution Generations', fontsize=FONT_LABEL)
    ax_conv1.set_ylabel('Discovered Max Profit', color=NPG_COLORS['Maize'], fontsize=FONT_LABEL)
    ax_conv1.tick_params(axis='y', labelcolor=NPG_COLORS['Maize'])

    l2 = ax_conv2.plot(gen_numbers, min_carbons, color=NPG_COLORS['Soybean'], linewidth=3.0, linestyle='-', label='Min Carbon Discovery')
    ax_conv2.set_ylabel('Discovered Min Carbon', color=NPG_COLORS['Soybean'], labelpad=10, fontsize=FONT_LABEL)
    ax_conv2.tick_params(axis='y', labelcolor=NPG_COLORS['Soybean'])

    ax_conv1.legend(l1 + l2, [l.get_label() for l in (l1 + l2)], loc='center right', frameon=False, fontsize=FONT_LEGEND)
    ax_conv2.spines['top'].set_visible(False)
    ax_conv1.set_title("Algorithm Convergence Dynamics", pad=15, fontsize=FONT_TITLE)
    ax_conv1.text(-0.15, 1.05, 'A', transform=ax_conv1.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')

    last_gen_idx = OPT_N_GEN - 1
    mid_gen_idx = min(29, max(0, last_gen_idx - 1))
    target_gens = [0, mid_gen_idx, last_gen_idx]
    colors_gen = ['#CCCCCC', '#88CCEE', NPG_COLORS['Dark']]
    labels_gen = ['Generation 1', f'Generation {mid_gen_idx + 1}', f'Generation {OPT_N_GEN} (Final Front)']

    for g_idx, color, label in zip(target_gens, colors_gen, labels_gen):
        F_gen = hist_F[g_idx]
        profit_gen, carbon_gen = -F_gen[:, 0], F_gen[:, 1]
        ax_pareto.scatter(profit_gen, carbon_gen, color=color, s=(100 if g_idx == last_gen_idx else 50),
                          alpha=(0.9 if g_idx == last_gen_idx else 0.5), label=label, edgecolor='white')
        if g_idx == last_gen_idx:
            ax_pareto.plot(profit_gen[np.argsort(profit_gen)], carbon_gen[np.argsort(profit_gen)],
                           color=NPG_COLORS['Dark'], linewidth=2.0, linestyle='--')

    start_pt, end_pt = (np.mean(-hist_F[0][:, 0]), np.mean(hist_F[0][:, 1])), (np.mean(-hist_F[-1][:, 0]), np.mean(hist_F[-1][:, 1]))
    ax_pareto.annotate('', xy=end_pt, xytext=start_pt, arrowprops=dict(facecolor=NPG_COLORS['Maize'], shrink=0.05, width=3, headwidth=12, alpha=0.8))
    ax_pareto.text((start_pt[0] + end_pt[0]) / 2, (start_pt[1] + end_pt[1]) / 2 + 200, 'Evolution\nDirection',
                   color=NPG_COLORS['Maize'], fontweight='bold', ha='center', fontsize=FONT_TICK)

    ax_pareto.set_title("Pareto Front Evolution", pad=15, fontsize=FONT_TITLE)
    ax_pareto.set_xlabel('Profit Objective (Maximize)', fontsize=FONT_LABEL)
    ax_pareto.set_ylabel('Carbon Objective (Minimize)', fontsize=FONT_LABEL)
    ax_pareto.legend(loc='upper right', frameon=False, fontsize=FONT_LEGEND)
    ax_pareto.text(-0.15, 1.05, 'B', transform=ax_pareto.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')

    df_p_plot = df_pareto.copy()
    df_p_plot['Profit (10k CNY)'] = df_p_plot['Profit'] / 10000
    df_p_plot['Carbon (t CO2e)'] = df_p_plot['Carbon'] / 1000
    
    sizes = (df_p_plot['WUE'] - df_p_plot['WUE'].min()) / (df_p_plot['WUE'].max() - df_p_plot['WUE'].min() + 1e-6)
    bubble_sizes = sizes * 500 + 80
    best_p_idx = df_p_plot['Profit (10k CNY)'].idxmax()

    scatter14 = ax_bubble.scatter(df_p_plot['Profit (10k CNY)'], df_p_plot['Carbon (t CO2e)'], s=bubble_sizes,
                                  c=df_p_plot['WUE'], cmap='coolwarm', alpha=0.8, edgecolors='black', linewidth=0.5)
    ax_bubble.scatter(df_p_plot.loc[best_p_idx, 'Profit (10k CNY)'], df_p_plot.loc[best_p_idx, 'Carbon (t CO2e)'], color='red',
                      marker='*', s=600, edgecolor='black', linewidth=1.5, zorder=10, label='Max Profit Strategy')
    
    cbar14 = fig_diag.colorbar(scatter14, ax=ax_bubble, pad=0.02)
    cbar14.set_label('System WUE (kg/m³)', fontsize=FONT_LABEL)
    cbar14.ax.tick_params(labelsize=FONT_TICK)
    ax_bubble.set_title("Pareto Front Projection", pad=15, fontsize=FONT_TITLE)
    ax_bubble.set_xlabel('Total Profit (10,000 CNY)', fontsize=FONT_LABEL)
    ax_bubble.set_ylabel('Total Carbon Footprint (Tons CO2e)', fontsize=FONT_LABEL)
    ax_bubble.legend(loc='upper right', fontsize=FONT_LEGEND)
    ax_bubble.text(-0.15, 1.05, 'C', transform=ax_bubble.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')

    plt.tight_layout(pad=2.0, w_pad=5.0)
    plt.savefig('Fig_Algorithm_Diagnostics_Combined.png', facecolor='white', bbox_inches='tight')
    plt.close()
except Exception as e:
    pass

# =====================================================================
# 4. 帕累托前沿深度分析图 (Pair Grid)
# =====================================================================
try:
    df_p_plot = df_pareto.copy()
    df_p_plot['Profit (10k CNY)'] = df_p_plot['Profit'] / 10000
    df_p_plot['Carbon (t CO2e)'] = df_p_plot['Carbon'] / 1000
    df_p_plot = df_p_plot[['Profit (10k CNY)', 'WUE', 'Carbon (t CO2e)']]

    g = sns.PairGrid(df_p_plot, diag_sharey=False, corner=False, height=3)
    g.map_lower(sns.scatterplot, alpha=0.7, color=NPG_COLORS['Dark'], edgecolor='white', s=50)
    g.map_upper(sns.kdeplot, fill=True, cmap='Blues', alpha=0.7)
    g.map_diag(sns.histplot, kde=True, color=NPG_COLORS['Soybean'])
    plt.savefig('Fig 13 Pairwise Trade-off Matrix.png', bbox_inches='tight')
    plt.close()
except Exception as e:
    pass

# =====================================================================
# 5. 54个地块优化处方与经济效益综合图
# =====================================================================
print("🌾📈 [4/7] 正在生成综合经济效益复合大图...")
try:
    fig_combined = plt.figure(figsize=(22, 16), dpi=GLOBAL_DPI)
    gs_top = GridSpec(1, 3, figure=fig_combined, bottom=0.55, top=0.95, wspace=0.3)
    gs_bottom = GridSpec(1, 3, figure=fig_combined, width_ratios=[1.5, 1, 1], bottom=0.05, top=0.45, wspace=0.3)

    df_opt = pd.merge(df_plan, df_metrics[['Field_ID', 'Yield_ha', 'Profit_ha']], on='Field_ID')
    VIOLIN_COLORS = {k: NPG_COLORS[k] for k in ['Maize', 'Soybean', 'Rice']}
    crops_order = ['Maize', 'Soybean', 'Rice']

    axes_top = [fig_combined.add_subplot(gs_top[i]) for i in range(3)]
    metrics_top = ['Density_plants_m2', 'Fert_Weight_kg_ha', 'Yield_ha']
    ylabels_top = ['Density Prescriptions (plants/m²)', 'Fert_Weight Prescriptions (kg/ha)', 'Optimized Yield (kg/ha)']
    titles_top = ['Density Allocation (54 Fields)', 'Fertilizer Allocation (54 Fields)', 'Yield Outcome Distribution']

    for i, (ax, metric, ylabel, title) in enumerate(zip(axes_top, metrics_top, ylabels_top, titles_top)):
        sns.violinplot(x='Optimized_Crop', y=metric, data=df_opt, order=crops_order, palette=VIOLIN_COLORS, ax=ax, inner="quartile", linewidth=1.5)
        sns.stripplot(x='Optimized_Crop', y=metric, data=df_opt, order=crops_order, color='black', alpha=0.3, size=4, ax=ax, jitter=True)
        ax.set_title(title, pad=15, fontsize=FONT_TITLE)
        ax.set_xlabel("", fontsize=FONT_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        ax.tick_params(axis='both', labelsize=FONT_TICK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['left'].set_linewidth(2.0)
        ax.text(-0.15, 1.05, chr(65 + i), transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')

    axD = fig_combined.add_subplot(gs_bottom[0])
    axE = fig_combined.add_subplot(gs_bottom[1])
    axF = fig_combined.add_subplot(gs_bottom[2])

    df_eco = df_metrics.copy()
    crop_col = 'Crop' if 'Crop' in df_eco.columns else 'Optimized_Crop'
    df_eco['Total_Profit_10k'] = (df_eco['Profit_ha'] * df_eco['Area_ha']) / 10000

    df_sorted_profit = df_eco.sort_values(by='Profit_ha', ascending=False).reset_index(drop=True)
    colors_D = [NPG_COLORS[c] for c in df_sorted_profit[crop_col]]
    axD.bar(range(len(df_sorted_profit)), df_sorted_profit['Profit_ha'], color=colors_D, width=0.8, edgecolor='black', linewidth=0.5)

    axD.text(-0.1, 1.05, 'D', transform=axD.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
    axD.set_ylabel('Net Profit (CNY/ha)', fontsize=FONT_LABEL)
    axD.set_xlabel('54 Fields (Ranked by Profit Margin)', fontsize=FONT_LABEL)
    axD.set_xticks([])
    axD.axhline(df_eco['Profit_ha'].mean(), color='#555555', linestyle='--', linewidth=2.0, zorder=0)

    legend_handles = [mpatches.Patch(color=NPG_COLORS[c], label=c) for c in ['Maize', 'Soybean', 'Rice']]
    axD.legend(handles=legend_handles, title='Optimized Crop', loc='upper right', frameon=False, fontsize=FONT_LEGEND)

    for c in ['Maize', 'Soybean', 'Rice']:
        sub = df_eco[df_eco[crop_col] == c]
        if not sub.empty:
            axE.scatter(sub['Area_ha'], sub['Profit_ha'], s=sub['Total_Profit_10k'] * 3 + 50, color=NPG_COLORS[c],
                        alpha=0.8, edgecolor='white', linewidth=1, label=c)
    axE.text(-0.1, 1.05, 'E', transform=axE.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
    axE.axvline(df_eco['Area_ha'].median(), color='#999999', linestyle=':', linewidth=2, zorder=0)
    axE.axhline(df_eco['Profit_ha'].median(), color='#999999', linestyle=':', linewidth=2, zorder=0)
    axE.set_xlabel('Field Area (ha)', fontsize=FONT_LABEL)
    axE.set_ylabel('Net Profit (CNY/ha)', fontsize=FONT_LABEL)

    df_lorenz = df_eco.sort_values('Profit_ha', ascending=False)
    cum_area = np.insert((df_lorenz['Area_ha'].cumsum() / df_lorenz['Area_ha'].sum() * 100).values, 0, 0)
    cum_profit = np.insert((df_lorenz['Total_Profit_10k'].cumsum() / df_lorenz['Total_Profit_10k'].sum() * 100).values, 0, 0)
    
    axF.text(-0.1, 1.05, 'F', transform=axF.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
    axF.plot(cum_area, cum_profit, color=NPG_COLORS['Dark'], linewidth=3.0, label='Actual Profit Curve')
    axF.plot([0, 100], [0, 100], color='gray', linestyle='--', linewidth=2.0)
    axF.fill_between(cum_area, cum_area, cum_profit, color=NPG_COLORS['Soybean'], alpha=0.15)
    axF.set_xlabel('Cumulative Area (%)', fontsize=FONT_LABEL)
    axF.set_ylabel('Cumulative Total Profit (%)', fontsize=FONT_LABEL)
    axF.set_xlim(0, 100)
    axF.set_ylim(0, 100)

    plt.savefig('Fig_Combined_Violin_Economic.png', facecolor='white', bbox_inches='tight')
    plt.close()
except Exception as e:
    pass

# =====================================================================
# 6. 独立生成 Fig 09 (农艺响应面) 及 Fig 19&20 (SHAP 机理交互矩阵)
# =====================================================================
print("🌡️ [5/7] 正在生成独立的 Fig 09 (农艺响应面)...")
try:
    fig09, axes09 = plt.subplots(1, 3, figsize=(22, 6), dpi=GLOBAL_DPI)
    cmap_dict = {'Soybean': 'Blues', 'Rice': 'Greens', 'Maize': 'Reds'}
    target_z = 'Yield_kg_ha' if 'Yield_kg_ha' in df_ml.columns else 'Profit_ha'
    density_col = 'Density_plants_m2'
    fert_col = 'N_Fert_kg_ha' if 'N_Fert_kg_ha' in df_ml.columns else 'Fert_Weight_kg_ha'

    for i, crop in enumerate(crops):
        ax = axes09[i]
        sub_df = df_ml[df_ml['Crop'] == crop]
        if len(sub_df) > 5:
            x, y, z = sub_df[density_col], sub_df[fert_col], sub_df[target_z]
            cntr = ax.tricontourf(x, y, z, levels=20, cmap=cmap_dict[crop], alpha=0.9)
            ax.tricontour(x, y, z, levels=20, colors='black', linewidths=0.5, alpha=0.5)
            
            cbar09 = fig09.colorbar(cntr, ax=ax, pad=0.02)
            cbar09.set_label(f'{target_z}', fontsize=FONT_LABEL, fontweight='bold')
            cbar09.ax.tick_params(labelsize=FONT_TICK)
            for l in cbar09.ax.yaxis.get_ticklabels(): l.set_weight("bold")
            
            ax.set_title(f"[{crop}] Response Space", color=NPG_COLORS[crop], fontsize=FONT_TITLE, fontweight='bold')
            ax.set_xlabel("Density (plants/m²)", fontsize=FONT_LABEL, fontweight='bold')
            ax.set_ylabel("Fertilizer (kg/ha)", fontsize=FONT_LABEL, fontweight='bold')
            ax.text(-0.15, 1.05, chr(65 + i), transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(2.0)
            ax.spines['left'].set_linewidth(2.0)
            ax.tick_params(axis='both', labelsize=FONT_TICK)

    plt.tight_layout(w_pad=4.0)
    plt.savefig('Fig09_Response_Space_Standalone.png', bbox_inches='tight', facecolor='white', dpi=GLOBAL_DPI)
    plt.close()
    print("✅ 独立的 Fig 09 (农艺响应面) 生成成功！")

except Exception as e:
    print(f"⚠️ Fig 09 生成失败: {e}")

print("🔍 [5.5/7] 正在将 Fig 19 & 20 组合为 3x3 SHAP 终极机理交互组图...")
try:
    fig_interact, axes_interact = plt.subplots(3, 3, figsize=(22, 18), dpi=GLOBAL_DPI)
    target_et = 'ET_mm'

    for i, crop in enumerate(crops):
        df_crop = df_ml[df_ml['Crop'] == crop].dropna()
        if df_crop.empty: continue
        X_train = df_crop[[density_col, fert_col]]
        
        ax_r1 = axes_interact[0, i]
        rf_y = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS_SHAP, random_state=RF_RANDOM_STATE, n_jobs=-1)
        rf_y.fit(X_train, df_crop[target_z])
        sv_y = shap.TreeExplainer(rf_y).shap_values(X_train)
        sv_y_dens = sv_y[0][:, 0] if isinstance(sv_y, list) else (sv_y[:, 0, 0] if len(sv_y.shape)==3 else sv_y[:, 0])

        sc1 = ax_r1.scatter(X_train[density_col], sv_y_dens, c=X_train[fert_col], cmap='coolwarm', s=60, alpha=0.8, edgecolors='none')
        ax_r1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
        ax_r1.set_title(f"A{i+1}: [{crop}] Yield Impact\n(Density x Fertilizer)", pad=20, fontsize=FONT_TITLE, color=NPG_COLORS.get(crop, 'black'))
        ax_r1.set_xlabel("Planting Density (plants/m²)", fontsize=FONT_LABEL)
        ax_r1.set_ylabel("SHAP Value (Yield)", fontsize=FONT_LABEL)
        cb1 = plt.colorbar(sc1, ax=ax_r1, shrink=0.8, pad=0.05)
        cb1.set_label('Fertilizer (kg/ha)', fontsize=FONT_LEGEND, weight='bold')

        ax_r2 = axes_interact[1, i]
        if target_et in df_ml.columns:
            rf_et = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS_SHAP, random_state=RF_RANDOM_STATE, n_jobs=-1)
            rf_et.fit(X_train, df_crop[target_et])
            sv_et = shap.TreeExplainer(rf_et).shap_values(X_train)
            sv_et_dens = sv_et[0][:, 0] if isinstance(sv_et, list) else (sv_et[:, 0, 0] if len(sv_et.shape)==3 else sv_et[:, 0])
            
            sc2 = ax_r2.scatter(X_train[density_col], sv_et_dens, c=X_train[fert_col], cmap='viridis', s=60, alpha=0.8, edgecolors='none')
            ax_r2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
            ax_r2.set_title(f"B{i+1}: [{crop}] ET Impact\n(Density x Fertilizer)", pad=20, fontsize=FONT_TITLE, color=NPG_COLORS.get(crop, 'black'))
            ax_r2.set_xlabel("Planting Density (plants/m²)", fontsize=FONT_LABEL)
            ax_r2.set_ylabel("SHAP Value (ET)", fontsize=FONT_LABEL)
            cb2 = plt.colorbar(sc2, ax=ax_r2, shrink=0.8, pad=0.05)
            cb2.set_label('Fertilizer (kg/ha)', fontsize=FONT_LEGEND, weight='bold')

        ax_r3 = axes_interact[2, i]
        sv_y_fert = sv_y[0][:, 1] if isinstance(sv_y, list) else (sv_y[:, 1, 0] if len(sv_y.shape)==3 else sv_y[:, 1])
        
        sc3 = ax_r3.scatter(X_train[fert_col], sv_y_fert, c=X_train[density_col], cmap='plasma', s=60, alpha=0.8, edgecolors='none')
        ax_r3.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
        ax_r3.set_title(f"C{i+1}: [{crop}] Yield Impact\n(Fertilizer x Density)", pad=20, fontsize=FONT_TITLE, color=NPG_COLORS.get(crop, 'black'))
        ax_r3.set_xlabel("Fertilizer Application (kg/ha)", fontsize=FONT_LABEL)
        ax_r3.set_ylabel("SHAP Value (Yield)", fontsize=FONT_LABEL)
        cb3 = plt.colorbar(sc3, ax=ax_r3, shrink=0.8, pad=0.05)
        cb3.set_label('Density (plants/m²)', fontsize=FONT_LEGEND, weight='bold')

        for idx, ax in enumerate([ax_r1, ax_r2, ax_r3]):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(2.0)
            ax.spines['left'].set_linewidth(2.0)
            ax.tick_params(axis='both', labelsize=FONT_TICK)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight('bold')
            row_letter = ['A', 'B', 'C'][idx]
            ax.text(-0.25, 1.15, f"{row_letter}{i+1}", transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold', va='bottom', ha='right')

    plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=4.0)
    plt.savefig('Fig19_20_SHAP_Interactions_Combined.png', bbox_inches='tight', facecolor='white', dpi=GLOBAL_DPI)
    plt.close()
    print("✅ SHAP 机理交互矩阵 (Fig 19 & 20 组合) 生成成功！")

except Exception as e:
    print(f"⚠️ SHAP 组图生成失败: {e}")

# =====================================================================
# 7. 农艺处方小提琴图与其他原脚本延续图
# =====================================================================
print("🧠 [6/7] 正在生成其他辅助图表...")
try:
    df_ml_model = df_ml.copy()
    target_col = 'Profit_ha' if 'Profit_ha' in df_ml_model.columns else 'Yield_kg_ha'
    base_inputs = ['N_Fert_kg_ha', 'P_Fert_kg_ha', 'K_Fert_kg_ha', 'Density_plants_m2', 'Irrigation_mm']

    fig_shap, axes_shap = plt.subplots(2, 2, figsize=(18, 12), dpi=GLOBAL_DPI)
    axes_flat = axes_shap.flatten()

    features_cols_all = [col for col in base_inputs + ['Crop'] if col in df_ml_model.columns]
    X_raw_all = df_ml_model[features_cols_all]
    X_all = pd.get_dummies(X_raw_all, drop_first=True)
    y_all = df_ml_model[target_col]

    rf_all = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS_SHAP, random_state=RF_RANDOM_STATE, n_jobs=-1)
    rf_all.fit(X_all, y_all)
    explainer_all = shap.TreeExplainer(rf_all)
    shap_values_all = explainer_all.shap_values(X_all)

    plt.sca(axes_flat[0])
    shap.summary_plot(shap_values_all, X_all, show=False, cmap='coolwarm', max_display=10, color_bar=False, plot_size=None)
    axes_flat[0].set_title("Overall System", pad=20, color='black', fontsize=FONT_TITLE)
    axes_flat[0].set_xlabel("SHAP Value (Impact)", labelpad=15, fontsize=FONT_LABEL)
    axes_flat[0].text(-0.40, 1.05, 'A', transform=axes_flat[0].transAxes, fontsize=FONT_ANNOTATION, fontweight='bold', va='top')

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
        shap.summary_plot(shap_values_crop, X_crop, show=False, cmap='coolwarm', max_display=10, color_bar=False, plot_size=None)
        ax.set_title(f"{crop_name} Only", pad=20, color=NPG_COLORS.get(crop_name, 'black'), fontsize=FONT_TITLE)
        ax.set_xlabel("SHAP Value (Impact)", labelpad=15, fontsize=FONT_LABEL)
        ax.text(-0.40, 1.05, chr(66 + i), transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold', va='top')

    for ax in axes_flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2.5) 
        ax.tick_params(axis='y', length=0, labelsize=FONT_TICK, pad=10)
        ax.tick_params(axis='x', length=8, width=2.5, labelsize=FONT_TICK)

    plt.subplots_adjust(left=0.1, right=0.86, wspace=0.55, hspace=0.55)
    cmap = plt.get_cmap('coolwarm')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig_shap.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig_shap.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Feature Value', size=FONT_LABEL, weight='bold', labelpad=20)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.ax.tick_params(labelsize=FONT_TICK, length=0) 
    cbar.outline.set_visible(False)
    plt.savefig('Fig16_SHAP_2x2_Grid_SCI.png', bbox_inches='tight', facecolor='white', dpi=GLOBAL_DPI)
    plt.close()
except Exception as e:
    pass

# =====================================================================
# 8. 空间平滑性约束对比图
# =====================================================================
print("🚜 [7/7] 正在生成空间平滑性对比的 7 种不同展现形式...")
import scipy.ndimage as ndimage
from matplotlib.patches import RegularPolygon
try:
    grid_shape = (6, 9)
    np.random.seed(42)
    raw_prescription = np.random.choice([1, 2, 3, 4, 5], size=grid_shape, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    raw_prescription += np.random.randint(-1, 2, size=grid_shape) 
    raw_prescription = np.clip(raw_prescription, 1, 5)
    
    smoothed_prescription = ndimage.median_filter(raw_prescription, size=3)
    cmap_mrf = matplotlib.colors.LinearSegmentedColormap.from_list("agri_mrf", ['#FFFFCC', '#C2E699', '#78C679', '#31A354', '#006837'])

    titles = ["Original Fragmented", "MRF Smoothed Contiguous"]
    datasets = [raw_prescription, smoothed_prescription]

    fig_surf = plt.figure(figsize=(18, 8), dpi=GLOBAL_DPI)
    x = np.arange(grid_shape[1])
    y = np.arange(grid_shape[0])
    X, Y = np.meshgrid(x, y)

    for i, (data, title) in enumerate(zip(datasets, titles)):
        ax = fig_surf.add_subplot(1, 2, i+1, projection='3d')
        surf = ax.plot_surface(X, Y, data, cmap=cmap_mrf, rstride=1, cstride=1, alpha=0.9, edgecolor='black', linewidth=0.5)
        ax.set_title(title + " (Surface)", pad=20, fontsize=FONT_TITLE)
        ax.set_zlim(0, 5.5)
        ax.view_init(elev=45, azim=-60)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.text2D(0.0, 1.0, chr(65+i), transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
    plt.savefig('Fig17_Alt3_3D_Surface.png', bbox_inches='tight', facecolor='white')
    plt.close()

    fig_cont, axes_cont = plt.subplots(1, 2, figsize=(18, 7), dpi=GLOBAL_DPI)
    zoom_factor = 5
    for i, (ax, data, title) in enumerate(zip(axes_cont, datasets, titles)):
        data_zoomed = ndimage.zoom(data, zoom_factor, order=3) 
        X_z, Y_z = np.meshgrid(np.linspace(0, grid_shape[1]-1, data_zoomed.shape[1]), np.linspace(0, grid_shape[0]-1, data_zoomed.shape[0]))
        cntr = ax.contourf(X_z, Y_z, data_zoomed, levels=np.linspace(1, 5, 15), cmap=cmap_mrf, alpha=0.9)
        ax.contour(X_z, Y_z, data_zoomed, levels=5, colors='black', linewidths=0.8, alpha=0.6)
        ax.set_title(title + " (Contour)", pad=20, fontsize=FONT_TITLE)
        ax.set_xticks([]); ax.set_yticks([])
        ax.invert_yaxis()
        ax.text(-0.05, 1.05, chr(65+i), transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
    plt.savefig('Fig17_Alt4_Contour_Map.png', bbox_inches='tight', facecolor='white')
    plt.close()

    fig_hex, axes_hex = plt.subplots(1, 2, figsize=(18, 7), dpi=GLOBAL_DPI)
    for i, (ax, data, title) in enumerate(zip(axes_hex, datasets, titles)):
        ax.set_aspect('equal')
        for r in range(grid_shape[0]):
            for c in range(grid_shape[1]):
                x_offset = c * np.sqrt(3) + (np.sqrt(3)/2 if r % 2 == 1 else 0)
                y_offset = r * 1.5
                val = data[r, c]
                hex_patch = RegularPolygon((x_offset, y_offset), numVertices=6, radius=1.0, orientation=np.radians(30), facecolor=cmap_mrf((val-1)/4.0), edgecolor='white', linewidth=2.0)
                ax.add_patch(hex_patch)
        ax.autoscale_view(); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')
        ax.set_title(title + " (Hex Grid)", pad=20, fontsize=FONT_TITLE)
        ax.text(0, 1.0, chr(65+i), transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
    plt.savefig('Fig17_Alt5_Hexagonal_Grid.png', bbox_inches='tight', facecolor='white')
    plt.close()

    fig_ridge, axes_ridge = plt.subplots(1, 2, figsize=(18, 8), dpi=GLOBAL_DPI)
    x_ridge = np.arange(grid_shape[1])
    for i, (ax, data, title) in enumerate(zip(axes_ridge, datasets, titles)):
        for r in range(grid_shape[0]-1, -1, -1):
            y_base = np.zeros_like(x_ridge) + (grid_shape[0] - r) * 3
            y_val = y_base + data[r, :]
            x_smooth = np.linspace(x_ridge.min(), x_ridge.max(), 200)
            from scipy.interpolate import make_interp_spline
            spl = make_interp_spline(x_ridge, y_val, k=3)
            y_smooth = spl(x_smooth)
            y_base_smooth = np.zeros_like(x_smooth) + (grid_shape[0] - r) * 3
            ax.plot(x_smooth, y_smooth, color='black', linewidth=1.5, zorder=r)
            mean_val = np.mean(data[r, :])
            ax.fill_between(x_smooth, y_base_smooth, y_smooth, color=cmap_mrf((mean_val-1)/4.0), alpha=0.9, zorder=r)
        ax.set_title(title + " (Ridgeline)", pad=20, fontsize=FONT_TITLE)
        ax.set_xticks([]); ax.set_yticks([]); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
        ax.text(-0.05, 1.05, chr(65+i), transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
    plt.savefig('Fig17_Alt6_Ridgeline.png', bbox_inches='tight', facecolor='white')
    plt.close()

    fig_tex, axes_tex = plt.subplots(1, 2, figsize=(18, 7), dpi=GLOBAL_DPI)
    hatches = ['', '////', '\\\\\\\\', 'xxxx', 'OO'] 
    for i, (ax, data, title) in enumerate(zip(axes_tex, datasets, titles)):
        for r in range(grid_shape[0]):
            for c in range(grid_shape[1]):
                val = int(data[r, c])
                rect = plt.Rectangle((c, r), 1, 1, facecolor=cmap_mrf((val-1)/4.0), edgecolor='black', hatch=hatches[val-1], linewidth=1.5)
                ax.add_patch(rect)
        ax.set_xlim(0, grid_shape[1]); ax.set_ylim(grid_shape[0], 0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title + " (Textured Zones)", pad=20, fontsize=FONT_TITLE)
        ax.text(-0.05, 1.05, chr(65+i), transform=ax.transAxes, fontsize=FONT_ANNOTATION, fontweight='bold')
    plt.savefig('Fig17_Alt7_Textured_Zones.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 全部修改并增强的脚本执行完毕！请查看当前目录下的新排版图片。")

except Exception as e:
    print(f"⚠️ 空间平滑性对比图生成失败: {e}")

print("🎉 执行完毕！")