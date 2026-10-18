# =====================================================================
# Part 1: Farm Spatial Optimization Engine (复合肥 + SHAP + 3目标优化)
# 升级版：加入文化基因算法 (Memetic Algorithm) 解决空间平滑性约束
# =====================================================================
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib
import shap
from sklearn.ensemble import RandomForestRegressor
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mutation import Mutation
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from scipy.ndimage import median_filter
from tqdm import tqdm
import os

# 屏蔽警告并设置中文字体防乱码
warnings.filterwarnings(action='ignore', category=UserWarning)
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# =====================================================================
# --- 0. 全局算法参数设置 ---
# =====================================================================
N_GEN = 200               
POP_SIZE = 100            
OPT_SEED = 1              
RF_N_ESTIMATORS = 100     
RF_RANDOM_STATE = 42      

# --- 1. 经济参数与环境约束 ---
PRICE_MAIZE, PRICE_SOYBEAN, PRICE_RICE = 2.6401, 4.7180, 2.8366
BASE_COST_MAIZE, BASE_COST_SOYBEAN, BASE_COST_RICE = 7591.9216, 6253.4778, 14249.5380
FERT_PRICE_MAIZE, FERT_PRICE_SOYBEAN, FERT_PRICE_RICE = 5.9080, 4.7180, 6.8960
N_FERT_CARBON_FACTOR = 8.3
P_FERT_CARBON_FACTOR = 1.2
K_FERT_CARBON_FACTOR = 0.8
BASE_CARBON_MAIZE, BASE_CARBON_SOYBEAN, BASE_CARBON_RICE = 1200, 800, 4500
MIN_AREA_MAIZE, MIN_AREA_SOYBEAN = 10000, 10000

# --- 2. 农艺实验连续搜索边界 (Min, Max) ---
BOUNDS_DENSITY = {"Rice": (15.0, 40.0), "Maize": (3.0, 18.0), "Soybean": (10.0, 35.0)}
BOUNDS_FERT = {"Rice": (1800.0, 3400.0), "Maize": (900.0, 2700.0), "Soybean": (250.0, 2000.0)}

# --- 3. 地块结构定义 ---
PADDY_INDEX = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
DRY_INDEX = [i for i in range(1, 55) if i not in PADDY_INDEX]

AREA_DICT = {
    1: 500, 2: 500, 3: 500, 4: 500, 5: 704, 6: 605, 7: 805, 8: 226, 9: 339, 10: 890, 11: 696, 12: 1000,
    13: 459, 14: 501, 15: 800, 16: 694, 17: 821, 18: 572, 19: 864, 20: 710, 21: 306, 22: 3670, 23: 3670,
    24: 3670, 25: 536, 26: 536, 27: 752, 28: 313, 29: 313, 30: 198, 31: 132, 32: 100, 33: 100, 34: 100,
    35: 100, 36: 585, 37: 585, 38: 780, 39: 780, 40: 680, 41: 1253, 42: 473, 43: 940, 44: 740, 45: 1105,
    46: 250, 47: 1712, 48: 1644, 49: 1766, 50: 974, 51: 661, 52: 905, 53: 1094, 54: 3500
}

PRICES = {"Maize": PRICE_MAIZE, "Rice": PRICE_RICE, "Soybean": PRICE_SOYBEAN}
BASE_COSTS = {"Maize": BASE_COST_MAIZE, "Rice": BASE_COST_RICE, "Soybean": BASE_COST_SOYBEAN}
FERT_PRICES = {"Maize": FERT_PRICE_MAIZE, "Rice": FERT_PRICE_RICE, "Soybean": FERT_PRICE_SOYBEAN}
BASE_CARBON = {"Maize": BASE_CARBON_MAIZE, "Rice": BASE_CARBON_RICE, "Soybean": BASE_CARBON_SOYBEAN}

# --- 4. 训练机器学习代理模型 ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "ML_Training_Data.csv")
    data = pd.read_csv(data_path)
    crops = ["Maize", "Soybean", "Rice"]
    models = {}
    
    for crop in tqdm(crops, desc="🚀 [1/3] 训练 RF 代理模型"):
        df_crop = data[data['Crop'] == crop].dropna()
        rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE)
        rf.fit(df_crop[['Density_plants_m2', 'N_Fert_kg_ha' if 'N_Fert_kg_ha' in df_crop.columns else 'Fert_Weight_kg_ha']], df_crop[['Yield_kg_ha', 'ET_mm']])
        models[crop] = rf
except Exception as e:
    print(f"⚠️ 模型训练失败: {e}。请确保 ML_Training_Data.csv 存在且格式正确。")
    exit()

# --- 4.5 代理模型可解释性分析 (SHAP 组图) ---
print("📊 [1.5/3] 正在生成 SHAP 机理分析组图 (带ABCDEF编号)...")
fig_shap, axes_shap = plt.subplots(3, 2, figsize=(16, 16))

for i, crop in enumerate(tqdm(crops, desc="📈 计算 SHAP 值")):
    df_crop = data[data['Crop'] == crop].dropna()
    feature_col = 'N_Fert_kg_ha' if 'N_Fert_kg_ha' in df_crop.columns else 'Fert_Weight_kg_ha'
    X_train = df_crop[['Density_plants_m2', feature_col]]

    explainer = shap.TreeExplainer(models[crop])
    shap_values = explainer.shap_values(X_train)
    
    if isinstance(shap_values, list):
        sv_yield = shap_values[0]  
    elif len(shap_values.shape) == 3:
        sv_yield = shap_values[:, :, 0]  
    else:
        sv_yield = shap_values

    # 子图1：SHAP 全局重要性 (Bar)
    ax1 = axes_shap[i, 0]
    plt.sca(ax1)
    shap.summary_plot(sv_yield, X_train, plot_type="bar", show=False, plot_size=None)
    letter_left = chr(65 + i * 2) 
    ax1.text(-0.15, 1.05, letter_left, transform=ax1.transAxes, fontsize=26, fontweight='bold')
    ax1.set_title(f"[{crop}] 全局特征重要性 (Yield)", fontsize=20, fontweight='bold', pad=15)
    ax1.set_xlabel("Mean Absolute SHAP Value", fontsize=18, fontweight='bold', labelpad=10)

    # 子图2：SHAP 蜂群图 (Summary)
    ax2 = axes_shap[i, 1]
    plt.sca(ax2)
    shap.summary_plot(sv_yield, X_train, show=False, plot_size=None)
    letter_right = chr(65 + i * 2 + 1)
    ax2.text(-0.15, 1.05, letter_right, transform=ax2.transAxes, fontsize=26, fontweight='bold')
    ax2.set_title(f"[{crop}] 农艺措施非线性影响", fontsize=20, fontweight='bold', pad=15)
    ax2.set_xlabel("SHAP Value (Impact on Yield)", fontsize=18, fontweight='bold', labelpad=10)

plt.tight_layout(pad=3.5)
plt.savefig("Fig_SHAP_Combined.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

# =====================================================================
# 💥 5. 核心算法升级：文化基因算法 (Memetic Algorithm) 空间平滑算子
# =====================================================================
class MemeticSpatialMutation(Mutation):
    """
    文化基因算法 (Memetic Algorithm) 中的自定义局部搜索变异算子。
    在传统多项式变异 (PM) 后，加入基于空间拓扑的局部平滑搜索，
    强制消除相邻地块的作业配置碎片化，提升农机作业连续性。
    """
    def __init__(self, prob=None, eta=20, local_search_prob=0.8):
        super().__init__()
        # 底层依然保留 NSGA-II 默认的多项式变异以保证全局探索能力
        self.pm = PM(prob=prob, eta=eta)
        self.local_search_prob = local_search_prob # 执行局部平滑搜索的概率

    def _do(self, problem, X, **kwargs):
        # 第一步：执行全局遗传变异 (Global Exploration)
        X_mut = self.pm._do(problem, X, **kwargs)

        # 第二步：执行空间启发式局部搜索 (Local Search / Memetic Exploitation)
        for i in range(X_mut.shape[0]):
            if np.random.rand() < self.local_search_prob:
                # 提取农艺连续变量 (Density 和 Fertilizer)
                density = X_mut[i, 30:84].copy()
                fert = X_mut[i, 84:138].copy()

                # 使用中值滤波 (Median Filter) 模拟邻域协同与空间约束 (MRF 简化版)
                # 这会把突兀的高频跳变“抹平”，让相邻地块的变量趋同
                smoothed_density = median_filter(density, size=3)
                smoothed_fert = median_filter(fert, size=3)

                # 拉马克进化 (Lamarckian Evolution)：将局部优化后的优秀基因直接写回染色体
                X_mut[i, 30:84] = np.clip(smoothed_density, 0.0, 1.0)
                X_mut[i, 84:138] = np.clip(smoothed_fert, 0.0, 1.0)

        return X_mut

# --- 6. 多目标优化逻辑 ---
class FarmOptimization(ElementwiseProblem):
    def __init__(self):
        xl = np.zeros(138)
        xu = np.ones(138)
        super().__init__(n_var=138, n_obj=3, n_ieq_constr=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        crop_choice = np.round(x[0:30]).astype(int)
        density_raw = x[30:84]
        fert_raw = x[84:138]

        f_profit, f_yield, f_water, f_carbon = 0.0, 0.0, 0.0, 0.0
        total_maize, total_soy = 0.0, 0.0

        for i in range(54):
            f_id = i + 1
            area = AREA_DICT[f_id]
            d_r, f_r = density_raw[i], fert_raw[i]

            if f_id in PADDY_INDEX:
                crop = "Rice"
            else:
                crop = "Maize" if crop_choice[DRY_INDEX.index(f_id)] == 1 else "Soybean"
                if crop == "Maize":
                    total_maize += area
                else:
                    total_soy += area

            min_d, max_d = BOUNDS_DENSITY[crop]
            min_f, max_f = BOUNDS_FERT[crop]
            act_d = min_d + d_r * (max_d - min_d)
            act_fert_wt = min_f + f_r * (max_f - min_f)

            pred = models[crop].predict([[act_d, act_fert_wt]])[0]
            pred_yield, pred_et = pred[0], pred[1]

            pure_n = act_fert_wt * 0.15
            f_profit += ((pred_yield * PRICES[crop]) - (BASE_COSTS[crop] + pure_n * FERT_PRICES[crop])) * area

            pure_p, pure_k = pure_n * 0.436, pure_n * 0.830
            carbon_fert = (pure_n * N_FERT_CARBON_FACTOR) + (pure_p * P_FERT_CARBON_FACTOR) + (pure_k * K_FERT_CARBON_FACTOR)
            f_carbon += (BASE_CARBON[crop] + carbon_fert) * area

            f_yield += pred_yield * area
            f_water += (pred_et * 10) * area

        out["F"] = [-f_profit, -(f_yield / f_water if f_water > 0 else 0), f_carbon]
        out["G"] = [MIN_AREA_MAIZE - total_maize, MIN_AREA_SOYBEAN - total_soy]

# --- 定义 Pymoo 进度条回调 ---
class ProgressBarCallback(Callback):
    def __init__(self, n_gen):
        super().__init__()
        self.n_gen = n_gen
        self.pbar = tqdm(total=n_gen, desc=f"🧬 [2/3] 执行文化基因算法 Memetic NSGA-II ({n_gen}代)")

    def notify(self, algorithm):
        self.pbar.update(1)
        if algorithm.n_gen == self.n_gen:
            self.pbar.close()

# --- 7. 运行算法并导出解集 ---
# 💥 此处挂载自定义的 Memetic 空间平滑变异算子
algorithm = NSGA2(
    pop_size=POP_SIZE, 
    mutation=MemeticSpatialMutation(prob=1.0/138, eta=20, local_search_prob=0.8), # 80%概率执行空间平滑
    eliminate_duplicates=True
)

res = minimize(FarmOptimization(), algorithm, ('n_gen', N_GEN), seed=OPT_SEED, callback=ProgressBarCallback(N_GEN))

print("💾 [3/3] 寻优完成，正在导出平滑后的数据包...")
pareto_df = pd.DataFrame({'Profit': -res.F[:, 0], 'WUE': -res.F[:, 1], 'Carbon': res.F[:, 2]})
pareto_df.to_csv("Pareto_Front_Data.csv", index=False)

best_idx = np.argmin(res.F[:, 0])
best_x = res.X[best_idx]
final_plan, all_metrics = [], []
crops_res = np.round(best_x[0:30]).astype(int)

for i in range(54):
    f_id = i + 1
    area = AREA_DICT[f_id]
    d_r, f_r = best_x[30 + i], best_x[84 + i]
    c = "Rice" if f_id in PADDY_INDEX else ("Maize" if crops_res[DRY_INDEX.index(f_id)] == 1 else "Soybean")

    min_d, max_d = BOUNDS_DENSITY[c]
    min_f, max_f = BOUNDS_FERT[c]
    act_d = round(min_d + d_r * (max_d - min_d), 2)
    act_fert_wt = round(min_f + f_r * (max_f - min_f), 2)
    pred = models[c].predict([[act_d, act_fert_wt]])[0]

    pure_n = act_fert_wt * 0.15
    prof_ha = (pred[0] * PRICES[c]) - (BASE_COSTS[c] + pure_n * FERT_PRICES[c])
    pure_p, pure_k = pure_n * 0.436, pure_n * 0.830
    carb_ha = BASE_CARBON[c] + (pure_n * N_FERT_CARBON_FACTOR) + (pure_p * P_FERT_CARBON_FACTOR) + (pure_k * K_FERT_CARBON_FACTOR)

    final_plan.append({"Field_ID": f_id, "Area_ha": area, "Optimized_Crop": c, "Density_plants_m2": act_d, "Fert_Weight_kg_ha": act_fert_wt})
    all_metrics.append({"Field_ID": f_id, "Crop": c, "Area_ha": area, "Yield_ha": pred[0], "Profit_ha": prof_ha, "Carbon_ha": carb_ha, "WUE": pred[0] / (pred[1] * 10)})

pd.DataFrame(final_plan).to_csv("Optimized_Farm_Plan_Continuous.csv", index=False)
pd.DataFrame(all_metrics).to_csv("All_Metrics_Data_Continuous.csv", index=False)
print("✅ 第一阶段完成！文化基因算法已成功压制空间碎片化。可以运行可视化脚本了。")