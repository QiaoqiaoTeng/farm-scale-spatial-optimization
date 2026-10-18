# =====================================================================
# Part 1: Farm Spatial Optimization Engine (复合肥 + SHAP + 3目标优化)
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
from pymoo.optimize import minimize

# 屏蔽警告并设置中文字体防乱码
warnings.filterwarnings(action='ignore', category=UserWarning)
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# =====================================================================
# --- 0. 全局算法参数设置 ---
# =====================================================================
# 多目标优化 (NSGA-II) 相关参数
N_GEN = 200               # 进化代数 (原 120)
POP_SIZE = 100            # 种群大小 (原 100)
OPT_SEED = 1              # 优化算法随机种子 (原 1)

# 机器学习代理模型 (Random Forest) 相关参数
RF_N_ESTIMATORS = 100     # 随机森林生成的树的数量 (原 100)
RF_RANDOM_STATE = 42      # 随机森林的随机种子 (原 42)
# =====================================================================

# --- 1. 经济参数与环境约束 ---
PRICE_MAIZE, PRICE_SOYBEAN, PRICE_RICE = 2.6401, 4.7180, 2.8366
BASE_COST_MAIZE, BASE_COST_SOYBEAN, BASE_COST_RICE = 7591.9216, 6253.4778, 14249.5380

# 论文设定：纯氮的单价
FERT_PRICE_MAIZE, FERT_PRICE_SOYBEAN, FERT_PRICE_RICE = 5.9080, 4.7180, 6.8960

# 碳排因子 (纯元素的碳排系数)
N_FERT_CARBON_FACTOR = 8.3
P_FERT_CARBON_FACTOR = 1.2
K_FERT_CARBON_FACTOR = 0.8

BASE_CARBON_MAIZE, BASE_CARBON_SOYBEAN, BASE_CARBON_RICE = 1200, 800, 4500
MIN_AREA_MAIZE, MIN_AREA_SOYBEAN = 10000, 10000

# --- 2. 农艺实验连续搜索边界 (Min, Max) ---
BOUNDS_DENSITY = {
    "Rice": (15.0, 40.0),
    "Maize": (3.0, 18.0),
    "Soybean": (10.0, 35.0)
}
# 复合肥(Fert_Weight_kg_ha)的上下限区间
BOUNDS_FERT = {
    "Rice": (1800.0, 3400.0),
    "Maize": (900.0, 2700.0),
    "Soybean": (250.0, 2000.0)
}

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
print("🚀 [1/3] 正在加载数据并训练代理模型...")
try:
    data = pd.read_csv("ML_Training_Data.csv")
    crops = ["Maize", "Soybean", "Rice"]
    models = {}
    for crop in crops:
        df_crop = data[data['Crop'] == crop].dropna()
        # 接入顶部的全局参数
        rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE)
        rf.fit(df_crop[['Density_plants_m2', 'Fert_Weight_kg_ha']], df_crop[['Yield_kg_ha', 'ET_mm']])
        models[crop] = rf
except Exception as e:
    print(f"⚠️ 模型训练失败: {e}。请确保 ML_Training_Data.csv 存在且格式正确。")
    exit()

# --- 4.5 代理模型可解释性分析 (SHAP) ---
print("📊 [1.5/3] 正在生成 SHAP 机理分析图 (共6张)...")
for crop in crops:
    df_crop = data[data['Crop'] == crop].dropna()
    X_train = df_crop[['Density_plants_m2', 'Fert_Weight_kg_ha']]

    explainer = shap.TreeExplainer(models[crop])
    shap_values = explainer.shap_values(X_train)
    # 针对多输出模型，提取第一列 (Yield) 的 SHAP 值
    sv_yield = shap_values[0] if isinstance(shap_values, list) else shap_values

    # 图1：SHAP 全局重要性 (Bar)
    plt.figure(figsize=(6, 4))
    shap.summary_plot(sv_yield, X_train, plot_type="bar", show=False)
    plt.title(f"{crop} 特征全局重要性 (Yield)")
    plt.tight_layout()
    plt.savefig(f"Fig_SHAP_Bar_{crop}.png", dpi=300)
    plt.close()

    # 图2：SHAP 蜂群图 (Summary)
    plt.figure(figsize=(7, 5))
    shap.summary_plot(sv_yield, X_train, show=False)
    plt.title(f"{crop} 农艺措施非线性影响分布 (SHAP)")
    plt.tight_layout()
    plt.savefig(f"Fig_SHAP_Summary_{crop}.png", dpi=300)
    plt.close()


# --- 5. 多目标优化逻辑 ---
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
            pred_yield = pred[0]
            pred_et = pred[1]

            # 经济解耦：从复合肥折算纯氮算利润
            pure_n = act_fert_wt * 0.15
            f_profit += ((pred_yield * PRICES[crop]) - (BASE_COSTS[crop] + pure_n * FERT_PRICES[crop])) * area

            # 环境足迹：真实核算 NPK 排放
            pure_p = pure_n * 0.436
            pure_k = pure_n * 0.830
            carbon_fert = (pure_n * N_FERT_CARBON_FACTOR) + (pure_p * P_FERT_CARBON_FACTOR) + (
                        pure_k * K_FERT_CARBON_FACTOR)
            f_carbon += (BASE_CARBON[crop] + carbon_fert) * area

            f_yield += pred_yield * area
            f_water += (pred_et * 10) * area

        # 3个目标：利润(最大化转负), WUE(最大化转负), 碳排(最小化)
        out["F"] = [-f_profit, -(f_yield / f_water if f_water > 0 else 0), f_carbon]
        out["G"] = [MIN_AREA_MAIZE - total_maize, MIN_AREA_SOYBEAN - total_soy]


# --- 6. 运行算法并导出解集 ---
print(f"🧬 [2/3] 正在执行 {N_GEN} 代进化算法 (三目标联合寻优)...")
# 接入顶部的全局参数
res = minimize(FarmOptimization(), NSGA2(pop_size=POP_SIZE, eliminate_duplicates=True), ('n_gen', N_GEN), seed=OPT_SEED)

print("💾 [3/3] 寻优完成，正在导出数据包...")
pareto_df = pd.DataFrame({
    'Profit': -res.F[:, 0], 'WUE': -res.F[:, 1], 'Carbon': res.F[:, 2]
})
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
    carb_ha = BASE_CARBON[c] + (pure_n * N_FERT_CARBON_FACTOR) + (pure_p * P_FERT_CARBON_FACTOR) + (
                pure_k * K_FERT_CARBON_FACTOR)

    final_plan.append({"Field_ID": f_id, "Area_ha": area, "Optimized_Crop": c, "Density_plants_m2": act_d,
                       "Fert_Weight_kg_ha": act_fert_wt})
    all_metrics.append(
        {"Field_ID": f_id, "Crop": c, "Area_ha": area, "Yield_ha": pred[0], "Profit_ha": prof_ha, "Carbon_ha": carb_ha,
         "WUE": pred[0] / (pred[1] * 10)})

pd.DataFrame(final_plan).to_csv("Optimized_Farm_Plan_Continuous.csv", index=False)
pd.DataFrame(all_metrics).to_csv("All_Metrics_Data_Continuous.csv", index=False)
print("✅ 第一阶段完成！可以运行可视化脚本了。")