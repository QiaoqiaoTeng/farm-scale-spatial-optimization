import os
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import qmc
import warnings

# 忽略 pandas 警告
warnings.filterwarnings('ignore')

# ================= 核心配置区 =================
DSSAT_EXE = r"REPLACE_WITH_YOUR_DSSAT48_EXECUTABLE"
WORK_DIR = r"REPLACE_WITH_YOUR_DSSAT_WORKING_DIR"
SAMPLES_PER_CROP = 700  # 每种作物跑700次


# ==============================================

def generate_lhs_design(crop_name, pop_bounds, fert_weight_bounds, num_samples):
    sampler = qmc.LatinHypercube(d=2, seed=42)
    sample = sampler.random(n=num_samples)

    l_bounds = [pop_bounds[0], fert_weight_bounds[0]]
    u_bounds = [pop_bounds[1], fert_weight_bounds[1]]
    scaled_samples = qmc.scale(sample, l_bounds, u_bounds)

    # 变量名改为复合肥总重量
    df = pd.DataFrame(scaled_samples, columns=['Density', 'Fert_Weight_kg_ha'])
    df['Crop'] = crop_name

    rice_plots = list(range(10, 22)) + list(range(36, 48))
    non_rice_plots = [i for i in range(1, 55) if i not in rice_plots]

    if crop_name == 'Rice':
        plot_pool = rice_plots
    else:
        plot_pool = non_rice_plots

    selected_plots = np.random.choice(plot_pool, size=num_samples, replace=True)
    df['Soil_ID'] = [f"XY{str(p).zfill(8)}" for p in selected_plots]
    return df


def run_dssat_batch(df_design):
    results = []

    for index, row in df_design.iterrows():
        crop = row['Crop']
        pop = round(row['Density'], 1)

        # ======== 核心：从 15-15-15 复合肥重量反推纯元素 ========
        fert_weight = int(round(row['Fert_Weight_kg_ha']))

        # 纯 N = 复合肥总重 * 15%
        n_fert = int(round(fert_weight * 0.15))
        # 纯 P = 复合肥总重 * 15% (P2O5) * 0.436
        p_fert = int(round(fert_weight * 0.15 * 0.436))
        # 纯 K = 复合肥总重 * 15% (K2O) * 0.830
        k_fert = int(round(fert_weight * 0.15 * 0.830))
        # =========================================================

        soil_id = row['Soil_ID']

        if crop == 'Maize':
            tpl_file, out_file, model_code = "TPL_MAIZE.MZX", "CHXF2401.MZX", "MZCER048"
        elif crop == 'Rice':
            tpl_file, out_file, model_code = "TPL_RICE.RIX", "CHXF2403.RIX", "RICER048"
        else:
            tpl_file, out_file, model_code = "TPL_SOY.SBX", "CHXF2402.SBX", "SBGRO048"

        with open(os.path.join(WORK_DIR, tpl_file), 'r', encoding='utf-8') as f:
            template = f.read()

        if "{POP}" not in template or "{FN}" not in template or "{SOIL_ID}" not in template:
            print(f"❌ 致命错误：{tpl_file} 缺少占位符！")
            return pd.DataFrame()

        new_content = template.replace("{SOIL_ID} ", f"{soil_id:<10}")
        new_content = new_content.replace("{POP} {POE}", f"{pop:>5.1f} {pop:>5.1f}")

        # 独立替换保证对齐
        new_content = new_content.replace("{FN}", f"{n_fert:>4}")
        new_content = new_content.replace("{FP}", f"{p_fert:>4}")
        new_content = new_content.replace("{FK}", f"{k_fert:>4}")

        with open(os.path.join(WORK_DIR, out_file), 'w', newline='\r\n', encoding='utf-8') as f:
            f.write(new_content)

        file_str = out_file.ljust(92)[:92]
        batch_content = "$BATCH(SIMULATION)\n! Directory    @FILEX                                                                                        TRTNO     RP     SQ     OP     CO\n"
        batch_content += f"{file_str}{1:>6}{1:>6}{0:>6}{1:>6}{0:>6}\n"

        with open(os.path.join(WORK_DIR, "DSSBatch.v48"), 'w', newline='\r\n', encoding='utf-8') as f:
            f.write(batch_content)

        summary_path = os.path.join(WORK_DIR, "Summary.OUT")
        warning_path = os.path.join(WORK_DIR, "WARNING.OUT")
        if os.path.exists(summary_path): os.remove(summary_path)
        if os.path.exists(warning_path): os.remove(warning_path)

        cmd = f"{DSSAT_EXE} {model_code} B DSSBatch.v48"
        subprocess.run(cmd, cwd=WORK_DIR, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        yield_kg, et_mm = np.nan, np.nan
        status = "Success"

        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    lines = f.readlines()
                    header_line, data_line = [], []
                    for line in reversed(lines):
                        if line.strip() and not line.startswith('!') and not line.startswith('@'):
                            if not data_line: data_line = line.split()
                        elif line.startswith('@'):
                            header_line = line.split()
                            break
                    if header_line and header_line[0] == '@':
                        header_line = header_line[1:]

                    idx_hwam = header_line.index('HWAM') if 'HWAM' in header_line else -1
                    idx_etcm = header_line.index('ETCM') if 'ETCM' in header_line else -1
                    if idx_hwam != -1 and len(data_line) > idx_hwam: yield_kg = float(data_line[idx_hwam])
                    if idx_etcm != -1 and len(data_line) > idx_etcm: et_mm = float(data_line[idx_etcm])
            except Exception as e:
                status = f"Parse Error: {e}"
        else:
            status = "DSSAT Failed"

        # 不再收集 N_Leach，只收集肥量、产量、蒸腾
        results.append({
            "Sim_ID": index + 1, "Crop": crop, "Soil_ID": soil_id,
            "Density_plants_m2": pop,
            "Fert_Weight_kg_ha": fert_weight,  # 核心指标：复合肥重量
            "Yield_kg_ha": yield_kg, "ET_mm": et_mm, "Status": status
        })

        print(
            f"Sim {index + 1:04d} | {crop: <7} | {soil_id} | 密度: {pop:>5.1f} | 复合肥: {fert_weight:>4}kg (折纯N:{n_fert}kg,纯P:{p_fert}kg,纯K:{k_fert}kg) -> 产量: {yield_kg} kg/ha")

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("🚀 1. 正在生成全量实验方案 (LHS)...")

    # ======== 复合肥重量的边界调整 ========
    # 旧的纯氮边界 / 0.15 = 大致的复合肥边界 (向下取整以符合常理)
    # 比如水稻原纯氮 280-500，对应复合肥就是 1866-3333，这里取整为 1800-3400
    df_rice = generate_lhs_design("Rice", [15, 40], [1800, 3400], SAMPLES_PER_CROP)
    df_maize = generate_lhs_design("Maize", [3, 18], [900, 2700], SAMPLES_PER_CROP)
    df_soy = generate_lhs_design("Soybean", [10, 35], [250, 2000], SAMPLES_PER_CROP)
    # ======================================

    design_all = pd.concat([df_maize, df_soy, df_rice], ignore_index=True)
    plan_path = os.path.join(WORK_DIR, "Planting_Plan_Details.csv")
    design_all.to_csv(plan_path, index_label="Sim_ID")

    print(f"⚙️ 2. 开始批量运行 DSSAT 引擎 (共 {len(design_all)} 次运行)...")
    results_df = run_dssat_batch(design_all)

    if results_df is not None and not results_df.empty:
        final_data_path = os.path.join(WORK_DIR, "ML_Training_Data.csv")
        results_df.to_csv(final_data_path, index=False)
        print(f"✅ 完整训练集已存入: {final_data_path}")

        valid_results = results_df[results_df['Status'] == 'Success']
        if not valid_results.empty:
            best_plans = valid_results.loc[valid_results.groupby('Crop')['Yield_kg_ha'].idxmax()]
            best_plans.to_csv(os.path.join(WORK_DIR, "Best_Observed_Plan.csv"), index=False)
            print("\n🏆 4. 提取本次随机实验中的最优观测方案...")
            for _, row in best_plans.iterrows():
                print(f"   🌾 {row['Crop']}: 地块 {row['Soil_ID']}, 密度 {row['Density_plants_m2']}, "
                      f"复合肥 {row['Fert_Weight_kg_ha']} kg/ha -> 最高产量 {row['Yield_kg_ha']} kg/ha")