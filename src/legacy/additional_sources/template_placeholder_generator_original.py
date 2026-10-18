import os


def generate_templates():
    # 原始文件与目标模板文件的映射字典
    file_mapping = {
        "CHXF2401.MZX": "TPL_MAIZE.MZX",
        "CHXF2403.RIX": "TPL_RICE.RIX",
        "CHXF2402.SBX": "TPL_SOY.SBX"
    }

    for original_file, template_file in file_mapping.items():
        # 检查文件是否存在
        if not os.path.exists(original_file):
            print(f"⚠️ 找不到文件: {original_file}，请确保脚本和文件在同一文件夹下。")
            continue

        with open(original_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # ==========================================
        # 1. 替换土壤/地块 ID (占10个字符)
        # ==========================================
        content = content.replace("XY00000001", "{SOIL_ID} ")

        # ==========================================
        # 2. 替换种植密度 PPOP 和 PPOE (严格保证11个字符长度)
        # ==========================================
        if original_file.endswith(".MZX"):
            # 原长度11，"{POP} {POE}" 长度也是 5 + 1 + 5 = 11
            content = content.replace("    6     6", "{POP} {POE}")

        elif original_file.endswith(".RIX"):
            content = content.replace("   75    75", "{POP} {POE}")

        elif original_file.endswith(".SBX"):
            content = content.replace("   25    25", "{POP} {POE}")

        # ==========================================
        # 3. 替换施肥量 FAMN (无机氮，严格保证5个字符长度)
        # ==========================================
        content = content.replace("  362", " {FN}")
        content = content.replace("   17", " {FP}")
        content = content.replace("   32", " {FK}")

        # 将替换后的内容写入新的模板文件
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ 成功生成模板: {template_file}")


if __name__ == "__main__":
    print("开始自动生成 DSSAT 模板文件...")
    generate_templates()
    print("大功告成！你可以继续运行批量实验生成的脚本了。")