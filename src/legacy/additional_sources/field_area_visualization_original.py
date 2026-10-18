import matplotlib.pyplot as plt

# ==========================================
# 1. Data Preparation
# ==========================================
plot_ids = list(range(1, 55))
areas = [
    500, 500, 500, 500, 704, 605, 805, 226, 339,                   # 1-9 (Dry)
    890, 696, 1000, 459, 501, 800, 694, 821, 572, 864, 710, 306,    # 10-21 (Paddy)
    3670, 3670, 3670, 536, 536, 752, 313, 313, 198, 132,
    100, 100, 100, 100,                                            # 22-35 (Dry)
    585, 585, 780, 780, 680, 1253, 473, 940, 740, 1105, 250, 1712, # 36-47 (Paddy)
    1644, 1766, 974, 661, 905, 1094, 3500                          # 48-54 (Dry)
]

# ==========================================
# 2. Advanced Color Palette (Morandi Style)
# ==========================================
# Using sophisticated, muted tones for a professional look
color_dryland = '#778899'  # LightSlateGray
color_paddy = '#66786D'    # Dark Sage
text_color = '#333333'     # Dark Grey for readability
grid_color = '#EEEEEE'     # Very light grey for subtle grid

colors = []
for i in plot_ids:
    if (10 <= i <= 21) or (36 <= i <= 47):
        colors.append(color_paddy)
    else:
        colors.append(color_dryland)

# ==========================================
# 3. Plotting with English Fonts
# ==========================================
# Explicitly use a standard Western font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(18, 9))
ax.set_facecolor('#FFFFFF') # Clean white background

# Create bars
bars = ax.bar(plot_ids, areas, color=colors, edgecolor='none', alpha=0.9, width=0.7)

# --- Grid and Spines ---
ax.grid(axis='y', color=grid_color, linestyle='-', linewidth=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

# --- Labels and Title (All English) ---
ax.set_xlabel('Plot ID', fontsize=14, color=text_color, fontweight='bold', labelpad=10)
ax.set_ylabel('Area (ha)', fontsize=14, color=text_color, fontweight='bold', labelpad=10)
ax.set_title('Distribution of Land Area by Cultivation Type', fontsize=18, color=text_color, fontweight='bold', pad=25)

# --- X and Y Ticks ---
ax.set_xticks(plot_ids)
ax.set_xticklabels(plot_ids, rotation=90, fontsize=12, color=text_color)
ax.tick_params(axis='y', colors=text_color, labelsize=11)

# --- Value Annotations ---
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{int(height)}', ha='center', va='bottom',
            rotation=90, fontsize=10, color='#555555')

# --- Legend ---
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=color_dryland, lw=8, label='Dry Land'),
    Line2D([0], [0], color=color_paddy, lw=8, label='Paddy Field')
]
max_height = max(areas)
ax.set_ylim(0, max_height * 1.2)
ax.legend(handles=legend_elements, loc='best', frameon=False, fontsize=12)
# ==========================================
# 4. Final Layout
# ==========================================

plt.tight_layout()
file_name = f'PLOTS54_AREA.png'
plt.savefig(file_name, bbox_inches='tight',dpi=600)

plt.show()