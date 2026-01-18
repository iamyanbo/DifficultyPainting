"""
Generate visualizations for DifficultyPainting README.
"""
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# --- 1. Detector Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 6))

classes = ['Car', 'Pedestrian', 'Cyclist', 'mAP']
pp_scores = [86.53, 68.34, 85.21, 80.03]
second_scores = [88.33, 73.71, 85.49, 82.51]

x = np.arange(len(classes))
width = 0.35

bars1 = ax.bar(x - width/2, pp_scores, width, label='PointPillars + Difficulty', color='#5A9BD5')
bars2 = ax.bar(x + width/2, second_scores, width, label='SECOND + Difficulty', color='#ED7D31')

ax.set_ylabel('3D AP (Moderate) %')
ax.set_title('Difficulty-Aware Detection Results on KITTI')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.set_ylim(60, 95)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('assets/detector_comparison.png', dpi=150, bbox_inches='tight')
print("Saved detector_comparison.png")
plt.close()

# --- 2. Feature Ablation Chart ---
fig, ax = plt.subplots(figsize=(10, 6))

configs = ['Image Only', '+ Depth', '+ BBox', '+ Depth + BBox', '+ All Features']
correlations = [64.3, 78.2, 71.5, 89.0, 85.3]
colors = ['#A5A5A5', '#7CB9E8', '#B0C4DE', '#2E8B57', '#CD5C5C']

bars = ax.barh(configs, correlations, color=colors)

ax.set_xlabel('Correlation with Ground Truth (%)')
ax.set_title('Feature Ablation Study - Difficulty Predictor')
ax.set_xlim(50, 95)

# Highlight best
bars[3].set_color('#2E8B57')
bars[3].set_edgecolor('black')
bars[3].set_linewidth(2)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.1f}%',
                xy=(width, bar.get_y() + bar.get_height()/2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('assets/ablation_chart.png', dpi=150, bbox_inches='tight')
print("Saved ablation_chart.png")
plt.close()

# --- 3. Pipeline Diagram (simple) ---
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')

# Draw pipeline boxes
boxes = [
    ('Baseline\nDetector', 0.05),
    ('Extract\nDifficulty', 0.25),
    ('Train\nPredictor', 0.45),
    ('Paint\nLiDAR', 0.65),
    ('Train\nDifficulty-Aware', 0.85)
]

for text, x in boxes:
    rect = plt.Rectangle((x-0.08, 0.3), 0.16, 0.4, fill=True, 
                          facecolor='#5A9BD5', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, 0.5, text, ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Draw arrows
for i in range(len(boxes)-1):
    ax.annotate('', xy=(boxes[i+1][1]-0.1, 0.5), xytext=(boxes[i][1]+0.1, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('assets/pipeline_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved pipeline_diagram.png")
plt.close()

print("\nAll visualizations generated!")
