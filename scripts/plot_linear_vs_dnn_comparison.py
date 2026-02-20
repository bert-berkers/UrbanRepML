import matplotlib.pyplot as plt
import numpy as np

# Data
levels = ['Level 1\n(2 classes)', 'Level 2\n(4 classes)', 'Level 3\n(8 classes)', 'Level 4\n(16 classes)',
          'Level 5\n(25 classes)', 'Level 6\n(56 classes)', 'Level 7\n(106 classes)']
linear_acc = [0.8526, 0.7176, 0.4500, 0.3033, 0.2292, 0.1216, 0.0778]
dnn_acc = [0.8937, 0.7784, 0.5590, 0.4063, 0.3349, 0.2601, 0.1727]

# Setup
x = np.arange(len(levels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

# Create bars
bars1 = ax.bar(x - width/2, linear_acc, width, label='Linear', color='steelblue', alpha=0.9)
bars2 = ax.bar(x + width/2, dnn_acc, width, label='DNN', color='coral', alpha=0.9)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Customize
ax.set_xlabel('Hierarchy Level', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax.set_title('Linear vs DNN Classification Probe — Urban Taxonomy', fontsize=13, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(levels, fontsize=10)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(r'C:\Users\Bert Berkers\PycharmProjects\UrbanRepML\data\study_areas\netherlands\stage3_analysis\linear_vs_dnn_comparison.png',
            dpi=150, bbox_inches='tight')
print("Chart saved to: data/study_areas/netherlands/stage3_analysis/linear_vs_dnn_comparison.png")
plt.close()
