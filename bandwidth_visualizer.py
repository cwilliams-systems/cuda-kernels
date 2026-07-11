import matplotlib.pyplot as plt
import numpy as np

sizes = [256, 512, 1024, 2048, 4096]

cpu_gbs         = [None, None, None, None, None]
naive_cuda_gbs  = [None, None, 24.08, None, None]
tiled_cuda_gbs  = [None, None, 34.93, None, None]
pytorch_gbs     = [None, None, None,  None, None]

fig, ax = plt.subplots(figsize=(10, 6))

def plot_line(gbs_list, label, color, marker):
    x = [sizes[i] for i in range(len(sizes)) if gbs_list[i] is not None]
    y = [gbs_list[i] for i in range(len(sizes)) if gbs_list[i] is not None]
    if x:
        ax.plot(x, y, marker=marker, color=color, label=label,
                linewidth=2, markersize=8)

plot_line(cpu_gbs,        'CPU (cpp-matrix-multiply)',  '#888888', 's')
plot_line(naive_cuda_gbs, 'Naive CUDA',                 '#E05A2B', 'o')
plot_line(tiled_cuda_gbs, 'Tiled CUDA (TILE_WIDTH=16)', '#185FA5', '^')
plot_line(pytorch_gbs,    'PyTorch CUDA baseline',      '#3B6D11', 'D')

ax.set_xlabel('Matrix size (N×N)', fontsize=13)
ax.set_ylabel('Memory Throughput GB/s (Nsight Compute)', fontsize=13)
ax.set_title('GEMM Performance: CPU vs Naive CUDA vs Tiled CUDA vs PyTorch\nTesla T4 · double precision · cwilliams-systems', fontsize=13)
ax.set_xticks(sizes)
ax.set_xticklabels([f'{n}×{n}' for n in sizes], rotation=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

for i, (n, gbs) in enumerate(zip(sizes, tiled_cuda_gbs)):
    if gbs is not None:
        ax.annotate(f'{gbs:.1f} GB/s', (n, gbs),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=10, color='#185FA5')

for i, (n, gbs) in enumerate(zip(sizes, naive_cuda_gbs)):
    if gbs is not None:
        ax.annotate(f'{gbs:.1f} GB/s', (n, gbs),
                    textcoords="offset points", xytext=(0, -15),
                    ha='center', fontsize=10, color='#E05A2B')

plt.tight_layout()
plt.savefig('bandwidth_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: bandwidth_chart.png")
