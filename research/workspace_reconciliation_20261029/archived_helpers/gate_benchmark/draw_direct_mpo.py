"""Exact index diagram for a proposed amplitude-encoded MPO router (not trained)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 17,
                     'axes.unicode_minus': False, 'svg.fonttype': 'path'})
ink, gray, blue, green = '#17283C', '#536373', '#286FA7', '#287E67'
fig, ax = plt.subplots(figsize=(14, 9.1), dpi=170)
fig.patch.set_facecolor('white')
ax.set(xlim=(0, 14), ylim=(0, 9.1)); ax.axis('off')

def text(x, y, value, size=17, color=ink, ha='center', weight='normal'):
    ax.text(x, y, value, ha=ha, va='center', fontsize=size, color=color, weight=weight)

def box(x, y, w, h, value, size=17, color=green):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.06',
        facecolor=color+'0C', edgecolor=color, linewidth=1.5, zorder=3))
    text(x, y, value, size)

def line(x1, y1, x2, y2, color=ink):
    ax.plot([x1, x2], [y1, y2], color=color, lw=1.8, zorder=1)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=16, color=gray, lw=1.5))

text(7, 8.72, '你的方案：联合幅度张量 → MPO → 联合幅度张量', 23, weight='bold')
text(7, 8.16, '每个 token 独立计算；B、L 只是批次轴，图中省略', 17, gray)

box(1.55, 7.32, 2.3, .74, '输入 x\n[B, L, D]', 16, blue)
box(5.2, 7.32, 3.2, .74, '线性投影 P\n[B, L, 16]', 16, blue)
box(10.15, 7.32, 4.7, .74, 'reshape\n[B, L, 2, 2, 2, 2]', 16, blue)
arrow(2.74, 7.32, 3.55, 7.32); arrow(6.84, 7.32, 7.74, 7.32)
line(.45, 6.65, 13.55, 6.65, '#D8E0E7')

text(7, 6.17, '上方 4 条开放腿：输出索引 s1、s2、s3、s4，各有 2 个取值', 18)
xs = [2.4, 5.45, 8.5, 11.55]
for k, x in enumerate(xs, 1):
    line(x, 4.98, x, 5.62)
    text(x+.35, 5.42, f's{k}', 17, green)
    box(x, 4.55, 1.25, .82, f'C{k}', 22, green)
    line(x, 3.02, x, 4.12)
    text(x+.36, 3.67, f'j{k}: 2', 17, blue)

for k in range(3):
    line(xs[k]+.645, 4.55, xs[k+1]-.645, 4.55)
    text((xs[k]+xs[k+1])/2, 4.91, f'α{k+1}: χ', 17, gray)

line(.7, 4.55, xs[0]-.645, 4.55)
line(xs[-1]+.645, 4.55, 13.25, 4.55)
text(.95, 4.94, '边界 1', 15, gray)
text(12.97, 4.94, '边界 1', 15, gray)
box(7, 2.52, 11.9, .98, 'a[j1, j2, j3, j4]：一个联合输入张量，共 16 个数\n由投影输出直接 reshape；不是 4 个二维向量的直积', 17, blue)

text(7, 1.52, '收缩输入索引 j 和内部索引 α；留下 s → 输出 y[s1,s2,s3,s4]', 18)
text(7, .96, 'MPO 核形状：[χ左, 2输出, 2输入, χ右]；内部 χ 可取 2、4 等', 17, gray)
text(7, .42, '输出 flatten 为 16 维后读出；64 专家时对应 6 个物理索引', 17, gray)

out = Path(__file__).resolve().parent / 'figures'
out.mkdir(exist_ok=True)
for ext in ['png', 'svg']:
    fig.savefig(out / f'direct_amplitude_mpo.{ext}', bbox_inches='tight', pad_inches=.15)
plt.close(fig)
print(out / 'direct_amplitude_mpo.png')
