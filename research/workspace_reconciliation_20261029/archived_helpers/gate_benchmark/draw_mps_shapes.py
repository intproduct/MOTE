"""Draw the actual state origin and tensor-network contraction, not training data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':18,'axes.unicode_minus':False})
root=Path('outputs/gate_benchmark/figures')
root.mkdir(parents=True,exist_ok=True)
ink='#17283C'; muted='#526171'; blue='#286FA7'; orange='#B86C1B'; green='#287E67'

def text(ax,x,y,label,size=18,color=ink,ha='center',weight='normal'):
    return ax.text(x,y,label,ha=ha,va='center',fontsize=size,color=color,weight=weight)

def box(ax,x,y,w,h,label,color=blue,size=18):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.04,rounding_size=.1',
                              facecolor=color+'0D',edgecolor=color,linewidth=1.5))
    text(ax,x,y,label,size)

def arrow(ax,a,b,color=muted):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,color=color,linewidth=1.6))

fig,ax=plt.subplots(figsize=(14,7.3),dpi=180)
fig.patch.set_facecolor('white');ax.set(xlim=(0,14),ylim=(0,7.3));ax.axis('off')
text(ax,7,6.95,'实际代码：输入特征 u 与内部状态 h 来自两条不同路径',22,weight='bold')
box(ax,1.25,5.5,2.0,.9,'输入 x\n[L, D]',blue)
box(ax,4.1,5.5,2.7,1.2,'LayerNorm\nLinear：D → 16\ntanh',blue,17)
box(ax,7.6,5.5,2.4,1.2,'输入特征 u\n[L, 16]',blue)
arrow(ax,(2.3,5.5),(2.7,5.5),blue);arrow(ax,(5.5,5.5),(6.35,5.5),blue)
box(ax,7.6,3.55,2.4,.85,'取第 i 列\nu_i：[L, 1]',blue,17)
arrow(ax,(7.6,4.85),(7.6,4.02),blue)
box(ax,1.25,1.75,2.0,1.1,'可学习边界 b\n[8]',orange,17)
box(ax,4.1,1.75,2.7,1.1,'沿 token 维展开\n同一边界复制 L 份',orange,17)
box(ax,7.6,1.75,2.4,1.1,'初始状态 h0\n[L, 8]',orange)
arrow(ax,(2.3,1.75),(2.7,1.75),orange);arrow(ax,(5.5,1.75),(6.35,1.75),orange)
box(ax,11.55,3.55,3.7,1.65,'第 i 个 site\nh @ A_i：[L, 8]\nu_i 调制、残差相加\n并做 L2 归一化',green,17)
arrow(ax,(8.85,3.55),(9.63,3.55),blue)
arrow(ax,(8.85,1.75),(10.1,2.67),orange)
box(ax,11.55,1.05,3.0,.9,'更新后的 h_i：[L, 8]',green,17)
arrow(ax,(11.55,2.67),(11.55,1.55),green)
text(ax,11.55,5.35,'A_i 是 [8, 8]\n作用于 h 的最后一维',17,muted)
arrow(ax,(11.55,4.8),(11.55,4.44),green)
text(ax,4.9,.45,'从 h0 开始重复 16 次；h 的宽度始终为 χ = 8。',17,muted)
fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99)
fig.savefig(root/'mps_u_and_h_shapes.png',dpi=180)
fig.savefig(root/'mps_u_and_h_shapes.svg')
plt.close(fig)

def edge(ax,a,b,label=None,label_xy=None):
    ax.plot([a[0],b[0]],[a[1],b[1]],color=ink,lw=1.7,zorder=1)
    if label is not None:
        x,y=label_xy or ((a[0]+b[0])/2,(a[1]+b[1])/2+.3)
        text(ax,x,y,label,16,muted)

def core(ax,x,y,name):
    ax.add_patch(Circle((x,y),.4,fc='#EAF4F0',ec=green,lw=1.7,zorder=3))
    text(ax,x,y,name,18)

def feature(ax,x,y,name):
    box(ax,x,y,1.7,.65,name,blue,17)

fig,ax=plt.subplots(figsize=(14,9.4),dpi=180)
fig.patch.set_facecolor('white');ax.set(xlim=(0,14),ylim=(0,9.4));ax.axis('off')
text(ax,7,9.05,'张量网络视角：h_i 是左侧部分收缩后留下的边界向量',22,weight='bold')
text(ax,7,8.5,'图示单个 token；先展开矩阵积部分，暂不画每步数值归一化',17,muted)

# One full chain, with the first two sites highlighted as the contraction region.
ax.add_patch(FancyBboxPatch((.35,4.95),5.5,3.05,boxstyle='round,pad=.03,rounding_size=.12',
                          fc='#F0F5FA',ec=blue,lw=1.3,ls='--',zorder=0))
y=5.8
ax.add_patch(Circle((.9,y),.28,fc='#FAF0E6',ec=orange,lw=1.6,zorder=3));text(ax,.9,y,'b',18)
for x,name in [(2.65,'G1'),(4.65,'G2'),(9.1,'G16')]:core(ax,x,y,name)
for x,name in [(2.65,'φ1 = (1,u1)'),(4.65,'φ2 = (1,u2)'),(9.1,'φ16 = (1,u16)')]:
    feature(ax,x,7.55,name);edge(ax,(x,7.18),(x,6.2),'2',(x+.22,6.7))
edge(ax,(1.18,y),(2.25,y),'8')
edge(ax,(3.05,y),(4.25,y),'8')
edge(ax,(5.05,y),(6.25,y),'8',(5.35,y+.34))
text(ax,6.7,y,'…',28)
edge(ax,(7.15,y),(8.7,y),'8')
box(ax,11.2,y,1.6,.85,'读出 R',green,17)
edge(ax,(9.5,y),(10.35,y),'8')
edge(ax,(12.05,y),(13.25,y),'16')
text(ax,12.8,5.1,'专家指标 e',16,muted)
for x in (2.65,4.65,9.1):text(ax,x,5.15,'[8, 2, 8]',16,muted)
text(ax,.9,5.15,'[8]',16,muted)
text(ax,11.2,5.1,'[8, 16]',16,muted)
ax.plot([5.7,5.7],[5.45,6.15],color=orange,lw=3,zorder=4)
text(ax,6.95,4.65,'切口只有一条腿，维数为 8',17,orange)
arrow(ax,(2.95,4.84),(2.95,3.2),blue)
text(ax,1.15,4.0,'收缩\n左侧区域',17,blue)

# Replace that entire region by its eight-component contraction vector.
y=2.3
box(ax,2.95,y,2.05,1.05,'h2\n[8]',orange,19)
core(ax,5.0,y,'G3');core(ax,9.1,y,'G16')
feature(ax,5.0,3.9,'φ3 = (1,u3)');edge(ax,(5.0,3.53),(5.0,2.7),'2',(5.22,3.1))
feature(ax,9.1,3.9,'φ16 = (1,u16)');edge(ax,(9.1,3.53),(9.1,2.7),'2',(9.32,3.1))
edge(ax,(4.02,y),(4.6,y),'8')
edge(ax,(5.4,y),(6.25,y),'8')
text(ax,6.7,y,'…',28)
edge(ax,(7.15,y),(8.7,y),'8')
box(ax,11.2,y,1.6,.85,'读出 R',green,17)
edge(ax,(9.5,y),(10.35,y),'8')
edge(ax,(12.05,y),(13.25,y),'16')
text(ax,7,.8,'一个 token 的部分收缩结果：[8]；L 个 token 并排计算：[L, 8]',19,weight='bold')
text(ax,7,.25,'对 h_i 做 L2 归一化只改变数值，不会增加或删除这条 8 维指标。',16,muted)
fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99)
fig.savefig(root/'mps_partial_contraction.png',dpi=180)
fig.savefig(root/'mps_partial_contraction.svg')
plt.close(fig)
print(root/'mps_u_and_h_shapes.png')
print(root/'mps_partial_contraction.png')
