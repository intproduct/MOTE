"""Literal tensor contraction diagrams of the implemented residual recurrence."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':18,'axes.unicode_minus':False})
ink='#17283C'; gray='#536373'; blue='#286FA7'; green='#287E67'; orange='#B86C1B'
root=Path('outputs/gate_benchmark/figures')
fig,ax=plt.subplots(figsize=(14,9.4),dpi=180)
fig.patch.set_facecolor('white')
ax.set(xlim=(0,14),ylim=(0,9.4));ax.axis('off')

def txt(x,y,s,size=18,color=ink,ha='center',weight='normal'):
    ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,weight=weight)

def circle(x,y,s,r=.46):
    ax.add_patch(Circle((x,y),r,fc='#FAF0E6',ec=orange,lw=1.6,zorder=3))
    txt(x,y,s,21)

def box(x,y,w,h,s,size=19,color=green):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.03,rounding_size=.08',
                              fc=color+'0B',ec=color,lw=1.6,zorder=3))
    txt(x,y,s,size)

def line(x0,x1,y,label=None):
    ax.plot([x0,x1],[y,y],color=ink,lw=1.8,zorder=1)
    if label:txt((x0+x1)/2,y+.32,label,16,gray)

def arrow(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,color=gray,lw=1.5))

txt(7,9.03,'只画原公式：标量 u 调制矩阵，辅助状态 h 沿链更新',22,weight='bold')
txt(.45,8.35,'① 一步更新：两张收缩网络相加',19,ha='left',weight='bold')
y=7.35
circle(1.2,y,r'$h_{i-1}$')
line(1.66,2.55,y,'β：8')
txt(3.1,y,'+',27)
txt(4.15,y,r'$\frac{u_i}{4}$',28,blue)
txt(4.95,y,'×',24,blue)
circle(5.9,y,r'$h_{i-1}$')
line(6.36,7.4,y,'α：8')
box(8.0,y,1.1,.9,r'$A_i$',22)
line(8.58,9.55,y,'β：8')
txt(10.05,y,'=',25)
circle(11.05,y,r'$\widehat{h}_i$')
line(11.51,12.7,y,'β：8')
txt(1.2,6.65,'[8]',16,gray)
txt(5.9,6.65,'[8]',16,gray)
txt(8,6.65,'[8, 8]',16,gray)
txt(11.05,6.65,'[8]',16,gray)
txt(4.15,6.15,'u_i / 4 是标量系数，没有张量腿',17,blue)
txt(10.25,6.15,'两项留下同一条 β 腿，所以可以相加',17,gray)

box(2.0,5.2,1.9,.75,r'$\widehat{h}_i$'+'：[8]',19,orange)
box(5.3,5.2,2.6,.75,'L2 归一化',18,gray)
box(8.65,5.2,1.9,.75,r'$h_i$'+'：[8]',19,orange)
arrow((2.99,5.2),(3.92,5.2));arrow((6.67,5.2),(7.65,5.2))
txt(11.6,5.2,'进入下一个 site\n仍然只有 8 个分量',17,gray)
arrow((9.65,5.2),(10.1,5.2))
ax.plot([.4,13.6],[4.45,4.45],color='#DAE1E7',lw=1)

txt(.45,3.95,'② 将每步两项合并后，矩阵连乘部分如下',19,ha='left',weight='bold')
y=2.85
circle(.9,y,r'$h_0$',r=.39)
line(1.29,1.82,y,'8')
box(3.0,y,2.3,1.0,r'$I+\frac{u_1}{4}A_1$',20)
line(4.18,4.72,y,'8')
box(5.9,y,2.3,1.0,r'$I+\frac{u_2}{4}A_2$',20)
line(7.08,7.62,y,'8')
txt(8.05,y,'…',28)
line(8.5,9.02,y,'8')
box(10.3,y,2.5,1.0,r'$I+\frac{u_{16}}{4}A_{16}$',20)
line(11.58,13.0,y,'8')
txt(.9,2.07,'辅助边界\n[8]',16,orange)
txt(3,2.07,'[8, 8]',17,gray)
txt(5.9,2.07,'[8, 8]',17,gray)
txt(10.3,2.07,'[8, 8]',17,gray)
txt(12.65,1.97,'连乘结果\n[8]',16,gray)
txt(7,1.2,'实线＝8维指标的收缩；u_i 只是框内的标量参数。没有引入 G 或维数为2的腿。',17,weight='bold')
txt(7,.62,'未触发 epsilon 保护时：连乘结果归一化，再经过读出矩阵，等价于原代码的最终输出。',16,gray)
txt(7,.12,'每个 token 独立走这条链；将 L 个 token 的8维中间结果并排放置，就是 [L, 8]。',16,gray)
fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99)
fig.savefig(root/'mps_original_formula_network.png',dpi=180)
fig.savefig(root/'mps_original_formula_network.svg')
plt.close(fig)
print(root/'mps_original_formula_network.png')
