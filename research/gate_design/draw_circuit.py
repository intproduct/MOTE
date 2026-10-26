from html import escape
from pathlib import Path
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1700" height="710" viewBox="0 0 1700 710">','<rect width="1700" height="710" fill="white"/>','<g font-family="Arial, sans-serif" fill="#17283b">']
def text(x,y,s,size=21,anchor='start',color='#17283b',weight='normal'):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{escape(s)}</text>')
def rect(x,y,w,h,fill,stroke='none',rx=9):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
def line(x1,y1,x2,y2):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#8b9bad" stroke-width="2"/>')
def box(x,y,label,color='#e5efff',stroke='#4778b7',w=154):
    rect(x-w/2,y-26,w,52,color,stroke)
    text(x,y+7,label,21,'middle')
text(40,52,'Compact Born gate: four qubits produce sixteen expert probabilities',31,weight='bold')
text(40,90,'Proposed research circuit for E = 16, depth = 2. This is not the existing blockwise QuantumGate.',21,color='#536577')
rect(28,117,1644,432,'#f6f8fb')
for x,s in [(265,'Input layer 1'),(485,'Pair gates'),(725,'Input layer 2'),(945,'Pair gate'),(1155,'Readout axis'),(1320,'Z basis')]:
    text(x,166,s,20,'middle','#536577')
ys=[220,310,400,490]
for i,y in enumerate(ys):
    text(53,y+7,f'q{i}: |0⟩',22)
    line(151,y,1380,y)
    box(265,y,'Ry → Rz')
    box(725,y,'Ry → Rz')
    box(1155,y,'Ry(β)','#fff0dd','#b78a3a')
    box(1320,y,'MZ','#e1f3ec','#3c816c',86)
for start in (0,2):
    rect(445,ys[start]-26,80,142,'#e4e0fa','#8470b4')
    text(485,(ys[start]+ys[start+1])/2+7,f'G{start}{start+1}',23,'middle')
rect(905,ys[1]-26,80,142,'#e4e0fa','#8470b4')
text(945,(ys[1]+ys[2])/2+7,'G12',23,'middle')
rect(1410,194,230,330,'#ffffff','#b8c8d8')
text(1525,235,'Joint distribution',21,'middle',weight='bold')
for y,s in [(283,'0000 → expert 0'),(326,'0001 → expert 1'),(369,'0010 → expert 2'),(411,'⋮'),(456,'1111 → expert 15')]:
    text(1525,y,s,21,'middle')
text(1525,498,'p(e) = |amplitude(e)|²',18,'middle','#536577')
text(43,592,'Each Ry / Rz receives its own token-dependent angle at each layer.',23)
text(43,630,'Gij contains two controlled-Ry rotations in opposite directions, with separate trainable angles.',22,color='#536577')
text(43,675,'16 joint probabilities → log probabilities → existing temperature / TopK / capacity / auxiliary loss',24,weight='bold')
parts.extend(['</g>','</svg>'])
Path(__file__).with_name('compact_born.svg').write_text('\n'.join(parts))
