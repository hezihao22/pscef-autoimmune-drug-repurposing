from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
F=Path(__file__).resolve().parents[1]/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(6.65,4.1));fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,10),ylim=(0,6.15));ax.axis('off')
def box(x,y,w,h,text,bold=False):
 ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='black',lw=.85));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontweight='bold' if bold else 'normal',linespacing=1.3)
def line(pts,arrow=True,dashed=False):
 for a,b in zip(pts[:-2],pts[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],c='black',lw=.8,ls='--' if dashed else '-')
 a,b=pts[-2:];ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>' if arrow else '-','lw':.8,'color':'black','linestyle':'--' if dashed else '-' ,'shrinkA':0,'shrinkB':0})
box(.3,5.12,7.0,.66,'Drug–disease candidate universe (4,477 drugs per disease)\nLiteral graph evidence; candidates without paths retained',True)
for x,label in [(.3,'Semantic expert\nQwen embeddings\nSource-disease regression'),(2.7,'Lexical expert\nTF-IDF features\nSource-disease regression'),(5.1,'Knowledge expert\nDeepSeek plausibility')]:
 box(x,3.63,2.2,.85,label)
 line([(x+1.1,5.12),(x+1.1,4.48)])
box(.3,2.35,7.0,.63,'Within-expert percentile normalization\nPreserves ranking; aligns heterogeneous score scales')
for x in [1.4,3.8,6.2]:line([(x,3.63),(x,2.98)])
box(7.7,3.63,2.0,.85,'Three verified\npositive supports',True)
box(7.7,2.35,2.0,.63,'Positive-rank evidence\nSoftmax expert weights')
line([(8.7,3.63),(8.7,2.98)])
line([(7.3,2.665),(7.7,2.665)])
box(2.7,1.02,4.6,.68,'Weighted upper-tail fusion\nDisease-specific candidate ranking',True)
line([(3.8,2.35),(3.8,1.70)])
line([(8.7,2.35),(8.7,1.36),(7.3,1.36)])
ax.text(.32,1.37,'Fixed fusion rule\nNo query-label input',ha='left',va='center',fontsize=7.5)
box(2.7,.16,4.6,.48,'Research shortlist and evidence inspection')
line([(5,1.02),(5,.64)])
ax.text(8.65,.45,'Held-out query labels:\nevaluation only',ha='center',va='center',fontsize=7)
for ext in ['png','pdf','svg']:fig.savefig(F/f'Figure1.{ext}',dpi=300,facecolor='white')
fig.savefig(F/'Figure1.tiff',dpi=1200,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
print(F)
