"""Publication figures: full-universe performance, heterogeneity and calibration.
Python-only exports; points are diseases, support resamples are not independent n.
"""
from pathlib import Path
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
W=Path(__file__).resolve().parent;F=W.parent/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7})
blue='#346A9B';orange='#B66D38';gray='#7F8B96'
contract={'backend':'Python matplotlib','archetype':'schematic-led composite and quantitative grids','conclusion':'Positive-support calibration improves mean early ranking in this exploratory six-disease benchmark, with disease-dependent gains.','evidence':['Figure 1 model design','Figure 3 complete benchmark and disease variation','Figure 4 fusion weights and support-size sensitivity','Figure 2 full candidate coverage'],'export':'6.7 inch width, vector PDF/SVG plus 300 dpi PNG and 600 dpi TIFF; source CSV supplied; no image manipulations','uncertainty':'Six diseases are the units; thirty support samples do not create thirty independent disease cohorts.'}
(F/'figure_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf8')
def save(fig,n):
 for ext in ['png','pdf','svg','tiff']:fig.savefig(F/(n+'.'+ext),dpi=600 if ext=='tiff' else 300,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def box(ax,x,y,w,h,text,color='#EDF2F5',size=9):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',facecolor=color,edgecolor=gray,lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
fig,ax=plt.subplots(figsize=(6.7,3.25));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.015,.65,.22,.27,'Candidate pair\n+ graph facts\nor no-path flag',size=8)
for y,t in [(.76,'Qwen + logistic regression'),(.45,'TF-IDF + logistic regression'),(.14,'DeepSeek plausibility score')]:box(ax,.32,y,.35,.16,t,size=8);ax.annotate('',(.32,y+.08),(.23,.77),arrowprops={'arrowstyle':'->','color':gray})
box(ax,.72,.50,.26,.36,'Rank normalization\n↓\n3 known positive supports\n↓\nExpert weights',size=8)
for y in [.84,.53,.22]:ax.annotate('',(.72,.67),(.67,y),arrowprops={'arrowstyle':'->','color':gray})
box(ax,.73,.10,.24,.20,'Weighted upper-tail fusion\n→ ranked candidates',color='#DCE8F0',size=8);ax.annotate('',(.85,.30),(.85,.50),arrowprops={'arrowstyle':'->','color':blue})
ax.text(.02,.025,'Source diseases train the two regressions. Target query labels never set expert weights.',fontsize=8)
save(fig,'Figure1')
summary=pd.read_csv(W/'results/primary_summary.csv',index_col=0);bd=pd.read_csv(W/'results/primary_by_disease.csv');order=[m for m in ['PSCEF','RRF (same three experts)','Indication frequency','Qwen LR + supports','Qwen literal LR','K-Paths + DeepSeek','LLaDR','TxGNN','TxGNN-Qwen','REDDA','RGCN','DistMult','ComplEx','RotatE'] if m in summary.index]
labels={'RRF (same three experts)':'RRF · same experts','Qwen LR + supports':'Qwen LR + support','Qwen literal LR':'Qwen literal LR','K-Paths + DeepSeek':'K-Paths + DeepSeek'}
fig,ax=plt.subplots(figsize=(6.7,4.0));yy=np.arange(len(order));ax.barh(yy,[summary.loc[m,'AP'] for m in order],color=[blue if m=='PSCEF' else '#CDD6DD' for m in order],height=.64)
for j,m in enumerate(order):
 vals=bd[bd.method==m].AP.to_numpy();ax.scatter(vals,np.repeat(j,len(vals)),s=17,facecolors='white',edgecolors=orange,lw=.7,zorder=3)
ax.set_yticks(yy,[labels.get(m,m) for m in order]);ax.invert_yaxis();ax.set(xlabel='Average precision',xlim=(0,1));fig.tight_layout();save(fig,'Figure3')
weights=pd.read_csv(W/'results/fusion_weights.csv').groupby('disease')[['Qwen','TF-IDF','DeepSeek']].mean();abbr=['T1D','MS','AS','SLE','RA','PsA'];fig,axs=plt.subplots(1,2,figsize=(6.7,3),gridspec_kw={'width_ratios':[1,1.1]})
im=axs[0].imshow(weights.to_numpy(),cmap='Blues',vmin=0,vmax=1,aspect='auto');axs[0].set_xticks(range(3),['Qwen','TF-IDF','DeepSeek']);axs[0].set_yticks(range(6),abbr)
for i in range(6):
 for j in range(3):axs[0].text(j,i,f'{weights.iloc[i,j]:.2f}',ha='center',va='center',color='white' if weights.iloc[i,j]>.6 else 'black',fontsize=8)
axs[0].set_title('a  Mean calibrated weights',loc='left',fontweight='bold',fontsize=9)
s=pd.read_csv(W/'results/support_size_metrics.csv');v=s.groupby(['method','support_size']).AP.mean()
for m,c in [('PSCEF',blue),('RRF',gray),('Equal evidence fusion',orange),('Qwen literal LR','#739A83')]:axs[1].plot([1,3,5],[v.loc[m,k] for k in [1,3,5]],'o-',label=m,color=c,lw=1.2,ms=3)
axs[1].set(xlabel='Known positive supports',ylabel='Macro average precision',xticks=[1,3,5],ylim=(0,1));axs[1].legend(fontsize=7,loc='lower right');axs[1].set_title('b  Support-size sensitivity',loc='left',fontweight='bold',fontsize=9);fig.tight_layout(w_pad=2);save(fig,'Figure4')
cov=json.loads((W.parent/'data/processed/benchmark_coverage.json').read_text('utf8'));ds=json.loads((W/'work/graph.json').read_text('utf8'))['diseases'];cov=[next(x for x in cov if x['disease']==d) for d in ds];fig,axs=plt.subplots(1,2,figsize=(6.7,2.8));y=np.arange(6)
axs[0].barh(y,[x['aug_candidates'] for x in cov],color=blue,label='Two-edge path');axs[0].barh(y,[4477-x['aug_candidates'] for x in cov],left=[x['aug_candidates'] for x in cov],color='#E0E5E9',label='No two-edge path');axs[0].set_yticks(y,abbr);axs[0].invert_yaxis();axs[0].set(xlabel='Candidates per disease',xlim=(0,4477));axs[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2)
axs[1].barh(y,[x['aug_positive'] for x in cov],color=blue);axs[1].barh(y,[x['curated_positive_total']-x['aug_positive'] for x in cov],left=[x['aug_positive'] for x in cov],color='#E0E5E9');axs[1].set_yticks(y,abbr);axs[1].invert_yaxis();axs[1].set(xlabel='Mapped known positives');axs[0].set_title('a  Entire candidate universe',loc='left',fontsize=9,fontweight='bold');axs[1].set_title('b  Known-positive coverage',loc='left',fontsize=9,fontweight='bold');fig.tight_layout();save(fig,'Figure2')
fig,ax=plt.subplots(figsize=(3.25,1.75));ax.axis('off');box(ax,.02,.61,.29,.27,'Graph facts\n+ text',size=8);box(ax,.37,.61,.29,.27,'3 complementary\nranking experts',size=7);box(ax,.73,.61,.24,.27,'3 known\npositives',size=8);box(ax,.23,.13,.56,.26,'Positive-support calibration\n→ candidate prioritization',size=8)
for x in [.16,.52,.85]:ax.annotate('',(.51,.4),(x,.60),arrowprops={'arrowstyle':'->','color':gray,'lw':.8})
fig.subplots_adjust(0,0,1,1)
for ext in ['png','tiff','pdf']:fig.savefig(F/('TOC.'+ext),dpi=300,facecolor='white')
plt.close(fig);print('FIGURES COMPLETE',flush=True)
