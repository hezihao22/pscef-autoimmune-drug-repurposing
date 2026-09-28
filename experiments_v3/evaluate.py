"""Frozen PSCEF comparison, full-universe and path-coverage strata.
The six disease domains are exploratory; target-query labels used only in metrics.
"""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import json,time,sys,hashlib
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W.parent/'experiments_v2'));from gpu_logistic import gpu_fit_predict
items=json.loads((W/'work/full_items.json').read_text('utf8'));g=json.loads((W/'work/graph.json').read_text('utf8'));keys=np.array(sorted(items));y=np.array([items[k]['label'] for k in keys]);ds=np.array([items[k]['disease'] for k in keys]);diseases=g['diseases'];drugs=g['drugs'];di={k:i for i,k in enumerate(drugs)};groups={d:np.flatnonzero(ds==d) for d in diseases};covered=np.array([bool(items[k]['paths']) for k in keys]);P=W.parent/'data/processed';pathkeys=set(json.loads((P/'benchmark_items.json').read_text('utf8')))
deepdict=json.loads((W/'work/full_deepseek_combined.json').read_text('utf8'));deep=np.array([deepdict[k]['score'] for k in keys]);kdict={r['key']:r['score'] for r in map(json.loads,(W/'work/kpaths_scores.jsonl').read_text('utf8').splitlines())};kscores=np.array([kdict[k] for k in keys]);cache=W/'work/base_cache';cache.mkdir(exist_ok=True)
def pct(v):return (rankdata(v,method='average')-.5)/len(v)
def metric(idx,s):
 yy=y[idx];pos=np.flatnonzero(yy);un=np.flatnonzero(yy==0)
 if not len(pos) or not len(un):return {'AP':float('nan'),'AUROC':float('nan'),'MRR':float('nan'),'Recall10':float('nan')}
 ranks=[1+np.sum((s[un]>s[p])|((s[un]==s[p])&(keys[idx[un]]<keys[idx[p]]))) for p in pos];top=np.lexsort((keys[idx],-s))[:10]
 return {'AP':average_precision_score(yy,s),'AUROC':roc_auc_score(yy,s),'MRR':np.mean(1/np.array(ranks)),'Recall10':yy[top].sum()/yy.sum()}
def fuse(r,sp,columns=None,equal=False):
 ex=np.arange(r.shape[1]) if columns is None else np.array(columns);a=-np.log(np.maximum(1-r[sp][:,ex],1/(2*len(r)))).mean(0);w=np.ones(len(ex))/len(ex) if equal else np.exp(a-a.max())/np.exp(a-a.max()).sum();return np.exp(4*(r[:,ex]-1))@w,w,a
def make_bases():
 while not (W/'work/full_embeddings.npz').exists():time.sleep(10)
 z=np.load(W/'work/full_embeddings.npz');assert list(z['keys'])==list(keys);X=normalize(z['embedding']).astype('float32');del z
 for d in diseases:
  tr=np.flatnonzero(ds!=d);ix=groups[d];f=cache/(d.split(':')[-1]+'.npz')
  if f.exists():continue
  vec=TfidfVectorizer(token_pattern=r'(?u)\b[\w:]+\b',ngram_range=(1,2),max_features=15000);tf=vec.fit_transform([items[k]['text'] for k in keys[tr]]);tq=vec.transform([items[k]['text'] for k in keys[ix]]);lr=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,solver='liblinear');lr.fit(tf,y[tr]);tfs=lr.decision_function(tq)
  raw=gpu_fit_predict(X[tr],y[tr],X[ix],cs=(1,))[:,0];support_scores=[]
  for seed in [100,101,102]:
   rng=np.random.default_rng(seed);sp=rng.choice(np.flatnonzero(y[ix]),3,replace=False);sn=rng.choice(np.flatnonzero(y[ix]==0),3,replace=False);fitix=np.r_[tr,ix[sp],ix[sn]];support_scores.append(gpu_fit_predict(X[fitix],y[fitix],X[ix],cs=(1,))[:,0])
  np.savez_compressed(f,keys=keys[ix],qwen=raw,tfidf=tfs,support_scores=np.array(support_scores));print('BASES',d,flush=True)
 del X
def run():
 make_bases();records=[];preds=[];splits=[];weightrows=[]
 Y=np.array([[int(p in g['positives'][d]) for d in diseases] for p in drugs])
 for held,d in enumerate(diseases):
  ix=groups[d];z=np.load(cache/(d.split(':')[-1]+'.npz'));assert list(z['keys'])==list(keys[ix]);base=np.column_stack([z['qwen'],z['tfidf'],deep[ix]]);ranks=np.column_stack([pct(base[:,j]) for j in range(3)]);frequency=np.delete(Y,held,axis=1).sum(1)
  for seed in range(100,130):
   rng=np.random.default_rng(seed);sp=rng.choice(np.flatnonzero(y[ix]),3,replace=False);sn=rng.choice(np.flatnonzero(y[ix]==0),3,replace=False);q=np.array(sorted(set(range(len(ix)))-set(sp)-set(sn)));ss,w,a=fuse(ranks,sp)
   result={'PSCEF':ss,'Qwen literal LR':base[:,0],'TF-IDF LR':base[:,1],'DeepSeek':base[:,2],'K-Paths + DeepSeek':kscores[ix],'Indication frequency':frequency,'Equal evidence fusion':fuse(ranks,sp,equal=True)[0],'Mean percentile':ranks.mean(1),'RRF (same three experts)':np.sum(1/(60+len(ix)*(1-ranks)+.5),1),'Support-selected expert':base[:,int(a.argmax())]}
   for col,label in [(0,'Qwen'),(1,'TF-IDF'),(2,'DeepSeek')]:result['PSCEF without '+label]=fuse(ranks,sp,[j for j in range(3) if j!=col])[0]
   if seed<=102:
    result['Qwen LR + supports']=z['support_scores'][seed-100]
    for name in ['TxGNN','TxGNN-Qwen','REDDA','RGCN','DistMult','ComplEx','RotatE','LLaDR']:
     f=W/'results'/f'{name}_{held}_{seed}.npz'
     if not f.exists():continue
     zz=np.load(f);assert list(zz['drugs'])==drugs;assert np.array_equal(zz['positive_support'],sp);assert np.array_equal(zz['unlabelled_support'],sn);result[name]=zz['scores']
   splits.append({'disease':d,'seed':seed,'positive_support':keys[ix[sp]].tolist(),'unlabelled_support':keys[ix[sn]].tolist(),'query_count':len(q),'query_positive_count':int(y[ix[q]].sum())})
   weightrows.append({'disease':d,'seed':seed,**{m:float(v) for m,v in zip(['Qwen','TF-IDF','DeepSeek'],w)}})
   for name,s in result.items():
    for subset in ['All drugs','Path-covered','No two-edge path']:
     qi=q if subset=='All drugs' else q[covered[ix[q]]] if subset=='Path-covered' else q[~covered[ix[q]]]
     records.append({'disease':d,'seed':seed,'method':name,'subset':subset,'n_query':len(qi),'n_positive':int(y[ix[qi]].sum()),**metric(ix[qi],s[qi])})
    # Store every scored query for complete independent metric reconstruction.
    if seed<=102:preds.extend({'key':keys[k],'disease':d,'seed':seed,'method':name,'label':int(y[k]),'score':float(v)} for k,v in zip(ix[q],s[q]))
  print('EVALUATED',d,flush=True)
 pd.DataFrame(records).to_csv(W/'results/all_metrics.csv',index=False);pd.DataFrame(preds).to_csv(W/'results/predictions_3seeds.csv.gz',index=False,compression='gzip');pd.DataFrame(weightrows).to_csv(W/'results/fusion_weights.csv',index=False);(W/'results/support_splits.json').write_text(json.dumps(splits,indent=2),encoding='utf8');summarize()
def summarize():
 df=pd.read_csv(W/'results/all_metrics.csv');primary=df[(df.seed<=102)&(df.subset=='All drugs')];bd=primary.groupby(['method','disease'])[['AP','AUROC','MRR','Recall10']].mean();summary=bd.groupby('method').mean().sort_values('AP',ascending=False);summary.to_csv(W/'results/primary_summary.csv');bd.to_csv(W/'results/primary_by_disease.csv');df[df.subset=='All drugs'].groupby(['method','disease'])[['AP','AUROC','MRR','Recall10']].mean().groupby('method').mean().to_csv(W/'results/stability_summary.csv')
 rng=np.random.default_rng(20260927);boot=rng.integers(0,6,size=(20000,6));paired=[]
 for m in summary.index:
  if m=='PSCEF' or len(bd.loc[m])!=6:continue
  for met in ['AP','AUROC','MRR','Recall10']:
   delta=(bd.loc['PSCEF'][met]-bd.loc[m][met]).to_numpy();bs=delta[boot].mean(1);paired.append({'baseline':m,'metric':met,'difference':delta.mean(),'CI_low':np.quantile(bs,.025),'CI_high':np.quantile(bs,.975),'positive_diseases':int((delta>0).sum())})
 pd.DataFrame(paired).to_csv(W/'results/paired_comparisons.csv',index=False);print(summary.to_string(),flush=True)
if __name__=='__main__':run()
