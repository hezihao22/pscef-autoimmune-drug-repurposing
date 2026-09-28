from pathlib import Path
import json,hashlib,collections
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from scipy.stats import spearmanr
W=Path(__file__).resolve().parent;P=W.parent/'data/processed';g=json.loads((W/'work/graph.json').read_text('utf8'));items=json.loads((W/'work/full_items.json').read_text('utf8'));report={}
assert len(items)==26862 and sum(x['label'] for x in items.values())==144
assert len(g['drugs'])==4477 and len(g['genes'])==4406 and len(g['edges'])==58085
assert all(r!='indication' and 'treat' not in r.lower() for _,r,_ in g['edges'])
z=np.load(W/'work/full_embeddings.npz');old=np.load(P/'literal_embeddings.npz');ki={k:i for i,k in enumerate(z['keys'])};arr=z['embedding'];olda=old['embedding'];assert arr.shape==(26862,4096);assert np.array_equal(arr[[ki[k] for k in old['keys']]],olda);report['literal_cache_reuse_verified']=len(old['keys']);del z,old,arr,olda
e=np.load(W/'work/entity_embeddings.npz');assert list(e['keys'])==g['drugs']+g['diseases']+g['genes'];assert e['embedding'].shape==(8889,8192);del e
split=json.loads((W/'results/support_splits.json').read_text('utf8'));lookup={}
for r in split:
 p=set(r['positive_support']);n=set(r['unlabelled_support']);assert len(p)==len(n)==3 and not p&n;assert all(items[k]['label']==1 for k in p);assert all(items[k]['label']==0 for k in n);lookup[(r['disease'],r['seed'])]=(p,n)
assert len(lookup)==180;report['support_splits_checked']=180
pred=pd.read_csv(W/'results/predictions_3seeds.csv.gz',float_precision='round_trip');calc=[]
for (method,d,seed),df in pred.groupby(['method','disease','seed']):
 assert len(df)==4471 and df.key.nunique()==4471;assert np.isfinite(df.score).all();p,n=lookup[(d,seed)];assert not set(df.key)&(p|n);yy=df.label.to_numpy();ss=df.score.to_numpy();kk=df.key.to_numpy();pos=np.flatnonzero(yy);un=np.flatnonzero(yy==0);ranks=np.array([1+np.sum((ss[un]>ss[i])|((ss[un]==ss[i])&(kk[un]<kk[i]))) for i in pos]);order=np.lexsort((kk,-ss));calc.append({'method':method,'disease':d,'seed':seed,'AP':average_precision_score(yy,ss),'AUROC':roc_auc_score(yy,ss),'MRR':np.mean(1/ranks),'Recall10':yy[order[:10]].sum()/yy.sum()})
cc=pd.DataFrame(calc);saved=pd.read_csv(W/'results/all_metrics.csv',float_precision='round_trip');saved=saved[(saved.subset=='All drugs')&(saved.seed<=102)];joined=cc.merge(saved,on=['method','disease','seed'],suffixes=('_checked','_saved'));assert len(joined)==len(saved)
for k in ['AP','AUROC','MRR','Recall10']:assert np.allclose(joined[k+'_checked'],joined[k+'_saved'],atol=1e-12)
report['metric_rows_recomputed']=len(joined);report['methods']=sorted(cc.method.unique().tolist());report['prediction_rows']=len(pred)
for model in ['TxGNN','TxGNN-Qwen','REDDA','RGCN','DistMult','ComplEx','RotatE','LLaDR']:
 assert len(cc[cc.method==model])==18,model
 report[model+'_fits']=18
report['sha256']={str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [W/'protocol/frozen_protocol.json',W/'results/primary_summary.csv',W/'results/predictions_3seeds.csv.gz']};report['status']='passed'
(W/'results/verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
# Same-query SLE ranks with clinical interpretations kept separate from source labels.
dn=json.loads((P/'drug_names.json').read_text('utf8'));case=[];d='Disease::MONDO:0007915';canonical={'DB01611':'Hydroxychloroquine','DB00091':'Cyclosporine','DB08879':'Belimumab','DB00608':'Chloroquine','DB06186':'Ipilimumab','DB00139':'Succinic acid','DB09053':'Ibrutinib','DB06292':'Dapagliflozin'};selected=list(canonical)
for drug in selected:
 key='DrugBank::'+drug+'|||'+d
 if key not in items:continue
 row={'drug':drug,'drug_name':canonical[drug],'source_label':items[key]['label']}
 for method in ['PSCEF','RRF (same three experts)','Qwen LR + supports','K-Paths + DeepSeek','TxGNN','LLaDR']:
  df=pred[(pred.method==method)&(pred.disease==d)&(pred.seed==100)].sort_values(['score','key'],ascending=[False,True]);rr=dict(zip(df.key,range(1,len(df)+1)));row[method]=rr.get(key,'Support')
 case.append(row)
pd.DataFrame(case).to_csv(W/'results/sle_case_v3.csv',index=False)
