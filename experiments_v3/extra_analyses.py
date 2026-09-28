from evaluate import *
rows=[]
for d in diseases:
 ix=groups[d];z=np.load(cache/(d.split(':')[-1]+'.npz'));b=np.column_stack([z['qwen'],z['tfidf'],deep[ix]]);rr=np.column_stack([pct(b[:,i]) for i in range(3)])
 for k in [1,3,5]:
  for seed in range(100,130):
   rng=np.random.default_rng(seed);sp=rng.choice(np.flatnonzero(y[ix]),k,replace=False);sn=rng.choice(np.flatnonzero(y[ix]==0),k,replace=False);q=np.array(sorted(set(range(len(ix)))-set(sp)-set(sn)))
   scores={'PSCEF':fuse(rr,sp)[0],'RRF':np.sum(1/(60+len(ix)*(1-rr)+.5),1),'Equal evidence fusion':fuse(rr,sp,equal=True)[0],'Qwen literal LR':b[:,0]}
   for method,s in scores.items():rows.append({'disease':d,'seed':seed,'support_size':k,'method':method,**metric(ix[q],s[q])})
pd.DataFrame(rows).to_csv(W/'results/support_size_metrics.csv',index=False)
# Calibration theorem check: finite-pool null expected tail score; not clinical calibration.
null={str(n):float((-np.log(1-(np.arange(1,n+1)-.5)/n)).mean()) for n in [92,4477]};(W/'results/calibration_null.json').write_text(json.dumps(null),encoding='utf8')
print('SUPPORT SIZE COMPLETE',len(rows),flush=True)
