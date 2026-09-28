"""Published KGE scoring functions from LLaDR author implementation.
Train from scratch on the common graph with source validation, typed corruption.
"""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='3'
import sys,json,time,copy
import numpy as np,torch
from sklearn.metrics import average_precision_score
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W/'vendor/LLaDR/train'));from model_common import LLaDR
torch.set_num_threads(3);device='cuda';g=json.loads((W/'work/graph.json').read_text('utf8'));drugs=g['drugs'];diseases=g['diseases'];entities=drugs+diseases+g['genes'];ei={v:i for i,v in enumerate(entities)};di={v:i for i,v in enumerate(drugs)};dd={v:i for i,v in enumerate(diseases)}
rels=sorted({r for _,r,_ in g['edges']})+['indication'];ri={v:i for i,v in enumerate(rels)};molecular=np.array([(ei[h],ri[r],ei[t]) for h,r,t in g['edges']],dtype='int64')
Y=np.zeros((len(drugs),6),dtype='float32')
for d,pp in g['positives'].items():
 for p in pp:Y[di[p],dd[d]]=1
counts={'drug':(0,len(drugs)),'disease':(len(drugs),len(drugs)+6),'gene':(len(drugs)+6,len(entities))}
def split(held,seed):
 rng=np.random.default_rng(seed);sp=rng.choice(np.flatnonzero(Y[:,held]),3,replace=False);sn=rng.choice(np.flatnonzero(Y[:,held]==0),3,replace=False);known=np.argwhere(Y>0);source=known[known[:,1]!=held];rng=np.random.default_rng(seed+9000);valid=[]
 for d in range(6):
  if d!=held:
   pp=source[source[:,1]==d];valid.extend(pp[rng.choice(len(pp),max(1,int(len(pp)*.2)),replace=False)].tolist())
 return sp,sn,source,np.array(valid)
def create(name,seed):
 torch.manual_seed(seed);base='DistMult' if name=='LLaDR' else name;dim=1024 if name=='LLaDR' else 200
 text=torch.tensor(np.load(W/'work/entity_embeddings.npz')['embedding'],device=device) if name=='LLaDR' else torch.zeros(len(entities),dim*4,device=device)
 return LLaDR(base,len(entities),len(rels),dim,12.,base in ['ComplEx','RotatE'],base=='ComplEx',entity_text_embeddings=text,rho=.6 if name=='LLaDR' else 1.,zeta_2=.5 if name=='LLaDR' else 0.,zeta_3=1.8 if name=='LLaDR' else 1.,text_dist_constraint='true' if name=='LLaDR' else 'false').to(device)
def scores(model):
 triples=np.array([(i,ri['indication'],len(drugs)+d) for i in range(len(drugs)) for d in range(6)]);ss=[]
 with torch.no_grad():
  for start in range(0,len(triples),2048):ss.append(model(torch.tensor(triples[start:start+2048],device=device)).flatten().cpu().numpy())
 return np.concatenate(ss).reshape(len(drugs),6)
def train(model,pos,steps,seed,valid=None,held=None,target_unlabelled=None):
 rng=np.random.default_rng(seed);tt=np.array([(int(a),ri['indication'],len(drugs)+int(b)) for a,b in pos]);triples=np.concatenate([molecular,tt]);N=len(entities);R=len(rels);truehash=np.unique((triples[:,0]*R+triples[:,1])*N+triples[:,2]);opt=torch.optim.Adam(model.parameters(),lr=.001 if model.rho==1 else .0001);best=-1;beststep=0;beststate=None;trace=[]
 for step in range(steps):
  # Balanced relation families retain scarce indications rather than allowing PPI to dominate.
  mp=molecular[rng.integers(len(molecular),size=256)];ip=tt[rng.integers(len(tt),size=256)];batch=np.concatenate([mp,ip]);head=bool(step%2);neg=np.empty((len(batch),32),dtype='int64')
  v=batch[:,0 if head else 2];low=np.where(v<len(drugs),0,np.where(v<len(drugs)+6,len(drugs),len(drugs)+6))[:,None];high=np.where(v<len(drugs),len(drugs),np.where(v<len(drugs)+6,len(drugs)+6,len(entities)))[:,None];neg=rng.integers(low,high,size=(len(batch),32))
  target_rows=(batch[:,1]==ri['indication'])&(batch[:,2]==len(drugs)+held)
  if head and target_rows.any():neg[target_rows]=rng.choice(target_unlabelled,size=(target_rows.sum(),32))
  for attempt in range(16):
   hh=(neg*R+batch[:,1,None])*N+batch[:,2,None] if head else (batch[:,0,None]*R+batch[:,1,None])*N+neg
   poshash=np.searchsorted(truehash,hh);bad=truehash[np.minimum(poshash,len(truehash)-1)]==hh
   # Preserve the identical few-shot information boundary: no query pair may
   # become an indication training negative through entity corruption.
   cand_h=neg if head else np.broadcast_to(batch[:,0,None],neg.shape);cand_t=np.broadcast_to(batch[:,2,None],neg.shape) if head else neg
   forbidden=(batch[:,1,None]==ri['indication'])&(cand_t==len(drugs)+held)&(~np.isin(cand_h,target_unlabelled));bad|=forbidden
   if not bad.any():break
   if attempt<15:
    proposal=rng.integers(low,high,size=neg.shape)
    if head and target_rows.any():proposal[target_rows]=rng.choice(target_unlabelled,size=(target_rows.sum(),32))
    neg[bad]=proposal[bad]
  good=torch.tensor(~bad,device=device)
  p=torch.tensor(batch,device=device);n=torch.tensor(neg,device=device);model.train();opt.zero_grad();td,ns=model((p,n),'head-batch' if head else 'tail-batch');ps=model(p)
  # Author self-adversarial negative loss and text-distance regularization.
  weights=torch.softmax(ns.masked_fill(~good,-1e9),1).detach()*good;weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-9)
  nl=(weights*torch.nn.functional.logsigmoid(-ns)).sum(1).mean();pl=torch.nn.functional.logsigmoid(ps).mean();loss=model.zeta_3*(-pl-nl)/2+model.zeta_2*td.mean();loss.backward();opt.step()
  if valid is not None and ((step+1)%100==0 or step==steps-1):
   model.eval();ss=scores(model);vals=[]
   for d in range(6):
    if d==held:continue
    ix=np.flatnonzero(Y[:,d]==0).tolist()+valid[valid[:,1]==d,0].tolist();vals.append(average_precision_score(Y[ix,d],ss[ix,d]))
   ap=float(np.mean(vals));trace.append({'step':step+1,'loss':float(loss),'source_validation_AP':ap})
   if ap>best:best=ap;beststep=step+1
   if step+1>=400 and step+1-beststep>=300:break
 return beststep,trace
def run(name,smoke=False):
 for held in range(1 if smoke else 6):
  for seed in ([100] if smoke else [100,101,102]):
   out=W/'results'/f'{name}_{held}_{seed}.npz'
   if out.exists():continue
   t=time.time();sp,sn,source,valid=split(held,seed);vs=set(map(tuple,valid));target=np.column_stack([sp,np.repeat(held,3)]);trainpos=np.array([v for v in source if tuple(v) not in vs]+list(target));ts={100:11,101:29,102:47}[seed];model=create(name,ts)
   best,trace=train(model,trainpos,3 if smoke else 1500,ts,valid,held,sn)
   del model;torch.cuda.empty_cache();model=create(name,ts);train(model,np.concatenate([source,target]),max(best,1),ts,held=held,target_unlabelled=sn);model.eval();ss=scores(model)[:,held]
   if not smoke:
    np.savez_compressed(out,drugs=np.array(drugs),scores=ss,positive_support=sp,unlabelled_support=sn);(W/'results'/f'{name}_{held}_{seed}_trace.json').write_text(json.dumps({'selected_step':best,'trace':trace,'seconds':time.time()-t}),encoding='utf8')
   print(name,held,seed,'step',best,'seconds',round(time.time()-t,1),flush=True);del model;torch.cuda.empty_cache()
if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'DistMult','--smoke' in sys.argv)
