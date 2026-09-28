"""Author-code graph architectures on the frozen common graph; task adapters explicit.
CPU DGL for native Windows. No target-query labels used by fit/checkpoint selection.
"""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['DGLBACKEND']='pytorch'
import json,sys,types,importlib.util,time,copy,collections
import numpy as np,torch,dgl
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
torch.set_num_threads(4)
W=Path(__file__).resolve().parent;gdata=json.loads((W/'work/graph.json').read_text('utf8'))
drugs=gdata['drugs'];diseases=gdata['diseases'];genes=gdata['genes'];di={x:i for i,x in enumerate(drugs)};dd={x:i for i,x in enumerate(diseases)};gi={x:i for i,x in enumerate(genes)}
counts={'drug':len(drugs),'disease':len(diseases),'gene/protein':len(genes)};etype=('drug','indication','disease')
base=collections.defaultdict(lambda:[[],[]])
for h,r,t in gdata['edges']:
 a,b,c=(di,'drug',gi) if r=='drug_protein' else (dd,'disease',gi) if r=='rev_disease_protein' else (gi,'gene/protein',gi)
 et=(b,r,'gene/protein');base[et][0].append(a[h]);base[et][1].append(c[t])
 if b!='gene/protein':
  rev=('gene/protein','rev_drug_protein' if b=='drug' else 'disease_protein',b);base[rev][0].append(c[t]);base[rev][1].append(a[h])
Y=np.zeros((len(drugs),len(diseases)),dtype='float32')
for d,pp in gdata['positives'].items():
 for p in pp:Y[di[p],dd[d]]=1

def supports(held,seed):
 rng=np.random.default_rng(seed);p=rng.choice(np.flatnonzero(Y[:,held]),3,replace=False);n=rng.choice(np.flatnonzero(Y[:,held]==0),3,replace=False);return p,n
def graph(pos,seed,redda=False):
 data={e:(torch.tensor(s),torch.tensor(t)) for e,(s,t) in base.items()}
 data[etype]=(torch.tensor(pos[:,0]),torch.tensor(pos[:,1]));data[('disease','rev_indication','drug')]=(torch.tensor(pos[:,1]),torch.tensor(pos[:,0]))
 if redda:
  ren={'indication':'drug_disease','rev_indication':'disease_drug','rev_drug_protein':'protein_drug','rev_disease_protein':'disease_protein','disease_protein':'protein_disease'}
  data={(a.replace('gene/protein','protein'),ren.get(b,b),c.replace('gene/protein','protein')):v for (a,b,c),v in data.items()}
  cc={k.replace('gene/protein','protein'):v for k,v in counts.items()}
  # Identity within-type edges provide the self-connections required by REDDA subnets.
  for n in ['drug','disease']:
   z=torch.arange(cc[n]);data[(n,n+'_'+n,n)]=(z,z)
 else:cc=counts
 G=dgl.heterograph(data,num_nodes_dict=cc)
 rng=torch.Generator().manual_seed(seed)
 for n in G.ntypes:G.nodes[n].data['inp']=torch.randn(G.num_nodes(n),100,generator=rng)
 if '--text-features' in sys.argv:
  # Additional stronger comparator; label-free entity-name features, same dimension.
  path=W/'work/entity_features100.npz'
  if not path.exists():
   from sklearn.decomposition import PCA
   zz=np.load(W/'work/entity_name_embeddings.npz');assert list(zz['keys'])==drugs+diseases+genes
   arr=PCA(n_components=100,svd_solver='randomized',random_state=20260927).fit_transform(zz['embedding']);arr=arr/(np.linalg.norm(arr,axis=1,keepdims=True)+1e-12);np.savez_compressed(path,embedding=arr)
  feat=np.load(path)['embedding'];parts={'drug':feat[:len(drugs)],'disease':feat[len(drugs):len(drugs)+6],'gene/protein':feat[len(drugs)+6:]}
  for n in G.ntypes:G.nodes[n].data['inp']=torch.tensor(parts['gene/protein' if n=='protein' else n],dtype=torch.float32)
 return G

def load_tx():
 pkg=types.ModuleType('txgnn');pkg.__path__=[str(W/'vendor/TxGNN/txgnn')];sys.modules['txgnn']=pkg
 # Load exact helper function ASTs without the unrelated ontology split toolchain.
 import ast
 up=W/'vendor/TxGNN/txgnn/utils.py';tree=ast.parse(up.read_text('utf8'));wanted={'sim_matrix','exponential','obtain_disease_profile','obtain_protein_random_walk_profile','convert2str'}
 helper=types.ModuleType('txgnn.utils');helper.__dict__.update(torch=torch,np=np,F=F)
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted],type_ignores=[]),str(up),'exec'),helper.__dict__);sys.modules['txgnn.utils']=helper
 path=W/'vendor/TxGNN/txgnn/model.py';src=path.read_text('utf8')
 # Compatibility with single therapeutic relation: skip absent relation types only.
 marker="        self.node_types_dd = ['disease', 'drug']"
 src=src.replace(marker,"        self.etypes_dd = [e for e in self.etypes_dd if e in G.canonical_etypes and G.num_edges(e)>0]\n"+marker)
 mod=types.ModuleType('txgnn.model');mod.__package__='txgnn';exec(compile(src,str(path),'exec'),mod.__dict__);return mod.HeteroRGCN
TX=load_tx()
spec=importlib.util.spec_from_file_location('redda_author',W/'vendor/REDDA/model.py');rm=importlib.util.module_from_spec(spec);spec.loader.exec_module(rm)

def init_model(G,name):
 if name=='REDDA':return rm.Model(G.etypes,G.ntypes,100,64,5,0.)
 model=TX(G,100,100,100,False,name=='TxGNN',3,'protein_profile',None,'rarity',200,'bit',2,'custom',str(W/'work'),.7,'cpu')
 if '--cuda-encoder' in sys.argv:enable_cuda_encoder(model)
 return model
def gpu_layer(layer,G,feat):
 if not hasattr(G,'pscef_cuda_adjacency'):
  G.pscef_cuda_adjacency={}
  for et in G.canonical_etypes:
   src,dst=G.edges(etype=et)
   if not len(src):continue
   degree=torch.bincount(dst,minlength=G.num_nodes(et[2])).clamp_min(1)
   adj=torch.sparse_coo_tensor(torch.stack([dst,src]),1./degree[dst],(G.num_nodes(et[2]),G.num_nodes(et[0]))).coalesce().cuda()
   G.pscef_cuda_adjacency[et]=adj
 out={}
 for (src,rel,dst),adj in G.pscef_cuda_adjacency.items():
  if feat[src].is_cuda:x=feat[src]
  else:
   if not hasattr(G,'pscef_cuda_inputs'):G.pscef_cuda_inputs={}
   if src not in G.pscef_cuda_inputs:G.pscef_cuda_inputs[src]=feat[src].cuda()
   x=G.pscef_cuda_inputs[src]
  wh=layer.weight[rel](x);v=torch.sparse.mm(adj,wh);out[dst]=out[dst]+v if dst in out else v
 return out
def enable_cuda_encoder(model):
 for layer in [model.layer1,model.layer2]:layer.cuda();layer.forward=types.MethodType(gpu_layer,layer)
def predict(model,G,name):
 if name=='REDDA':return model(G,{n:G.nodes[n].data['inp'] for n in G.ntypes})
 h=model(G,None,return_h=True)
 if '--cuda-encoder' in sys.argv:
  return gpu_decoder(model,G,h)
 h={n:v.cpu() for n,v in h.items()}
 # A single all-candidate indication graph calls the original prototype predictor.
 a=torch.arange(len(drugs)).repeat_interleave(len(diseases));b=torch.arange(len(diseases)).repeat(len(drugs))
 Q=dgl.heterograph({etype:(a,b)},num_nodes_dict=counts)
 scores,_=model.pred(Q,G,h,False,'eval')
 return scores[etype].reshape(len(drugs),len(diseases))

def gpu_decoder(model,G,h):
 # Algebraically equivalent to the author predictor on the all-candidate graph.
 dh=h['disease'];dev=dh.device
 if model.pred.proto:
  assert torch.all(G.in_degrees(etype=etype)>0)
  sim=model.pred.sim_all_etypes[etype]
  if G.num_nodes('drug')==int((G.out_degrees(etype=etype)>0).sum()):
   val,idx=torch.topk(sim,model.pred.k+1);val=val[:,1:];idx=idx[:,1:]
  else:val,idx=torch.topk(sim,model.pred.k)
  val=F.normalize(val,p=1,dim=1).to(dev);out=(dh[idx.to(dev)]*val[:,:,None]).sum(1);degree=G.in_degrees(etype=etype).to(dev);coef=(model.pred.exp_lambda*torch.exp(-model.pred.exp_lambda*degree)+.2)[:,None];dh=(1-coef)*dh+coef*out
 rel=model.w_rels[model.pred.rel2idx[etype]].to(dev)
 return (h['drug']*rel)@dh.T

def pretrain(model,G,name,steps=100,held=None,unlabelled=None):
 if name=='REDDA':return
 opt=torch.optim.Adam(model.parameters(),lr=.001)
 for step in range(steps):
  model.train();opt.zero_grad();h=model(G,None,return_h=True);losses=[]
  # Equal relation weighting; typed uniform tail corruption, training graph only.
  for et in G.canonical_etypes:
   s,t=G.edges(etype=et)
   if not len(s):continue
   width=G.num_nodes(et[2]);known=torch.sort(s*width+t).values
   pick=torch.randperm(len(s))[:2048];s=s[pick];t=t[pick];neg=torch.randint(width,t.shape)
   target_src=(s==held) if et[1]=='rev_indication' else torch.zeros_like(s,dtype=torch.bool)
   allowed=torch.tensor(unlabelled)
   if target_src.any():neg[target_src]=allowed[torch.randint(len(allowed),(int(target_src.sum()),))]
   for retry in range(16):
    hh=s*width+neg;index=torch.searchsorted(known,hh).clamp_max(len(known)-1);bad=known[index]==hh
    if et[1]=='indication':bad|=(neg==held)&(~torch.isin(s,allowed))
    if et[1]=='rev_indication':bad|=(s==held)&(~torch.isin(neg,allowed))
    if not bad.any():break
    if retry<15:
     proposal=torch.randint(width,t.shape)
     if target_src.any():proposal[target_src]=allowed[torch.randint(len(allowed),(int(target_src.sum()),))]
     neg[bad]=proposal[bad]
   rel=model.w_rels[model.pred.rel2idx[et]].to(h[et[0]].device);ps=(h[et[0]][s]*rel*h[et[2]][t]).sum(1);ns=(h[et[0]][s]*rel*h[et[2]][neg]).sum(1)
   good=(~bad).to(ns.device);negative=F.softplus(ns[good]).mean() if good.any() else ns.sum()*0
   losses.append((F.softplus(-ps).mean()+negative)/2)
  loss=torch.stack(losses).mean();loss.backward();opt.step()
 # The original TxGNN training strategy resets relation decoder before fine-tuning.
 torch.nn.init.xavier_uniform_(model.w_rels,gain=torch.nn.init.calculate_gain('relu'))

def fit(held,seed,name,smoke=False):
 output_name=name+'-Qwen' if '--text-features' in sys.argv else name
 out=W/'results'/f'{output_name}_{held}_{seed}.npz'
 if out.exists():return
 start=time.time();p,n=supports(held,seed);rng=np.random.default_rng(seed+9000)
 known=np.argwhere(Y>0);source=known[known[:,1]!=held];validation=[]
 for d in range(6):
  if d!=held:
   pp=source[source[:,1]==d];validation.extend(pp[rng.choice(len(pp),max(1,int(len(pp)*.2)),replace=False)].tolist())
 validation=np.array(validation);vs={tuple(x) for x in validation};target=np.column_stack([p,np.repeat(held,len(p))]);trainpos=np.array([x for x in source if tuple(x) not in vs]+list(target))
 # Every source candidate can be used as an unlabeled contrast; target only supports.
 trainmask=np.ones_like(Y,dtype=bool);trainmask[:,held]=False;trainmask[p,held]=True;trainmask[n,held]=True;trainmask[validation[:,0],validation[:,1]]=False
 validmask=np.zeros_like(Y,dtype=bool)
 for d in range(6):
  if d!=held:validmask[Y[:,d]==0,d]=True
 validmask[validation[:,0],validation[:,1]]=True
 training_seed={100:11,101:29,102:47}[seed]
 torch.manual_seed(training_seed);G=graph(trainpos,training_seed,name=='REDDA');model=init_model(G,name)
 pretrain(model,G,name,3 if smoke else 100,held,n)
 opt=torch.optim.Adam(model.parameters(),lr=.001 if name!='REDDA' else .005)
 device='cuda' if '--cuda-encoder' in sys.argv else 'cpu';mask=torch.tensor(trainmask,device=device);yy=torch.tensor(Y,device=device);posweight=(mask.sum()-yy[mask].sum())/yy[mask].sum();best=-1;bestepoch=0;trace=[]
 maxepochs=3 if smoke else 300
 for epoch in range(maxepochs):
  model.train();opt.zero_grad();s=predict(model,G,name);loss=F.binary_cross_entropy_with_logits(s[mask],yy[mask],pos_weight=posweight);loss.backward();opt.step()
  if epoch%10==0 or epoch==maxepochs-1:
   model.eval()
   with torch.no_grad():ss=predict(model,G,name).cpu().numpy()
   val=float(np.mean([average_precision_score(Y[validmask[:,d],d],ss[validmask[:,d],d]) for d in range(6) if d!=held]));trace.append({'epoch':epoch+1,'loss':float(loss),'source_validation_AP':val})
   if val>best:best=val;bestepoch=epoch+1
   if smoke:print(name,'smoke',trace[-1],flush=True)
   if epoch+1>=80 and epoch+1-bestepoch>=60:break
 # Refit with all source positives for the source-validation-selected epoch count.
 allpos=np.concatenate([source,target]);trainmask=np.ones_like(Y,dtype=bool);trainmask[:,held]=False;trainmask[p,held]=True;trainmask[n,held]=True
 torch.manual_seed(training_seed);G=graph(allpos,training_seed,name=='REDDA');model=init_model(G,name);opt=torch.optim.Adam(model.parameters(),lr=.001 if name!='REDDA' else .005);mask=torch.tensor(trainmask,device=device);posweight=(mask.sum()-yy[mask].sum())/yy[mask].sum()
 pretrain(model,G,name,3 if smoke else 100,held,n)
 for epoch in range(bestepoch):
  model.train();opt.zero_grad();s=predict(model,G,name);loss=F.binary_cross_entropy_with_logits(s[mask],yy[mask],pos_weight=posweight);loss.backward();opt.step()
 model.eval()
 with torch.no_grad():scores=predict(model,G,name)[:,held].cpu().numpy()
 if not smoke:
  np.savez_compressed(out,drugs=np.array(drugs),scores=scores,positive_support=p,unlabelled_support=n)
  (W/'results'/f'{output_name}_{held}_{seed}_trace.json').write_text(json.dumps({'trace':trace,'selected_epoch':bestepoch,'source_validation_AP':best,'seconds':time.time()-start,'encoder_backend':'CUDA sparse algebra' if '--cuda-encoder' in sys.argv else 'DGL CPU','target_support_drugs':[drugs[i] for i in p]},indent=2),encoding='utf8')
 print(name,held,seed,'epoch',bestepoch,'seconds',round(time.time()-start,1),flush=True)

if __name__=='__main__':
 name=sys.argv[1] if len(sys.argv)>1 else 'TxGNN';smoke='--smoke' in sys.argv
 for held in range(1 if smoke else 6):
  for seed in ([100] if smoke else [100,101,102]):fit(held,seed,name,smoke)
