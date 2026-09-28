"""Check the actual KGE corruption sampler and PSCEF invariance properties."""
from pathlib import Path
import ast, json, hashlib
import numpy as np
from pscef import fuse_scores
import kge_bench as kb

W=Path(__file__).resolve().parent
source=(W/'kge_bench.py').read_text('utf8')
tree=ast.parse(source)
train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
loop=next(n for n in train.body if isinstance(n,ast.For))
sampler=[]
for node in loop.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='good' for t in node.targets):break
    sampler.append(node)
compiled=compile(ast.Module(body=sampler,type_ignores=[]),'actual_corruption_sampler','exec')
checked=0;masked=0
for held in range(6):
    for seed in [100,101,102]:
        sp,sn,sourcepos,valid=kb.split(held,seed)
        pos=np.concatenate([sourcepos,np.column_stack([sp,np.repeat(held,3)])])
        tt=np.array([(int(a),kb.ri['indication'],len(kb.drugs)+int(b)) for a,b in pos])
        triples=np.concatenate([kb.molecular,tt]);N=len(kb.entities);R=len(kb.rels)
        env=dict(vars(kb),rng=np.random.default_rng(seed),tt=tt,N=N,R=R,
                 truehash=np.unique((triples[:,0]*R+triples[:,1])*N+triples[:,2]),
                 held=held,target_unlabelled=sn)
        for step in range(12):
            env['step']=step;exec(compiled,env)
            good=~env['bad'];hh=env['hh'];batch=env['batch']
            assert not np.isin(hh[good],env['truehash']).any()
            indication=np.broadcast_to(batch[:,1,None]==kb.ri['indication'],good.shape)
            target=indication&(env['cand_t']==len(kb.drugs)+held)&good
            assert np.isin(env['cand_h'][target],sn).all()
            checked+=int(good.sum());masked+=int((~good).sum())

rng=np.random.default_rng(413);scores=rng.normal(size=(200,3));sp=np.array([4,12,78])
a=fuse_scores(scores,sp);b=fuse_scores(np.exp(scores),sp)
assert np.array_equal(a['scores'],b['scores'])
assert np.isclose(a['weights'].sum(),1)
ties=fuse_scores(np.ones((30,3)),[2,5,8])
assert np.allclose(ties['weights'],1/3) and np.ptp(ties['scores'])==0
for supports,kwargs in [([1.5],{}),([1,1],{}),([-1],{}),([0],{'temperature':float('nan')}),([0],{'weight_scale':-1})]:
    try:fuse_scores(scores,supports,**kwargs)
    except ValueError:pass
    else:raise AssertionError('Invalid public-core input accepted')
report={'status':'passed','sampler_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'episodes':18,'batches_per_episode':12,'accepted_corruptions_checked':checked,
        'masked_corruptions':masked,'target_query_negatives':0,'training_positive_corruptions':0,
        'core_monotone_invariance':True,'core_equal_tie_symmetry':True,
        'scope':'Unit audit executes the actual sampler AST; not a trace of every training update.'}
(W/'protocol/sampler_and_core_checks.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
