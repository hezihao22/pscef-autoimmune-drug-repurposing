"""Exact bounded relation-diverse retrieval for the common molecular graph.
At <=3 hops its drug-disease paths have exactly two possible relation sequences.
Lexicographic ties replace the implementation-dependent Yen-generator tie order.
"""
from pathlib import Path
import json,collections,networkx as nx
W=Path(__file__).resolve().parent;g=json.loads((W/'work/graph.json').read_text('utf8'));items=json.loads((W/'work/full_items.json').read_text('utf8'))
targets=collections.defaultdict(set);dg=collections.defaultdict(set);pp=collections.defaultdict(set);G=nx.Graph();G.add_nodes_from(g['drugs']+g['diseases']+g['genes'])
for h,r,t in g['edges']:
 G.add_edge(h,t,relation=r)
 if r=='drug_protein':targets[h].add(t)
 elif r=='rev_disease_protein':dg[h].add(t)
 else:pp[h].add(t)
component={n:i for i,c in enumerate(nx.connected_components(G)) for n in c}
out={};counts=collections.Counter()
for k,x in sorted(items.items()):
 drug=x['drug'];d=x['disease'];paths=[]
 common=sorted(targets[drug]&dg[d])
 if common:paths.append([(drug,'drug_protein',common[0]),(common[0],'protein_disease',d)])
 for a in sorted(targets[drug]):
  cand=sorted((pp[a]&dg[d])-{a})
  if cand:
   paths.append([(drug,'drug_protein',a),(a,'protein_protein',cand[0]),(cand[0],'protein_disease',d)]);break
 if component[drug]!=component[d]:
  # Original K-Paths fallback: one neighbor per relation at each endpoint.
  for v in [drug,d]:
   seen=set()
   for u in sorted(G.neighbors(v)):
    rel=G[v][u]['relation']
    if rel not in seen:paths.append([(v,rel,u)]);seen.add(rel)
 facts=['; '.join(f'{h} --{r}--> {t}' for h,r,t in path) for path in paths]
 text=f"Candidate {drug} ({x['drug_name']}) and disease {d} ({x['disease_name']}). Relation-diverse molecular graph paths (up to three edges): "+(' | '.join(facts) if facts else 'No path of at most three edges was retrieved.')+' Molecular association does not establish therapeutic benefit.'
 out[k]={'text':text,'paths':paths};counts[len(paths)]+=1
(W/'work/kpaths_items.json').write_text(json.dumps(out),encoding='utf8');print(dict(counts),flush=True)
