from pathlib import Path
import json, collections, csv, hashlib
W=Path(__file__).resolve().parent; ROOT=W.parent; P=ROOT/'data/processed'; D=ROOT/'data/raw'
old=json.loads((P/'benchmark_items.json').read_text('utf8')); diseases=sorted({x['disease'] for x in old.values()})
drugs=sorted(k for k,v in json.loads((D/'drug_status_lookup.json').read_text('utf8')).items() if v=='approved')
dn=json.loads((P/'drug_names.json').read_text('utf8')); names=json.loads((P/'disease_names.json').read_text('utf8'))
labels=collections.defaultdict(set)
for x in json.loads((P/'curated_labels.json').read_text('utf8')): labels[x['disease']].add(x['drug'])
items={}
for d in diseases:
 for drug in drugs:
  k=drug+'|||'+d
  items[k]=old.get(k,{'drug':drug,'disease':d,'drug_name':dn.get(drug,drug),'disease_name':names[d],'facts':[],'text':f'Candidate {drug} ({dn.get(drug,drug)}) and disease {d} ({names[d]}). Graph evidence: No allowed two-edge molecular path was retrieved. Absence of a path does not establish absence of a treatment association.','label':int(drug in labels[d]),'static_available':False,'paths':[]})
(W/'work/full_items.json').write_text(json.dumps(items,ensure_ascii=False),encoding='utf8')
mapping=collections.defaultdict(set)
for row in json.loads((P/'exact_disease_map.json').read_text('utf8')):
 if row['mondo'] in diseases:mapping[row['drkg_id']].add(row['mondo'])
edges=set();ppi=[];genes=set();drset=set(drugs)
for line in open(D/'drkg.tsv',encoding='utf8'):
 h,r,t=line.rstrip().split('\t');drug=h.replace('Compound::','DrugBank::')
 if drug in drset and t.startswith('Gene::') and r in ['DRUGBANK::target::Compound:Gene','Hetionet::CbG::Compound:Gene']:
  edges.add((drug,'drug_protein',t));genes.add(t)
 if h in mapping and t.startswith('Gene::') and r=='Hetionet::DaG::Disease:Gene':
  for d in mapping[h]:edges.add((d,'rev_disease_protein',t));genes.add(t)
 if t in mapping and h.startswith('Gene::') and r in ['GNBR::L::Gene:Disease','GNBR::G::Gene:Disease']:
  for d in mapping[t]:edges.add((d,'rev_disease_protein',h));genes.add(h)
 if r=='Hetionet::GiG::Gene:Gene':ppi.append((h,t))
for x in csv.DictReader(open(D/'llm_extracted_labeled_triplets_glm4.csv',encoding='utf8')):
 h,r,t=x['head'],x['relation'],x['tail']
 if h in drset and t.startswith('Gene::') and r=='interacts_with':edges.add((h,'drug_protein',t));genes.add(t)
 if h.startswith('Gene::') and t in diseases and r=='associated_with':edges.add((t,'rev_disease_protein',h));genes.add(h)
 if t.startswith('Gene::') and h in diseases and r=='associated_with':edges.add((h,'rev_disease_protein',t));genes.add(t)
for h,t in ppi:
 if h in genes and t in genes:edges.add((h,'protein_protein',t));edges.add((t,'protein_protein',h))
graph={'drugs':drugs,'diseases':diseases,'genes':sorted(genes),'edges':sorted(edges),'positives':{d:sorted(labels[d]&drset) for d in diseases}}
(W/'work/graph.json').write_text(json.dumps(graph),encoding='utf8')
spec={'status':'Exploratory extension after inspection of six disease domains; not an independent external validation','PSCEF':{'experts':['source-only Qwen C1 balanced LR','source-only word/bigram TFIDF C1 balanced LR','DeepSeek plausibility'],'rank':'average-tie percentile (rank-.5)/N','reliability':'mean positive-support -log(max(1-r,1/(2N)))','weights':'softmax reliability, beta=1','score':'weighted sum exp(4*(r-1))'},'support_seeds':list(range(100,130)),'graph_training_seeds':[11,29,47],'positive_support':3,'unlabelled_support':3,'candidate_sets':['original 4074 path-covered pairs','all 4477 approved-flag drugs per disease'],'target_truth':'only 3 positive supports may enter fitting; all remaining known indications masked','primary_metric':'macro disease average precision','secondary':['AUROC','filtered MRR','Recall@10'],'inference':'paired bootstrap over six diseases, not over correlated support draws; exploratory intervals','selection':'source-disease validation only; no target-query selection','test_blinding_limit':'domains and historical path-covered results already inspected','no_publication_authorized':True,'graph':{'genes':len(genes),'edges':len(edges),'relations':sorted({e[1] for e in edges})},'hashes':{f:hashlib.sha256((W/'work'/f).read_bytes()).hexdigest() for f in ['full_items.json','graph.json']}}
(W/'protocol/frozen_protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
print(json.dumps({'pairs':len(items),'positives':sum(x['label'] for x in items.values()),'graph':spec['graph']}),flush=True)
