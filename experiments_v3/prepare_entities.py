from pathlib import Path
import json,gzip,csv,requests,collections,hashlib
W=Path(__file__).resolve().parent;g=json.loads((W/'work/graph.json').read_text('utf8'));names=json.loads((W.parent/'data/processed/drug_names.json').read_text('utf8'));names.update(json.loads((W.parent/'data/processed/disease_names.json').read_text('utf8')))
url='https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz';f=W/'work/Homo_sapiens.gene_info.gz'
if not f.exists():
 r=requests.get(url,timeout=120);r.raise_for_status();f.write_bytes(r.content)
for row in csv.DictReader(gzip.open(f,'rt',encoding='utf8'),delimiter='\t'):names['Gene::'+row['GeneID']]=row['Symbol']+' ('+row['description']+')'
entities=g['drugs']+g['diseases']+g['genes'];items={k:{'name':names.get(k,k),'kind':'drug' if k.startswith('DrugBank') else 'disease' if k.startswith('Disease') else 'gene/protein'} for k in entities}
(W/'work/entity_items.json').write_text(json.dumps(items),encoding='utf8');(W/'protocol/entity_name_source.json').write_text(json.dumps({'url':url,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'entities':len(items),'mapped':sum(k!=v['name'] for k,v in items.items())}),encoding='utf8');print('ENTITIES',len(items),flush=True)
