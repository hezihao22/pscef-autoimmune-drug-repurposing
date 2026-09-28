from pathlib import Path
import os,json,time,requests,threading,concurrent.futures
W=Path(__file__).resolve().parent;items=json.loads((W/'work/entity_items.json').read_text('utf8'));out=W/'work/entity_descriptions.jsonl';ledger=W/'work/entity_description_usage.jsonl';lock=threading.Lock();pending=0.;spent=0.;done={}
if out.exists():done={x['key']:x for x in map(json.loads,out.read_text('utf8').splitlines())}
system='Provide short treatment-relevant biomedical concept descriptions for a retrospective representation-learning study. Describe the named drug, disease, or gene/protein using general biomedical knowledge: pharmacological class/target for drugs, pathophysiology for diseases, molecular function for genes. At most 30 words each. Do not invent facts when the identifier/name is unclear; state that function is unknown. Do not list drug-disease treatment pairs. Return JSON {"items":[{"id":integer,"summary":string}]}. No clinical advice.'
headers={'Authorization':'Bearer '+os.environ['DEEPSEEK_API_KEY']}
def balance(tag):
 r=requests.get('https://api.deepseek.com/user/balance',headers=headers,timeout=30);r.raise_for_status();j=r.json();(W/('work/entity_balance_'+tag+'.json')).write_text(json.dumps(j),encoding='utf8');return float(j['balance_infos'][0]['total_balance'])
ceiling=min(6.,balance('before')-.58);(W/'protocol/entity_description_prompt.json').write_text(json.dumps({'system':system,'model':'deepseek-flash','temperature':0,'thinking':'disabled','max_tokens':1600,'batch_size':12}),encoding='utf8')
def work(batch):
 global pending,spent
 content=json.dumps([{'id':i,'identifier':k,**items[k]} for i,k in enumerate(batch)]);reserve=((len((content+system).encode())+256)*2+1600*8)/1e6
 with lock:
  if spent+pending+reserve>ceiling:return
  pending+=reserve
 saved=[];charge=reserve;entry={'keys':batch,'timestamp':time.time()}
 try:
  r=requests.post('https://api.deepseek.com/chat/completions',headers=headers,json={'model':'deepseek-flash','messages':[{'role':'system','content':system},{'role':'user','content':content}],'temperature':0,'thinking':{'type':'disabled'},'max_tokens':1600,'response_format':{'type':'json_object'}},timeout=150);r.raise_for_status();j=r.json();u=j.get('usage',{});charge=(u.get('prompt_tokens',0)*2+u.get('completion_tokens',0)*8)/1e6 if u else reserve;response=j['choices'][0]['message']['content'];data={int(z['id']):z['summary'] for z in json.loads(response)['items']}
  for i,k in enumerate(batch):
   if isinstance(data.get(i),str) and data[i]:saved.append({'key':k,'summary':data[i],'model':j.get('model'),'request_id':j.get('id')})
  entry.update(status='success',usage=u,response=response)
 except Exception as e:entry.update(status='error',error_type=type(e).__name__)
 entry['conservative_cny']=charge
 with lock:
  spent+=charge;pending-=reserve
  with open(ledger,'a') as f:f.write(json.dumps(entry)+'\n')
  with open(out,'a') as f:
   for x in saved:f.write(json.dumps(x)+'\n');done[x['key']]=x
  if len(done)%240<12:print('described',len(done),'/',len(items),'bound',round(spent,3),flush=True)
keys=[k for k in items if k not in done]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(work,[keys[i:i+12] for i in range(0,len(keys),12)]))
balance('after');print('COMPLETE',len(done),len(items),flush=True)
