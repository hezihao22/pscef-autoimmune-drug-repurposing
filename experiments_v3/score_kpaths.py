from pathlib import Path
import os,json,time,requests,threading,concurrent.futures
W=Path(__file__).resolve().parent;ROOT=W.parent;items=json.loads((W/'work/kpaths_items.json').read_text('utf8'));out=W/'work/kpaths_scores.jsonl';ledger=W/'work/kpaths_usage.jsonl';lock=threading.Lock();pending=0.;spent=0.
done={}
if out.exists():done={r['key']:r for r in map(json.loads,out.read_text('utf8').splitlines())}
system=json.loads((ROOT/'configs/deepseek_baseline_prompt.json').read_text('utf8'))['system'];headers={'Authorization':'Bearer '+os.environ['DEEPSEEK_API_KEY']}
def balance(tag):
 r=requests.get('https://api.deepseek.com/user/balance',headers=headers,timeout=30);r.raise_for_status();j=r.json();(W/('work/kpaths_balance_'+tag+'.json')).write_text(json.dumps(j),encoding='utf8');return float(j['balance_infos'][0]['total_balance'])
ceiling=min(10.,balance('before')-.58)
def work(batch):
 global spent,pending
 content=json.dumps([{'id':i,'evidence':items[k]['text']} for i,k in enumerate(batch)]);reserve=((len((content+system).encode())+256)*2+1024*8)/1e6
 with lock:
  if spent+pending+reserve>ceiling:return
  pending+=reserve
 charge=reserve;entry={'stage':'K-Paths+DeepSeek common backbone','keys':batch,'timestamp':time.time()};saved=[]
 try:
  r=requests.post('https://api.deepseek.com/chat/completions',headers=headers,json={'model':'deepseek-flash','messages':[{'role':'system','content':system},{'role':'user','content':content}],'temperature':0,'thinking':{'type':'disabled'},'max_tokens':1024,'response_format':{'type':'json_object'}},timeout=150);r.raise_for_status();j=r.json();u=j.get('usage',{});charge=(u.get('prompt_tokens',0)*2+u.get('completion_tokens',0)*8)/1e6 if u else reserve;response=j['choices'][0]['message']['content'];parsed={int(z['id']):z['score'] for z in json.loads(response)['items']}
  for i,k in enumerate(batch):
   s=parsed.get(i)
   if isinstance(s,(float,int)) and 0<=s<=100:saved.append({'key':k,'score':s,'model':j.get('model'),'request_id':j.get('id')})
  entry.update(status='success',usage=u,response=response)
 except Exception as e:entry.update(status='error',error_type=type(e).__name__)
 entry['conservative_cny']=charge
 with lock:
  spent+=charge;pending-=reserve
  with open(ledger,'a') as f:f.write(json.dumps(entry)+'\n')
  with open(out,'a') as f:
   for x in saved:f.write(json.dumps(x)+'\n');done[x['key']]=x
  if len(done)%512<16:print('KPaths',len(done),'/',len(items),'bound CNY',round(spent,3),flush=True)
keys=sorted(set(items)-set(done));batches=[keys[i:i+16] for i in range(0,len(keys),16)]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(work,batches))
balance('after');print('COMPLETE',len(done),len(items),flush=True)
