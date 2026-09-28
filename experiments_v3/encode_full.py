from pathlib import Path
import os,json,requests,time,concurrent.futures,threading,numpy as np
W=Path(__file__).resolve().parent;ROOT=W.parent;P=ROOT/'data/processed';cache=W/'work/embedding_cache';cache.mkdir(exist_ok=True)
items=json.loads((W/'work/full_items.json').read_text('utf8'));old=np.load(P/'literal_embeddings.npz');existing=dict(zip(old['keys'],old['embedding']));keys=sorted(set(items)-set(existing));lock=threading.Lock();done=0
def work(start):
 global done
 batch=keys[start:start+32];f=cache/(str(start)+'.npz')
 if f.exists():return
 for a in range(5):
  try:
   r=requests.post('https://uni-api.cstcloud.cn/v1/embeddings',headers={'Authorization':'Bearer '+os.environ['EMBEDDING_API_KEY']},json={'model':'qwen3-embedding:8b','input':[items[k]['text'] for k in batch]},timeout=180);r.raise_for_status();j=r.json();arr=np.array([v['embedding'] for v in sorted(j['data'],key=lambda z:z['index'])],dtype='float32');assert arr.shape==(len(batch),4096)
   np.savez_compressed(f,keys=np.array(batch),embedding=arr)
   with lock:
    done+=len(batch)
    with open(W/'work/embedding_usage.jsonl','a') as out:out.write(json.dumps({'start':start,'n':len(batch),'usage':j.get('usage')})+'\n')
    if done%512==0:print('embedded',done,'/',len(keys),flush=True)
   return
  except Exception as e:
   if a==4:print('FAILED',start,type(e).__name__,flush=True);return
   time.sleep(2**a)
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:list(ex.map(work,range(0,len(keys),32)))
for start in range(0,len(keys),32):
 z=np.load(cache/(str(start)+'.npz'));existing.update(zip(z['keys'],z['embedding']))
allkeys=sorted(items);np.savez_compressed(W/'work/full_embeddings.npz',keys=np.array(allkeys),embedding=np.array([existing[k] for k in allkeys]));print('COMPLETE',len(allkeys),flush=True)
