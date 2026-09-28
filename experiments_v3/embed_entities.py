from pathlib import Path
import os,sys,json,requests,time,concurrent.futures,threading,numpy as np
W=Path(__file__).resolve().parent;items=json.loads((W/'work/entity_items.json').read_text('utf8'));keys=list(items);mode=sys.argv[1] if len(sys.argv)>1 else 'name';cache=W/('work/entity_'+mode+'_cache');cache.mkdir(exist_ok=True);lock=threading.Lock();done=0
if mode=='description':
 desc={x['key']:x['summary'] for x in map(json.loads,(W/'work/entity_descriptions.jsonl').read_text('utf8').splitlines())};assert set(desc)==set(keys)
texts=[items[k]['name'] if mode=='name' else desc[k] for k in keys]
def work(start):
 global done
 f=cache/(str(start)+'.npz');batch=keys[start:start+32]
 if f.exists():return
 for a in range(5):
  try:
   r=requests.post('https://uni-api.cstcloud.cn/v1/embeddings',headers={'Authorization':'Bearer '+os.environ['EMBEDDING_API_KEY']},json={'model':'qwen3-embedding:8b','input':texts[start:start+32]},timeout=180);r.raise_for_status();j=r.json();arr=np.array([v['embedding'] for v in sorted(j['data'],key=lambda z:z['index'])],dtype='float32');assert arr.shape==(len(batch),4096);np.savez_compressed(f,keys=np.array(batch),embedding=arr)
   with lock:
    done+=len(batch)
    if done%512==0:print(mode,done,len(keys),flush=True)
   return
  except Exception as e:
   if a==4:print('FAILED',mode,start,type(e).__name__,flush=True);return
   time.sleep(2**a)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:list(ex.map(work,range(0,len(keys),32)))
arr=np.concatenate([np.load(cache/(str(s)+'.npz'))['embedding'] for s in range(0,len(keys),32)]);np.savez_compressed(W/('work/entity_'+mode+'_embeddings.npz'),keys=np.array(keys),embedding=arr)
if mode=='description':
 names=np.load(W/'work/entity_name_embeddings.npz');assert list(names['keys'])==keys;np.savez_compressed(W/'work/entity_embeddings.npz',keys=np.array(keys),embedding=np.concatenate([names['embedding'],arr],axis=1))
print('COMPLETE',mode,len(keys),flush=True)
