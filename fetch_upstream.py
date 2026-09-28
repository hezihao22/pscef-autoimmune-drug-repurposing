from pathlib import Path
import json,requests
W=Path(__file__).resolve().parent/'experiments_v3'
for repo in json.loads((W/'protocol/upstream_sources.json').read_text('utf8')):
 if repo['repository'].endswith('KGIA'):continue
 name=repo['repository'].split('/')[-1];commit=repo['commit'];url='https://api.github.com/repos/'+repo['repository']+'/git/trees/'+commit+'?recursive=1'
 response=requests.get(url,timeout=60);response.raise_for_status()
 for entry in response.json()['tree']:
  path=entry['path']
  if entry['type']!='blob' or (not path.endswith(('.py','.md','.txt','.yaml','.yml','.toml','.cfg')) and 'LICENSE' not in path):continue
  if entry.get('size',0)>2000000:continue
  target=W/'vendor'/name/path;target.parent.mkdir(parents=True,exist_ok=True)
  raw=requests.get('https://raw.githubusercontent.com/'+repo['repository']+'/'+commit+'/'+path,timeout=60);raw.raise_for_status();target.write_bytes(raw.content)
 print('Fetched',name,commit)
