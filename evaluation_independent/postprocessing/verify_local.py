"""CPU-only local verification of transferred server evaluation artifacts."""
import datetime,hashlib,json,math,sys
from pathlib import Path
from transformers import AutoTokenizer
r=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
tokenizer_path=Path(sys.argv[2]) if len(sys.argv)>2 else r.parent/'model'
read=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
protocol=json.load(open(r/'protocol.json'));protocolsha=sha(r/'protocol.json')
for name,h in protocol['code'].items():assert sha(r/name)==h,name
for name,h in protocol['inputs'].items():assert sha(r/name)==h,name
summary=json.load(open(r/'summary.json'));delivery=json.load(open(r/'artifacts/delivery_verification.json'))
assert summary['protocol_sha256']==protocolsha
assert sha(r/'summary.json')==delivery['summary_sha256']
assert sha(r/'REPORT.md')==delivery['report_sha256']
tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
total=0;hashes={}
for path in sorted((r/'predictions').glob('*.jsonl')):
 data=read(path);expected=64 if path.stem.endswith('_text') else 1500
 assert len(data)==expected and len({x['id'] for x in data})==expected
 assert sha(path)==delivery['prediction_hashes'][path.name]
 decoded=tok.batch_decode([x['generated_ids'] for x in data],skip_special_tokens=True)
 for x,y in zip(data,decoded):
  assert y==x['prediction']
  assert len(x['generated_ids'])==x['tokens'] and 0<len(x['generated_ids'])<=16
  assert all(0<=v<len(tok) for v in x['generated_ids'])
  assert x['hit_length_limit']==(len(x['generated_ids'])==16 and x['generated_ids'][-1]!=tok.eos_token_id)
  assert x['protocol_sha256']==protocolsha
 total+=len(data);hashes[path.name]=sha(path)
assert total==18256 and len(hashes)==16
status=json.load(open(r/'status.json'));assert status['status']=='complete' and all(v==0 for v in status['exits'].values())
out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'verified','execution':'CPU tokenizer only on local Mac; no model inference','rows':total,'files':len(hashes),'protocol_sha256':protocolsha,'checks':['every prediction ID/count and frozen input/code hash','every generated token sequence decoded independently with local pinned tokenizer','length cap and token ID range','server delivery and transferred report/summary/prediction SHA256','all worker exits zero'],'prediction_hashes':hashes}
(r/'artifacts/local_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('LOCAL_VERIFIED',total,'predictions;',len(hashes),'files')
