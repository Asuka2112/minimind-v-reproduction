"""Generate reproducible three-condition results and a Chinese text regression."""
import argparse
from collections import defaultdict
import time
import os

from PIL import Image
import torch

from common import HERE, dump, dump_rows, load, now, read_rows, sha
from model_io import choose_model, load_model, prompt_ids

BATCH=12

def grouped(ids):
    lengths=defaultdict(list)
    for i,x in enumerate(ids):
        lengths[len(x)].append(i)
    for indices in lengths.values():
        for start in range(0,len(indices),BATCH):
            yield indices[start:start+BATCH]

@torch.inference_mode()
def generate(model,tok,questions,images,limit):
    ids=[prompt_ids(tok,q,images is not None) for q in questions]
    results=[None]*len(ids)
    for indices in grouped(ids):
        x=torch.tensor([ids[i] for i in indices],device='cuda')
        pixels=None
        if images is not None:
            opened=[]
            for i in indices:
                with Image.open(images[i]) as im:
                    opened.append(im.convert('RGB'))
            pixels={k:v.cuda() for k,v in model.processor(images=opened,return_tensors='pt').items()}
        out=model.generate(inputs=x,pixel_values=pixels,max_new_tokens=limit,logits_to_keep=1,do_sample=False,
                           top_k=0,top_p=1.,temperature=1.,repetition_penalty=1.,eos_token_id=tok.eos_token_id)
        for i,tokens in zip(indices,out[:,x.shape[1]:].tolist()):
            if tok.eos_token_id in tokens:
                tokens=tokens[:tokens.index(tok.eos_token_id)+1]
            results[i]=tokens
    assert all(results)
    return results

def prefix(tok,ids,n):
    tokens=ids[:n]
    reached=len(tokens)==n and tokens[-1]!=tok.eos_token_id
    return {'prediction':tok.decode(tokens,skip_special_tokens=True),
            'generated_ids':tokens,'hit_length_cap':reached}

@torch.inference_mode()
def text_cases(model,tok,examples):
    choices=[]
    tasks=[]
    for item in examples:
        q=item['question']+'\n请直接回答。'
        prompt=prompt_ids(tok,q,False,short=False)
        for i,a in enumerate(item['options']):
            target=tok(a,add_special_tokens=False).input_ids
            assert target
            choices.append(prompt+target)
            tasks.append((item['id'],i,len(prompt),target))
    scores=defaultdict(dict)
    for indices in grouped(choices):
        ids=torch.tensor([choices[i] for i in indices],device='cuda')
        logs=torch.log_softmax(model(input_ids=ids).logits.float(),dim=-1)
        for j,i in enumerate(indices):
            key,choice,start,tokens=tasks[i]
            seq=logs[j,start-1:start+len(tokens)-1]
            vals=seq.gather(1,torch.tensor(tokens,device='cuda').view(-1,1))
            scores[key][choice]=float(vals.mean().item())
    output=[]
    for item in examples:
        options=item['options']
        score=scores[item['id']]
        pred=options[max(range(len(options)),key=score.get)]
        output.append({'id':item['id'],'kind':item['kind'],'correct':pred==item['answer'],'prediction':pred,
                       'answer':item['answer'],'mean_logp':score,'gold_nll_per_token':-score[options.index(item['answer'])]})
    return output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model',choices=['base','official','control','targeted'])
    parser.add_argument('split',choices=['dev','test','challenge'])
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=now()
    protocol=load(HERE/'protocol.json')
    proto=sha(HERE/'protocol.json')
    manifest=load(HERE/'data/manifest.json')
    assert sha(HERE/'data/manifest.json')==protocol['data_manifest_sha256']
    path=HERE/f'data/{args.split}.jsonl'
    assert sha(path)==manifest['files'][path.name]
    rows=read_rows(path)
    all_by_id={r['id']:r for r in rows}
    weight,expected=choose_model(args.model,HERE,args.seed)
    if expected is None:
        receipt=load(HERE/f'receipts/{args.model}-{args.seed}.json')
        expected=receipt['checkpoint_sha256']
    model,tok=load_model(weight,expected)
    model.float().eval().cuda()
    for r in rows:
        assert sha(HERE/'data'/r['image'])==r['image_sha256']
        other=all_by_id[r['swapped_id']]
        assert other['question']==r['question'] and other['answer']!=r['answer']
    q=[r['question'] for r in rows]
    ims=[HERE/'data'/r['image'] for r in rows]
    limit=max(protocol['eval']['caps'])
    id_images={r['id']:z for r,z in zip(rows,generate(model,tok,q,ims,limit))}
    print('original inference complete', args.model,args.split,len(rows),flush=True)
    no_image=dict(zip(sorted(set(q)),generate(model,tok,sorted(set(q)),None,limit)))
    # Same question and actual image bytes produce exactly the same decode input.
    for r in rows[:2]:
        other=all_by_id[r['swapped_id']]
        single=generate(model,tok,[r['question']],[HERE/'data'/other['image']],limit)[0]
        assert single[:32]==id_images[other['id']][:32], 'Unstable batch/single prefix'
    out=[]
    for row in rows:
        partner=all_by_id[row['swapped_id']]
        for c in protocol['eval']['conditions']:
            shown={'original':row,'swapped':partner,'none':None}[c]
            ids=no_image[row['question']] if shown is None else id_images[shown['id']]
            for cap in protocol['eval']['caps']:
                out.append({'id':row['id'],'condition':c,'cap':cap,'model':args.model,'split':args.split,
                            'seed':args.seed,'image_sha256':None if shown is None else shown['image_sha256'],
                            'protocol_sha256':proto,**prefix(tok,ids,cap)})
    pred=HERE/f'predictions/{args.split}-{args.model}-{args.seed}.jsonl'
    dump_rows(pred,out)
    text=HERE/'data/text_regression.jsonl'
    assert sha(text)==protocol['text_regression_sha256']
    text_path=HERE/f'predictions/text-{args.model}-{args.seed}.jsonl'
    if args.split=='test':
        dump_rows(text_path,text_cases(model,tok,read_rows(text)))
    dump(HERE/f'receipts/{args.split}-{args.model}-{args.seed}.json',
         {'started_at':started,'finished_at':now(),'model':args.model,'seed':args.seed,'split':args.split,
          'rows':len(out),'prediction_sha256':sha(pred),'checkpoint_sha256':expected,
          'protocol_sha256':proto,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'text_sha256':sha(text_path) if text_path.exists() else None})
    print('DONE',args.model,args.split,len(out),flush=True)

if __name__=='__main__':
    main()
