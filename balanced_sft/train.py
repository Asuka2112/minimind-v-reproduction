"""One GPU per arm, deterministic order, and identical update configuration."""
import argparse
from collections import Counter
import hashlib
import math
import os
import random
import time

from PIL import Image
import torch
from torch.nn import functional as F

from common import BASE, BASE_SHA, HERE, dump, load, now, read_rows, sha
from model_io import freeze, load_model, train_record

GLOBAL_BATCH=16
MICRO_BATCH=8
EPOCHS=2
LR=5e-5

def tensor_sha(t):
    h=hashlib.sha256()
    h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def collate(model, tokenizer, prepared):
    width=max(len(x[0]) for x,_,_ in prepared)
    ids=[]
    labels=[]
    photos=[]
    for (token_ids, target),_,p in prepared:
        pad=width-len(token_ids)
        ids.append(token_ids+[tokenizer.pad_token_id]*pad)
        labels.append(target+[-100]*pad)
        with Image.open(p) as im:
            photos.append(im.convert('RGB'))
    pv={k:v.cuda() for k,v in model.processor(images=photos,return_tensors='pt').items()}
    return torch.tensor(ids,device='cuda'),torch.tensor(labels,device='cuda'),pv

def token_ce(model, data):
    ids,labels,pixels=data
    out=model(input_ids=ids,pixel_values=pixels).logits
    return F.cross_entropy(out[:,:-1].float().reshape(-1,out.shape[-1]),labels[:,1:].reshape(-1),
                           ignore_index=-100,reduction='sum')

@torch.inference_mode()
def measure(model,data):
    model.eval()
    loss=correct=answer_count=0
    for batch in data:
        ids,labels,pixels=batch
        out=model(input_ids=ids,pixel_values=pixels).logits[:,:-1].float()
        gold=labels[:,1:]
        mask=gold!=-100
        ok=(out.argmax(-1)==gold)|~mask
        correct+=int(ok.all(dim=1).sum())
        answer_count+=int(mask.sum())
        loss+=float(F.cross_entropy(out.reshape(-1,out.shape[-1]),gold.reshape(-1),ignore_index=-100,reduction='sum'))
    return {'nll':loss/answer_count,'teacher_forced_exact_examples':correct,
            'n':sum(d[0].shape[0] for d in data)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('arm',choices=['smoke','control','targeted'])
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=now()
    root=HERE/'data'
    protocol_path=HERE/'protocol.json'
    protocol=load(protocol_path)
    manifest=load(root/'manifest.json')
    data_path=root/f"{'targeted' if args.arm=='smoke' else args.arm}.jsonl"
    assert sha(data_path)==manifest['files'][data_path.name]
    assert sha(root/'manifest.json')==protocol['data_manifest_sha256']
    model,tok=load_model(BASE,BASE_SHA)
    freeze(model)
    initial={n:tensor_sha(p) for n,p in model.named_parameters()}
    model.cuda()
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([p for _,p in trainable],lr=LR,fused=True)
    actual=read_rows(data_path)
    if args.arm=='smoke':
        actual=[r for t in ('object','color','count','position') for r in [x for x in actual if x['task']==t][:8]]
    assert len(actual)==(32 if args.arm=='smoke' else protocol['training']['examples_per_arm'])
    prepared=[]
    for row in actual:
        image=root/row['image']
        assert sha(image)==row['image_sha256']
        inputs=train_record(tok,row['question'],row['answer'])
        prepared.append((inputs,row['id'],image))
    planned=len(prepared)//GLOBAL_BATCH*EPOCHS
    assert planned<=1000 and len(prepared)%GLOBAL_BATCH==0
    if args.arm=='smoke':
        planned=160
        cache=[collate(model,tok,prepared[i:i+MICRO_BATCH]) for i in range(0,32,MICRO_BATCH)]
        initial_measure=measure(model,cache)
        print('SMOKE_BEFORE',initial_measure,flush=True)
    else:
        assert planned==protocol['training']['updates_per_arm']
    model.train()
    model.vision_encoder.eval()
    step=0
    loss_record=[]
    example_order=[]
    while step<planned:
        order=list(range(len(prepared)))
        if args.arm!='smoke':
            random.Random(args.seed+step//(len(prepared)//GLOBAL_BATCH)).shuffle(order)
        for j in range(0,len(order),GLOBAL_BATCH):
            idx=order[j:j+GLOBAL_BATCH]
            example_order.extend(prepared[i][1] for i in idx)
            if args.arm=='smoke':
                pairs=cache[(j//MICRO_BATCH):(j//MICRO_BATCH+2)]
            else:
                pairs=[collate(model,tok,[prepared[i] for i in idx[k:k+MICRO_BATCH]])
                       for k in range(0,len(idx),MICRO_BATCH)]
            supervised=sum(int((d[1][:,1:]!=-100).sum()) for d in pairs)
            total=0.
            if args.arm=='smoke':
                rate=3e-4
            else:
                rate=LR*(.1+.9*(1+math.cos(math.pi*step/planned))/2)
            for group in optimizer.param_groups:
                group['lr']=rate
            optimizer.zero_grad(set_to_none=True)
            for data in pairs:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=token_ce(model,data)/supervised
                assert torch.isfinite(loss)
                loss.backward()
                total+=float(loss.detach())
            norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0,error_if_nonfinite=True))
            optimizer.step()
            step+=1
            item={'step':step,'loss':total,'grad_norm_before_clip':norm,'lr':rate}
            loss_record.append(item)
            if step==1 or step%20==0 or step==planned:
                print(item,flush=True)
            if step==planned:
                break
    step_states={int(optimizer.state[p]['step'].item()) for _,p in trainable}
    assert step_states=={planned}
    changed=[]
    for name,p in model.named_parameters():
        different=tensor_sha(p)!=initial[name]
        if p.requires_grad and different:
            changed.append(name)
        else:
            assert p.requires_grad or not different,name
    assert changed
    result={'arm':args.arm,'seed':args.seed,'base_sha256':BASE_SHA,'data_sha256':sha(data_path),
            'protocol_sha256':sha(protocol_path),
            'started_at':started,'finished_at':now(),'updates':step,'rows':len(actual),
            'task_counts':dict(Counter(r['task'] for r in actual)),
            'batch':GLOBAL_BATCH,'micro_batch':MICRO_BATCH,'learning_rate':LR if args.arm!='smoke' else 3e-4,
            'freeze_policy':'projector and first/last LM layer only', 'frozen_parameters_unchanged':True,
            'changed_trainable_tensors':changed,'supervision':'assistant answer and EOS only, no image or user labels',
            'input_example_order_sha256':hashlib.sha256(('\n'.join(example_order)).encode()).hexdigest(),
            'losses':loss_record,'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated()}
    if args.arm=='smoke':
        final=measure(model,cache)
        result.update(before=initial_measure,after=final,
                      pass_gate=final['nll']<=.50*initial_measure['nll'] and final['teacher_forced_exact_examples']>=24)
        dump(HERE/'receipts/smoke.json',result)
        assert result['pass_gate'],result
    else:
        assert len(example_order)==len(prepared)*EPOCHS
        state={k:v.detach().half().cpu() for k,v in model.state_dict().items() if not k.startswith('vision_encoder.')}
        out=HERE/'weights'/f'{args.arm}-{args.seed}.pth'
        out.parent.mkdir(exist_ok=True)
        tmp=out.with_suffix('.tmp')
        torch.save(state,tmp)
        os.replace(tmp,out)
        result['checkpoint_sha256']=sha(out)
        dump(HERE/f'receipts/{args.arm}-{args.seed}.json',result)
    print('DONE',args.arm,step,flush=True)

if __name__=='__main__':
    main()
