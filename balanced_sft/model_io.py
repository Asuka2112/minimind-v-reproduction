"""Use the same pinned native model and assistant-only supervision in both arms."""
from functools import lru_cache
import os
import sys

from common import SOURCE, BASE, BASE_SHA, OFFICIAL, OFFICIAL_SHA, sha
sys.path.insert(0,str(SOURCE))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

import torch
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig

VISION = SOURCE/'model/siglip2-base-p32-256-ve'
MARKER='<|image_pad|>'
ASK='\n请用中文简短回答，只给出答案。'

def load_model(path, expected=None):
    if expected:
        assert sha(path)==expected
    model=MiniMindVLM(VLMConfig(),vision_model_path=str(VISION))
    assert model.vision_encoder is not None
    check=model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=False)
    assert not check.unexpected_keys
    assert set(check.missing_keys)=={k for k in model.state_dict() if k.startswith('vision_encoder.')}
    tokenizer=AutoTokenizer.from_pretrained(SOURCE/'model',local_files_only=True)
    return model,tokenizer

def freeze(model):
    last=len(model.model.layers)-1
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(('vision_proj.','model.layers.0.',f'model.layers.{last}.')))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==15_931_776

def prompt_ids(tokenizer, q, with_image, short=True):
    content=(MARKER*64 + '\n' if with_image else '')+q+(ASK if short else '')
    raw=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,
                                       add_generation_prompt=True,open_thinking=False)
    ids=tokenizer(raw,add_special_tokens=False).input_ids
    image_id=tokenizer.convert_tokens_to_ids(MARKER)
    assert ids.count(image_id)==(64 if with_image else 0)
    if with_image:
        first=ids.index(image_id)
        assert ids[first:first+64]==[image_id]*64
    return ids

def train_record(tokenizer,q,a):
    ids=prompt_ids(tokenizer,q,True)
    answer=tokenizer(a,add_special_tokens=False).input_ids
    assert answer and tokenizer.eos_token_id not in answer
    inputs=ids+answer+[tokenizer.eos_token_id]
    labels=[-100]*len(ids)+answer+[tokenizer.eos_token_id]
    assert len(inputs)<=192
    assert tokenizer.decode(answer).strip()==a
    assert len(inputs)==len(labels) and all(x==-100 for x in labels[:len(ids)])
    return inputs,labels

def choose_model(label,root,seed=42):
    if label=='base':
        return BASE,BASE_SHA
    if label=='official':
        return OFFICIAL,OFFICIAL_SHA
    return root/'weights'/f'{label}-{seed}.pth',None
