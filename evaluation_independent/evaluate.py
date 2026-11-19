"""Single-GPU worker: native FP32, fixed greedy generation, no judge model."""
import argparse
import re
import sys
import time
from common import *
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, str(TRAIN / 'src'))
import torch
from PIL import Image
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig

@torch.inference_mode()
def greedy(model, ids, pixels, eos, limit=16):
    out = model(input_ids=ids, pixel_values=pixels, use_cache=True, logits_to_keep=1)
    first = out.logits[0, -1].float().clone()
    generated = []
    for step in range(limit):
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(token.item())
        if token.item() == eos or step == limit-1: break
        out = model(input_ids=token, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
    return generated, first

def main(label):
    protocol = json.loads((HERE / 'protocol.json').read_text())
    for name, digest in protocol['code'].items(): assert sha(HERE / name) == digest, name
    for name, digest in protocol['inputs'].items(): assert sha(HERE / name) == digest, name
    for name, digest in protocol['source_files'].items(): assert sha(name) == digest, name
    for name, digest in protocol['vision_files'].items(): assert sha(name) == digest, name
    weight = MODELS[label]
    assert sha(weight) == protocol['models'][label]['sha256']
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = MiniMindVLM(VLMConfig(), vision_model_path=str(TRAIN / 'src/model/siglip2-base-p32-256-ve'))
    assert model.vision_encoder is not None
    result = model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=False)
    allowed = ('vision_encoder.', 'vision_proj.') if label == 'initial' else ('vision_encoder.',)
    assert not result.unexpected_keys and all(k.startswith(allowed) for k in result.missing_keys)
    model = model.float().eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(TRAIN / 'src/model', local_files_only=True)
    letters = [tokenizer(x, add_special_tokens=False).input_ids for x in 'ABCD']
    assert all(len(x) == 1 for x in letters) and len(set(x[0] for x in letters)) == 4
    letter_ids = [x[0] for x in letters]
    protocol_sha = sha(HERE / 'protocol.json')
    receipt = {'model': label, 'started_at': now(), 'pid': os.getpid(), 'gpu': os.getenv('CUDA_VISIBLE_DEVICES'),
               'weight_sha256': sha(weight), 'protocol_sha256': protocol_sha, 'dtype': str(next(model.parameters()).dtype),
               'torch': torch.__version__, 'letter_token_ids': letter_ids, 'parity': []}
    for condition in ['correct', 'shuffled', 'none', 'text']:
        dataset = rows(HERE / ('data/text64.jsonl' if condition == 'text' else 'data/mmstar.jsonl'))
        path = HERE / f'predictions/{label}_{condition}.jsonl'
        path.parent.mkdir(exist_ok=True)
        old = rows(path) if path.exists() else []
        assert [x['id'] for x in old] == [x['id'] for x in dataset[:len(old)]]
        assert all(x['protocol_sha256'] == protocol_sha for x in old)
        # A completed worker receipt is required as well as prediction rows for delivery.
        with path.open('a', buffering=1) as stream:
            for index, row in enumerate(dataset):
                if index < len(old): continue
                question = row['question'] + (protocol['prompt_suffix'] if condition != 'text' else '')
                has_image = condition in ['correct', 'shuffled']
                content = (model.config.image_special_token * 64 + '\n' if has_image else '') + question
                prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True, open_thinking=False)
                ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
                assert ids.shape[1] < 8192-16
                pixels = None
                image_sha = None
                if has_image:
                    image_path = row['image_file' if condition == 'correct' else 'shuffled_image_file']
                    image_sha = row['image_sha256' if condition == 'correct' else 'shuffled_image_sha256']
                    with Image.open(image_path) as image:
                        pixels = {k: v.cuda() for k, v in model.image2tensor(image.convert('RGB'), model.processor).items()}
                    assert (ids == model.config.image_ids[0]).sum().item() == 64
                else:
                    assert (ids == model.config.image_ids[0]).sum().item() == 0
                started = time.monotonic()
                generated, first = greedy(model, ids, pixels, tokenizer.eos_token_id)
                prediction = tokenizer.decode(generated, skip_special_tokens=True)
                if index < 2:
                    with torch.inference_mode():
                        reference = model.generate(inputs=ids, pixel_values=pixels, max_new_tokens=16, do_sample=False,
                            top_k=0, top_p=1.0, temperature=1.0, repetition_penalty=1.0, eos_token_id=tokenizer.eos_token_id)
                    assert reference[0, ids.shape[1]:].tolist() == generated, 'Efficient greedy differs from upstream generate'
                    receipt['parity'].append({'condition': condition, 'id': row['id'], 'upstream_greedy_token_equal': True})
                item = {'id': row['id'], 'model': label, 'condition': condition, 'protocol_sha256': protocol_sha,
                        'image_sha256': image_sha, 'question_tokens': ids.shape[1], 'generated_ids': generated,
                        'prediction': prediction, 'tokens': len(generated),
                        'hit_length_limit': len(generated) == 16 and generated[-1] != tokenizer.eos_token_id,
                        'seconds': time.monotonic()-started}
                if condition == 'text':
                    # Separate forward after generation: gold never enters generation or choice scoring.
                    target = tokenizer(row['answer'], add_special_tokens=False).input_ids
                    full = torch.cat([ids, torch.tensor([target], device='cuda')], dim=1)
                    with torch.inference_mode():
                        logits = model(input_ids=full).logits[0, ids.shape[1]-1:full.shape[1]-1].float()
                        nll = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device='cuda'), reduction='sum').item()
                    item.update(exact_numeric=(prediction.strip() == row['answer']), answer_tokens=len(target), nll_sum=nll)
                else:
                    logprobs = torch.log_softmax(first, dim=-1)[letter_ids].cpu().tolist()
                    item.update(choice_logprobs=logprobs, restricted_choice='ABCD'[max(range(4), key=lambda x: logprobs[x])],
                                strict_choice=strict_choice(prediction), official_choice=official_choice(prediction))
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                if (index + 1) % 50 == 0 or index == len(dataset)-1:
                    dump(HERE / f'artifacts/progress_{label}.json', {'at': now(), 'model': label, 'condition': condition, 'completed': index+1, 'total': len(dataset), 'pid': os.getpid()})
        print(label, condition, 'COMPLETE', len(dataset), flush=True)
    receipt.update(finished_at=now(), status='complete', exit_code=0,
                   files={p.name: sha(p) for p in sorted((HERE / 'predictions').glob(f'{label}_*.jsonl'))})
    dump(HERE / f'artifacts/receipt_{label}.json', receipt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--model', choices=list(MODELS), required=True)
    main(parser.parse_args().model)
