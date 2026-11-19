"""Real-weight checks for image conditioning, failure recovery, and cancellation."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from local_runtime import LocalVLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, help='Directory of a local Transformers model export')
    parser.add_argument('--output', type=Path, help='Path for the JSON verification record')
    args = parser.parse_args()
    engine = LocalVLM(model_path=args.model_path)
    report = {'model': engine.model_name, 'device': engine.device,
              'total_parameters': engine.parameter_count, 'checks': {}}
    paths = sorted((ROOT / 'dataset/eval_images').glob('*.jpg'))
    prompt = '请描述这张图中的主要物体和场景。'
    inputs, pixels = engine.prepare(prompt, str(paths[0]))
    assert (inputs.input_ids == engine.config.image_ids[0]).sum().item() == 64
    report['checks']['image_placeholder_count'] = 64
    with torch.inference_mode():
        logits_a = engine.model(inputs.input_ids, pixel_values=pixels, logits_to_keep=1).logits
        _, pixels_b = engine.prepare(prompt, str(paths[3]))
        logits_b = engine.model(inputs.input_ids, pixel_values=pixels_b, logits_to_keep=1).logits
    assert torch.isfinite(logits_a).all() and torch.isfinite(logits_b).all()
    delta = (logits_a - logits_b).abs().max().item()
    assert delta > 0.01, 'Image did not affect model output'
    report['checks']['different_images_change_logits'] = delta
    try:
        list(engine.answer(prompt, '/missing-minimind-image.jpg', max_new_tokens=1))
        raise AssertionError('Missing image was accepted')
    except FileNotFoundError:
        report['checks']['missing_image_error_propagates'] = True
    try:
        list(engine.answer('长文本' * 4000, max_new_tokens=1))
        raise AssertionError('Oversize prompt was accepted')
    except ValueError:
        report['checks']['context_limit_error_propagates'] = True
    stream = engine.answer(prompt, str(paths[0]), max_new_tokens=512)
    next(stream)
    stream.close()
    assert engine.lock.acquire(timeout=15), 'Cancelled generation did not release the model'
    engine.lock.release()
    report['checks']['cancellation_releases_model'] = True
    started = time.monotonic()
    car_question = '这辆车是什么颜色的？'
    answer = ''.join(engine.answer(car_question, str(paths[3]), max_new_tokens=64, sample=False))
    assert answer.strip()
    report['checks']['generation_recovers_after_errors_and_cancel'] = True
    report['example'] = {'image': paths[3].name, 'question': car_question,
                         'answer': answer, 'seconds': round(time.monotonic() - started, 2)}
    default_name = 'runtime_custom.json' if args.model_path else 'runtime.json'
    destination = args.output or ROOT / 'validation' / default_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
