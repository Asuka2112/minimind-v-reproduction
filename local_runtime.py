"""Local MiniMind-V inference using the repository's own model implementation."""
import json
import os
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import torch
from PIL import Image, ImageOps
from transformers import AutoTokenizer, TextStreamer
from safetensors.torch import load_file
from model.model_vlm import MiniMindVLM, VLMConfig

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'weights' / 'minimind-3v'
VISION_PATH = ROOT / 'model' / 'siglip2-base-p32-256-ve'


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class GenerationCancelled(Exception):
    pass


class QueueStreamer(TextStreamer):
    def __init__(self, tokenizer, queue, cancelled):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.cancelled = cancelled

    def put(self, value):
        if self.cancelled.is_set():
            raise GenerationCancelled()
        super().put(value)

    def on_finalized_text(self, text, stream_end=False):
        if text:
            self.queue.put(('text', text))


class LocalVLM:
    def __init__(self, device='auto', model_path=None, vision_path=None):
        self.device = default_device() if device == 'auto' else device
        model_path = Path(model_path).expanduser().resolve() if model_path else MODEL_PATH
        vision_path = Path(vision_path).expanduser().resolve() if vision_path else VISION_PATH
        self.model_name = model_path.name
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        if not any((model_path / name).is_file() for name in ['pytorch_model.bin', 'model.safetensors']):
            raise FileNotFoundError('模型尚未安装，请运行 scripts/download_local_models.py')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        config = VLMConfig(**json.loads((model_path / 'config.json').read_text()))
        self.model = MiniMindVLM(config, vision_model_path=str(vision_path))
        if self.model.vision_encoder is None:
            raise FileNotFoundError(f'视觉编码器缺失或不兼容：{vision_path}')
        if (model_path / 'model.safetensors').is_file():
            state = load_file(str(model_path / 'model.safetensors'), device='cpu')
        else:
            state = torch.load(model_path / 'pytorch_model.bin', map_location='cpu', weights_only=True)
        result = self.model.load_state_dict(state, strict=False)
        missing = [key for key in result.missing_keys if not key.startswith('vision_encoder.')]
        if missing or result.unexpected_keys:
            raise RuntimeError(f'模型权重不匹配：missing={missing}, unexpected={result.unexpected_keys}')
        self.model = self.model.eval().to(self.device)
        self.lock = Lock()
        self.config = config
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def prepare(self, prompt, image_path=None, history=None):
        prompt = prompt.strip()
        if not prompt:
            raise ValueError('请输入问题。')
        # Image placeholders belong to the runtime, never to user-entered text.
        if self.config.image_special_token in prompt:
            raise ValueError('问题中不能包含模型的图像占位符。')
        messages = [{'role': item['role'], 'content': item['content']}
                    for item in (history or []) if isinstance(item.get('content'), str)]
        messages.append({'role': 'user', 'content': prompt})
        pixels = None
        if image_path:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
            pixels = {key: value.to(self.device) for key, value in
                      MiniMindVLM.image2tensor(image, self.model.processor).items()}
            # Reuse the same image for every follow-up in this conversation.
            messages[0]['content'] = self.config.image_special_token * self.config.image_token_len + '\n' + messages[0]['content']
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, open_thinking=False)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        if inputs.input_ids.shape[1] > 3072:
            raise ValueError('当前对话过长，请点击“清空对话”后继续。')
        return inputs, pixels

    def answer(self, prompt, image_path=None, history=None, temperature=0.7,
               max_new_tokens=256, sample=True, cancelled=None):
        queue = Queue()
        cancelled = cancelled if cancelled is not None else Event()

        def generate():
            try:
                with self.lock:
                    if cancelled.is_set():
                        return
                    inputs, pixels = self.prepare(prompt, image_path, history)
                    streamer = QueueStreamer(self.tokenizer, queue, cancelled)
                    self.model.generate(inputs=inputs.input_ids, attention_mask=inputs.attention_mask,
                                        pixel_values=pixels, max_new_tokens=int(max_new_tokens),
                                        temperature=max(float(temperature), 0.01), top_p=0.9,
                                        repetition_penalty=1.1, do_sample=sample,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id, streamer=streamer)
            except GenerationCancelled:
                pass
            except Exception as error:
                queue.put(('error', error))
            finally:
                queue.put(('end', None))

        thread = Thread(target=generate, daemon=True)
        thread.start()
        completed = False
        try:
            while True:
                kind, value = queue.get(timeout=120)
                if kind == 'end':
                    completed = True
                    break
                if kind == 'error':
                    raise value
                yield value
        finally:
            if not completed:
                cancelled.set()
