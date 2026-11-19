"""Ask the official local model from a terminal."""
import argparse
from pathlib import Path
from local_runtime import LocalVLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MiniMind-V 本地命令行图片问答')
    parser.add_argument('--image', type=Path, help='图片路径；省略时进行纯文本问答')
    parser.add_argument('--question', default='请描述这张图中的主要物体和场景。')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--greedy', action='store_true', help='使用确定性的贪心解码')
    parser.add_argument('--model-path', type=Path, help='训练后导出的模型文件夹')
    parser.add_argument('--vision-path', type=Path, help='视觉编码器文件夹')
    args = parser.parse_args()
    if args.image and not args.image.is_file():
        parser.error(f'图片不存在：{args.image}')
    if not 1 <= args.max_new_tokens <= 512:
        parser.error('--max-new-tokens 应为 1 到 512')
    engine = LocalVLM(args.device, args.model_path, args.vision_path)
    for text in engine.answer(args.question, str(args.image) if args.image else None,
                              max_new_tokens=args.max_new_tokens, sample=not args.greedy):
        print(text, end='', flush=True)
    print()
