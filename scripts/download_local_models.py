"""Download pinned official weights, with size and SHA256 verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def valid(path, item):
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    if item.get('sha256'):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
    return True


def download(item, check=False):
    target = ROOT / item['destination'] / item['file']
    if valid(target, item):
        print(f'OK {target.relative_to(ROOT)}', flush=True)
        return
    if check:
        raise RuntimeError(f'资源缺失或校验失败：{target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.download')
    url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{item['file']}"
    print(f'下载 {target.relative_to(ROOT)}', flush=True)
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '20',
                    '--max-time', '900', '--silent', '--show-error', url, '-o', str(partial)], check=True)
    if not valid(partial, item):
        raise RuntimeError(f'下载校验失败：{partial}')
    partial.replace(target)
    print(f'完成 {target.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    items = json.loads((ROOT / 'resources.json').read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: download(item, args.check), items))
    # Preserve the upstream default eval_vlm.py entry point as well.
    native = ROOT / 'out' / 'sft_vlm_768.pth'
    if not args.check and not native.exists() and not native.is_symlink():
        native.parent.mkdir(exist_ok=True)
        native.symlink_to('../weights/minimind-3v/pytorch_model.bin')
    print('全部模型资源校验通过。')
