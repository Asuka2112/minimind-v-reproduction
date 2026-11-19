"""Run: .venv/bin/python local_app.py --open"""
import argparse
import atexit
import fcntl
import json
import os
import time
from pathlib import Path
from threading import Event

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import gradio as gr
from local_runtime import LocalVLM, ROOT


def create_app(engine):
    active = {}

    def stop_generation(request: gr.Request):
        event = active.get(request.session_hash)
        if event is not None:
            event.set()
        return '已停止生成'

    def clear_conversation(request: gr.Request):
        stop_generation(request)
        return [], '对话已清空'

    def change_image(request: gr.Request):
        stop_generation(request)
        return [], '图片已更新，可以开始提问'

    def respond(question, image, history, temperature, token_limit, request: gr.Request):
        if not question or not question.strip():
            raise gr.Error('请输入问题，或选择下方的示例。')
        previous = list(history or [])
        history = previous + [{'role': 'user', 'content': question.strip()},
                              {'role': 'assistant', 'content': ''}]
        cancelled = Event()
        active[request.session_hash] = cancelled
        started = time.monotonic()
        stream = engine.answer(question, image, previous, temperature, token_limit, cancelled=cancelled)
        try:
            yield history, '', '正在阅读图片并生成回答…'
            for chunk in stream:
                history[-1]['content'] += chunk
                yield history, '', '正在生成…'
        except Exception as error:
            raise gr.Error(f'生成失败：{error}') from error
        finally:
            stream.close()
            if active.get(request.session_hash) is cancelled:
                active.pop(request.session_hash, None)
        state = '已停止生成' if cancelled.is_set() else '完成'
        yield history, '', f'{state} · {time.monotonic() - started:.1f} 秒 · {engine.device.upper()}'

    css = '''
    .gradio-container {max-width:1200px!important;margin:auto!important}
    #hero {padding:12px 0 14px;border-bottom:1px solid var(--border-color-primary);margin-bottom:8px}
    #hero h1 {font-size:32px;letter-spacing:-1px;margin:0 0 8px}
    #hero p {margin:0;color:var(--body-text-color-subdued)}
    #status {font-size:13px;color:var(--body-text-color-subdued)}
    footer {display:none!important}
    '''
    theme = gr.themes.Soft(primary_hue='teal', font=['system-ui', 'sans-serif'],
                           font_mono=['ui-monospace', 'monospace'])
    with gr.Blocks(title='MiniMind-V · 本地视觉问答', theme=theme, css=css) as demo:
        gr.HTML('<div id="hero"><h1>MiniMind-V <span style="color:#0d9488">视觉问答</span></h1>'
                '<p>上传一张图片，从一个问题开始。模型在本机运行，支持中文和英文。</p></div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=300):
                image = gr.Image(label='图片', type='filepath', sources=['upload', 'clipboard'], height=230)
                question = gr.Textbox(label='问题', placeholder='这张图片里有什么？', lines=2)
                with gr.Row():
                    submit = gr.Button('发送问题', variant='primary', scale=3)
                    stop = gr.Button('停止', scale=1)
                gr.Markdown('**试试示例** · 点击图片后发送问题')
                sample_paths = sorted((ROOT / 'dataset' / 'eval_images').glob('*.jpg'))
                examples = gr.Gallery(value=[str(path) for path in sample_paths], columns=6, rows=1,
                                      height=88, show_label=False, allow_preview=False,
                                      object_fit='cover', interactive=False, show_download_button=False)
                def choose_example(event: gr.SelectData):
                    return str(sample_paths[event.index]), '请描述这张图中的主要物体和场景。'
                examples.select(choose_example, outputs=[image, question], api_name=False)
                with gr.Accordion('生成设置', open=False):
                    temperature = gr.Slider(0.1, 1.2, value=0.7, step=0.1, label='创造性')
                    token_limit = gr.Slider(32, 512, value=256, step=32, label='最长回答（token）')
            with gr.Column(scale=6, min_width=320):
                chatbot = gr.Chatbot(type='messages', label='对话', height=480,
                                     placeholder='上传图片并提问，回答将在这里逐字显示。')
                status = gr.Markdown(f'已就绪 · {engine.model_name} · {engine.device.upper()} · 完全本地推理', elem_id='status')
                clear = gr.Button('清空对话')
        gr.Markdown('更换图片会开始新对话；保持同一张图片可连续追问。小型模型可能看错细节。 '
                    '[原项目与 Apache-2.0 许可](https://github.com/jingyaogong/minimind-v)')
        inputs = [question, image, chatbot, temperature, token_limit]
        outputs = [chatbot, question, status]
        send_event = submit.click(respond, inputs, outputs, api_name='chat', concurrency_limit=1)
        enter_event = question.submit(respond, inputs, outputs, api_name=False, concurrency_limit=1)
        stop.click(fn=stop_generation, outputs=status, queue=False, api_name=False)
        clear.click(fn=clear_conversation, outputs=[chatbot, status],
                    cancels=[send_event, enter_event], queue=False)
        image.change(fn=change_image, outputs=[chatbot, status],
                     cancels=[send_event, enter_event], queue=False)
    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MiniMind-V 本地图片问答')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--model-path', type=Path, help='官方或训练后导出的模型文件夹')
    parser.add_argument('--vision-path', type=Path, help='视觉编码器文件夹，默认使用已安装的 SigLIP')
    args = parser.parse_args()
    service_lock = (ROOT / '.local-app.lock').open('w')
    try:
        fcntl.flock(service_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(1, '本项目服务已经运行，请打开 http://127.0.0.1:8888\n')
    print('正在加载本地模型…', flush=True)
    engine = LocalVLM(args.device, args.model_path, args.vision_path)
    print(f'模型已加载：{engine.parameter_count / 1e6:.2f}M 总参数，设备 {engine.device}', flush=True)
    demo = create_app(engine).queue(default_concurrency_limit=1, max_size=16)
    pid_path = ROOT / '.local-app.pid'
    def clean_pid():
        if pid_path.exists() and json.loads(pid_path.read_text()).get('pid') == os.getpid():
            pid_path.unlink()
    atexit.register(clean_pid)
    pid_path.write_text(json.dumps({'pid': os.getpid(), 'url': f'http://{args.host}:{args.port}'}))
    demo.launch(
        server_name=args.host, server_port=args.port, inbrowser=args.open, share=False,
        allowed_paths=[str(ROOT / 'dataset' / 'eval_images')], show_error=True)
