# 官方两阶段训练与全链路复现

本轮按原网站流程补齐训练与交付。服务器使用本机已保存的 SSH 别名 `research-gpu`，工作目录独立于之前的 MiniMind/Qwen 实验。未创建定时任务；你手动回来检查即可。服务器上的单个持久进程会按顺序完成训练、评估、导出和报告。

## 检查进度

在本项目目录运行：

```bash
.venv/bin/python training/control.py status
```

输出包含控制器是否存活、当前阶段、训练 epoch/step/loss 和最近日志；同时保存到 `training/server_status.json`。`status=complete` 才表示正式训练及最终校验完成。`status=failed` 或 `process_alive=false` 时，应根据日志处理后再恢复，不把旧进度视为完成。

## 下载结果并在本机运行

```bash
# 手动同步日志、审计证据、报告和导出的模型；不复制数 GB 原始数据或原生恢复状态
.venv/bin/python training/control.py sync

# 完成后用自训 SFT 导出模型启动 WebUI；先停止当前已运行的本地服务
bash 停止服务.command
.venv/bin/python local_app.py --open --model-path training/results/exports/sft

# 自训模型命令行图片问答
.venv/bin/python local_cli.py --model-path training/results/exports/sft \
  --image dataset/eval_images/image-04-yellow-car.jpg \
  --question '请描述这张图片。'
```

模型尚未导出时，不运行自训模型启动命令。当前原有 WebUI 继续使用官方权重。

`exports/smoke` 仅用于八步恢复与格式验证，不是正式训练结果。最终使用 `exports/sft`，并核对全链路完成状态。

## 固定流程

1. 核对官方完整 Parquet 的 SHA256、行数和字段；核对官方语言基座与视觉编码器。八条真实样本经过原版 tokenizer、标签掩码和图像处理器检查。
2. 单卡烟测在第 2 次 Adam 更新后保存并退出，从原生状态恢复至共 8 次更新；核对恢复事件、Adam 计数、有限值和冻结参数。
3. 用烟测权重验证官方格式转换及外部视觉编码器重载；在正式长训练前比较原生 FP16 与导出模型的全部持久张量和一张真实图片前向。
4. 四卡烟测执行两次真实 DDP 更新与两次保存，确认各 rank 正常退出和检查点一致。
5. 用固定贪心解码记录初始模型、官方发布模型在网站六张示例图上的输出。
6. **Pretrain**：官方 `pretrain_i2t.parquet` 完整文件，2 轮，`freeze_llm=2`，仅训练视觉投影；每卡 batch16，LR4e-4，seq450。
7. 审计 Pretrain 检查点、导出为 Transformers 格式，运行相同六图推理。
8. **SFT**：从刚训练的 Pretrain 权重开始，官方 `sft_i2t.parquet` 完整文件，2 轮，`freeze_llm=1`，训练视觉投影与语言首尾层；每卡 batch4，LR5e-6，seq768。
9. 审计 SFT 检查点与导出；运行固定六图推理，以及原版 `eval_vlm.py` 入口（只修复绝对路径拼接）；生成汇总报告、实际统计与训练 loss 曲线。

完整参数见 `config.json`。两阶段使用 4 张空闲 RTX 4090（物理 2、3、4、5），全局 batch 分别为 64/16，BF16 autocast、FP32 主权重、累计1、clip1、workers0、compile0、seed42。这是原仓库支持的 DDP 配置；全局 batch 与单卡默认值不同，不宣称单卡逐步等价。

## 版本与实现边界

- 源码固定提交 `041d528e037a8a373c3c8aa7278a3ba3bd26b6dc`。
- 语言权重、视觉权重和数据版本/校验值全部在 `config.json`。语言基座使用官方 `llm_768.pth`，与旧项目自训基座分开。
- 原始上游保存在服务器 `upstream/`；运行副本在 `src/`。训练补丁保存为 `artifacts/trainer.patch`，仅处理 DDP 保存同步、原子保存、跨 rank 日志均值、非有限损失检查；评估绝对路径拼接补丁独立保存在 `artifacts/eval_path.patch`。不更换架构、损失、优化器或训练目标。
- 使用已有服务器 Python 环境，只读取其安装包，不修改旧环境；实际完整版本保存在 `artifacts/environment.lock.txt`，关键版本为 torch2.5.1+cu121、transformers4.57.6、datasets3.6.0、pyarrow20.0.0。
- 官方转换器保留 `.bin` 导出，同时生成 Safetensors 供服务器 Transformers 加载并校验；不关闭其 PyTorch 版本安全检查。本地入口优先读取 Safetensors。
- 每 5000 步和每个 epoch 末保存。保留最新及上一份原生恢复状态，阶段最终发布权重单独保存；失败后可从本阶段原生状态继续。
- 原生恢复包含模型/Adam/scaler/epoch/step；上游数据增强与 worker RNG 没有完整序列化，不宣称逐比特恢复。
- 使用网站提供的完整处理后数据，不把其他实验的清洗、划分、指标或 Qwen 结果归入本次复现。
- 六张图片用于网站示例的定性演示，不能报告测试准确率、OCR 提升或优于官方。后续优化和独立评测另行开展。

## 结果位置

服务器目录：`/workspace/projects/minimind-v-official-20261102`。同步后，本机 `training/results/` 中包含：

| 文件/目录 | 用途 |
| --- | --- |
| `pipeline_status.json` | 最后阶段与状态 |
| `REPORT.md` / `results.json` | 仅在全链路成功后生成的总结和实际指标 |
| `runs/*/events.jsonl` | 原始训练进度与 checkpoint/resume 事件 |
| `runs/*/checkpoint_audit.json` | Adam 计数、参数变化与冻结参数验证 |
| `runs/*/receipt-*.json` | 各 rank 的实际退出码、耗时和显存统计 |
| `exports/pretrain` / `exports/sft` | 两阶段的 Transformers 导出与一致性证明 |
| `evaluation/*/predictions.jsonl` | 固定图片与各模型原始回答 |
| `artifacts/` / `logs/` | 数据与环境清单、补丁、精确命令和日志 |
| `training_loss.png` | 日志中记录的训练 loss，不是验证 accuracy |

控制器重启命令（只在确认旧控制器已退出且处理好错误后执行）：

```bash
ssh research-gpu
cd /workspace/projects/minimind-v-official-20261102
nohup /workspace/projects/minimind3-reproduction-20261019/env311/bin/python \
  training/pipeline.py >> logs/pipeline.log 2>&1 < /dev/null &
```

控制器有单实例锁；已完成并通过审计的阶段会跳过，有本阶段恢复状态时使用原生恢复。若所选 GPU 被其他任务占用，会停止并记录原因，不终止其他进程。

## 简历材料

全链路完成前，只写“已部署/正在复现”。完成后，`REPORT.md` 会依据真实行数、Adam 计数和退出证明生成项目描述。可强调数据版本校验、两阶段训练、DDP/BF16、恢复验证、导出一致性及本地部署；能力提升数值需要之后的独立评测支持。
