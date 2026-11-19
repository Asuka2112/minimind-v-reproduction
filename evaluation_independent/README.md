# 独立评测

本目录用于已完成的官方 MiniMind-V 复现模型的外部验证，不会更改训练权重。

已于北京时间 2026-11-03 03:43 完成。先阅读 [结论与解读](结论与解读.md)，再查看 [完整报告](REPORT.md)。

- `protocol.json`：第一次新推理前固定的模型、数据、代码哈希和评分规则。
- `predictions/`：12 组 MMStar 原始预测和 4 组中文算术诊断。
- `summary.json`：从原始预测重算的指标、分类结果、置信区间和对照差值。
- `REPORT.md`：中文报告。
- `artifacts/overlap_audit.json`：两份完整训练原始数据的图片重合核查。
- `artifacts/delivery_verification.json`：输出完整性和评分解析一致性验证。

服务器目录：`/workspace/projects/minimind-v-independent-eval-20261103`。

检查进度：

```bash
ssh research-gpu 'cat /workspace/projects/minimind-v-independent-eval-20261103/status.json'
```

在服务器目录中，使用已有环境重新计算指标（不再推理）：

```bash
/workspace/projects/minimind3-reproduction-20261019/env311/bin/python aggregate.py
```

`controller.py` 是一次性、可脱离 SSH 连接运行的执行入口。它先检查并租用空闲 GPU 0–3，再运行四个模型及 CPU 图片重合核查，全部成功才汇总。已有进度不会自动定时检查或发送通知。预测逐条保存；本次未验证故障后的续跑，失败后需保留现场并核对日志。协议、输入及脚本必须与冻结哈希一致。不要覆盖已冻结的协议来修改评测方法。

严格单字母生成、官方前缀解析、强制 A–D 选项概率是三种不同指标，不应互相替代。64 条中文算术题只是小样本诊断。任何“能力提升”“无污染”或简历数字均应以完整报告的范围和限制为准。
