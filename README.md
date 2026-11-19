# MiniMind-V Reproduction and Visual Grounding Evaluation

An end-to-end reproduction of MiniMind-V Dense with an Apple MPS image QA app and two controlled Chinese visual instruction-tuning experiments. This repository contains the research record, reproducible scoring, source, and checkpoint releases.

## Research question

Does balancing counting answers and spatial supervision improve image-dependent Chinese VQA when two continuation arms share the images, starting weights, and update budget?

| Evaluation | Conventional A | Balanced B | Paired B − A |
| --- | ---: | ---: | ---: |
| Natural photos, semantic accuracy | 113/240 (47.08%) | 113/240 (47.08%) | 0.00 pp, 95% CI [−5.00, +5.00] |
| Held-out constructed challenge, semantic accuracy | 65/240 (27.08%) | 130/240 (54.17%) | +27.08 pp, 95% CI [+19.58, +35.00] |
| Constructed challenge, both images correct | 1/120 | 25/120 | +20.00 pp, 95% CI [+12.50, +27.50] |

Each arm used 3,200 matched images and 400 AdamW updates from the same prior SFT checkpoint. The natural-photo acceptance gate failed in seed 42, so the second seed was not triggered. The constructed challenge used held-out source photos with the same rendering family as training. Most of its gain came from integer counting and card occlusion order. Full reports record the remaining response biases, failure cases, and the sensitivity analysis for ambiguous photo labels.

[Final report (PDF, Chinese)](reports/MiniMind-V_项目收尾报告.pdf) · [Methods and results (Markdown, Chinese)](balanced_sft/实验结论.md) · [Reproduction guide](docs/reproduction.md) · [Portfolio notes (Chinese)](balanced_sft/简历与面试材料.md)

## Engineering and evaluation record

| Component | Evidence |
| --- | --- |
| Dense training reproduction | 4×RTX 4090, PyTorch DDP/BF16, 1,274,698 projector-alignment rows and 2,904,511 SFT rows. Each stage used two epochs, for 402,900 recorded Adam updates. [Training report](training/results/REPORT.md) |
| Checkpoint and deployment | Resume and frozen-tensor audits, native to Safetensors verification, loopback WebUI and Apple MPS real-weight checks. [Local setup](本地使用说明.md) · [Runtime evidence](validation/) · [Private checkpoint release](https://github.com/Asuka2112/minimind-v-reproduction/releases/tag/v1.0.0) |
| Independent benchmarks | MMStar original/swapped/no-image diagnostics and the initial Chinese VQA study, with retained predictions and score definitions. [MMStar](evaluation_independent/REPORT.md) · [Chinese VQA](evaluation_chinese/验收结论.md) |
| First controlled continuation | Same-image 2,048-example A/B with 6,720 visual prediction records. The directed arm reached 43.33% against 25.42% for the starting SFT, while its +1.25 pp over A missed the frozen gate. [First A/B report](targeted_sft/实验结论.md) |
| Balanced continuation | Frozen data and decode rules, 9,120 visual and 300 text prediction records, 132 blinded judgments, per-arm receipts and paired bootstrap results. [Protocol](balanced_sft/protocol.json) · [Analysis](balanced_sft/acceptance_analysis.json) |

## Verify the archive

With Python 3.12 from the repository root:

```sh
python3 scripts/verify_archive.py
```

This checks the compressed research streams, hashes and record counts. Install the pinned replay dependencies and use `--replay` to recompute both A/B conclusions from per-item predictions. Use the [reproduction guide](docs/reproduction.md) to install the private checkpoint assets, run the WebUI, and locate dataset and training dependencies.

## Attribution and citation

The model architecture, tokenizer, starter trainers and example images originate from [jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v) at commit [041d528](https://github.com/jingyaogong/minimind-v/tree/041d528e037a8a373c3c8aa7278a3ba3bd26b6dc). This project reproduced training, added local deployment and matched evaluation, and archived the resulting evidence. Use [CITATION.cff](CITATION.cff) for this repository and cite upstream work separately. [License](LICENSE) · [Third-party sources](docs/provenance.md)
