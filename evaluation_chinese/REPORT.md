# 中文图片问答评测

152 题，76 个同题异答案图片对。官方权重和本次复现权重使用相同的 FP32 推理、192 token 贪心解码。
第一种提示直接提问，第二种提示追加简短作答要求。换图保留题目，通过同题下另一张图检查答案是否跟随图片。

主指标使用冻结的标准答案与同义词做归一化精确匹配。review.json 保存真实图片回答供语义复核。
真实图片沿用此前外部评测图源并重写题目。真实子集采用定向选样，可控图像由程序生成。

| 子集 | 模型/提示 | 原图正确率 | 换图对原图答案 | 换图对新图答案 | 无图正确率 | 原图长度上限 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| synthetic (128) | official/baseline | 0.0% | 0.0% | 0.0% | 0.0% | 3.1% |
| synthetic (128) | official/short | 0.0% | 0.0% | 0.0% | 0.0% | 2.3% |
| synthetic (128) | reproduced/baseline | 0.0% | 0.0% | 0.0% | 0.0% | 2.3% |
| synthetic (128) | reproduced/short | 0.0% | 0.0% | 0.0% | 0.0% | 3.9% |
| real_and_mirrored (24) | official/baseline | 0.0% | 0.0% | 0.0% | 0.0% | 4.2% |
| real_and_mirrored (24) | official/short | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| real_and_mirrored (24) | reproduced/baseline | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| real_and_mirrored (24) | reproduced/short | 0.0% | 0.0% | 0.0% | 0.0% | 16.7% |
| real_unmodified (21) | official/baseline | 0.0% | 0.0% | 0.0% | 0.0% | 4.8% |
| real_unmodified (21) | official/short | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| real_unmodified (21) | reproduced/baseline | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| real_unmodified (21) | reproduced/short | 0.0% | 0.0% | 0.0% | 0.0% | 19.0% |

results.json 附带逐类别分数、换图答案变化比例、两张图都答对的比例及配对区间。
