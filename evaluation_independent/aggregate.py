"""Recompute all scores from raw predictions, including pinned official rule parity."""
import ast
import collections
import copy
import math
import numpy as np
import pandas as pd
from common import *

def main():
    protocol_sha = sha(HERE / 'protocol.json')
    protocol = json.loads((HERE / 'protocol.json').read_text())
    test = rows(HERE / 'data/mmstar.jsonl')
    audit = json.loads((HERE / 'artifacts/overlap_audit.json').read_text())
    assert audit['status'] == 'complete' and audit['protocol_sha256'] == protocol_sha
    truth = {x['id']: x for x in test}
    source = HERE / 'reference_mmstar.py'
    assert sha(source) == '297af733471dbada39f252889742ae1f0c5707a751278ca0a5b1c3d2280da1f7'
    func = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'MMStar_eval')
    official_eval = compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])), str(source), 'exec')
    class Quiet:
        def info(self, *args): pass
    predictions = {}; metrics = {}; official_parities = []
    for model in MODELS:
        receipt = json.loads((HERE / f'artifacts/receipt_{model}.json').read_text())
        assert receipt['status'] == 'complete' and receipt['exit_code'] == 0
        assert receipt['protocol_sha256'] == protocol_sha
        assert receipt['weight_sha256'] == protocol['models'][model]['sha256']
        assert len(receipt['parity']) == 8 and all(x['upstream_greedy_token_equal'] for x in receipt['parity'])
        for condition in ['correct', 'shuffled', 'none']:
            name = f'{model}_{condition}'
            path = HERE / f'predictions/{name}.jsonl'
            assert sha(path) == receipt['files'][path.name]
            result = rows(path)
            assert [x['id'] for x in result] == [x['id'] for x in test]
            assert all(x['protocol_sha256'] == protocol_sha for x in result)
            for x in result:
                y = truth[x['id']]
                assert x['strict_choice'] == strict_choice(x['prediction'])
                assert x['official_choice'] == official_choice(x['prediction'])
                assert len(x['choice_logprobs']) == 4 and all(math.isfinite(v) for v in x['choice_logprobs'])
                assert x['restricted_choice'] == 'ABCD'[int(np.argmax(x['choice_logprobs']))]
                if condition == 'none': assert x['image_sha256'] is None
                else: assert x['image_sha256'] == y['image_sha256' if condition == 'correct' else 'shuffled_image_sha256']
            predictions[name] = result
            # Execute the pinned upstream function itself with IO/logging replaced, not its parser.
            frame = pd.DataFrame([{'prediction': x['prediction'], 'answer': y['answer'], 'bench': y['meta_info']['source'],
                                   'category': y['category'], 'l2_category': y['l2_category']} for x, y in zip(result, test)])
            captured = {}
            ns = {'deepcopy': copy.deepcopy, 'load': lambda _: frame, 'get_logger': lambda _: Quiet(),
                  'tqdm': lambda x: x, 'dump': lambda value, _: captured.update(value)}
            exec(official_eval, ns)
            ns['MMStar_eval']('predictions.xlsx')
            own = sum(official_choice(x['prediction']) == y['answer'] for x, y in zip(result, test)) / len(test)
            assert own == captured['final score']
            official_parities.append({'job': name, 'same_score_as_executed_pinned_official_function': True, 'score': own})
    categories = sorted({x['category'] for x in test})
    subsets = {'full': np.ones(len(test), dtype=bool),
               'exact_free': np.array([audit['row_masks'][x['id']]['exact_free'] for x in test]),
               'conservative_free': np.array([audit['row_masks'][x['id']]['conservative_free'] for x in test])}
    comparisons = [('sft_correct', 'initial_correct'), ('sft_correct', 'official_correct'),
                   ('sft_correct', 'sft_shuffled'), ('sft_correct', 'sft_none')]
    diffs = {}
    for subset, mask in subsets.items():
        selected = np.flatnonzero(mask)
        assert len(selected)
        images = sorted({test[i]['image_sha256'] for i in selected})
        image_index = {v: i for i, v in enumerate(images)}
        group = np.array([image_index[test[i]['image_sha256']] for i in selected])
        counts = np.bincount(group, minlength=len(images))
        rng = np.random.default_rng(20261103)
        sampled = rng.integers(0, len(images), size=(2000, len(images)))
        denominators = counts[sampled].sum(1)
        vectors = {}
        def interval(vector):
            successes = np.bincount(group, weights=vector[selected], minlength=len(images))
            samples = successes[sampled].sum(1) / denominators
            return [float(v) for v in np.percentile(samples, [2.5, 97.5])]
        metrics[subset] = {'rows': len(selected), 'unique_rgb_images': len(images), 'jobs': {}}
        for name, result in predictions.items():
            scores = {metric: np.array([x[key] == y['answer'] for x, y in zip(result, test)], dtype=float)
                      for metric, key in [('strict', 'strict_choice'), ('official', 'official_choice'), ('restricted', 'restricted_choice')]}
            vectors[name] = scores
            item = {metric: {'accuracy': float(vector[mask].mean()), 'correct': int(vector[mask].sum()), 'ci95_image_cluster': interval(vector)} for metric, vector in scores.items()}
            item.update(strict_format_rate=float(np.mean([result[i]['strict_choice'] is not None for i in selected])),
                        length_cap_rate=float(np.mean([result[i]['hit_length_limit'] for i in selected])),
                        restricted_letter_counts=dict(collections.Counter(result[i]['restricted_choice'] for i in selected)),
                        strict_letter_counts=dict(collections.Counter(str(result[i]['strict_choice']) for i in selected)),
                        categories={cat: {'rows': sum(test[i]['category'] == cat for i in selected),
                                          **{metric: float(vector[mask & np.array([x['category'] == cat for x in test])].mean()) for metric, vector in scores.items()}}
                                    for cat in categories})
            metrics[subset]['jobs'][name] = item
        diffs[subset] = {}
        for a, b in comparisons:
            diffs[subset][f'{a}_minus_{b}'] = {}
            for metric in ['strict', 'official', 'restricted']:
                vector = vectors[a][metric] - vectors[b][metric]
                diffs[subset][f'{a}_minus_{b}'][metric] = {'difference_pp': float(vector[mask].mean()*100),
                    'ci95_image_cluster_pp': [x*100 for x in interval(vector)]}
    text = {}
    for model in MODELS:
        results = rows(HERE / f'predictions/{model}_text.jsonl')
        gold = rows(HERE / 'data/text64.jsonl')
        assert [x['id'] for x in results] == [x['id'] for x in gold]
        assert all(x['exact_numeric'] == (x['prediction'].strip() == y['answer']) for x, y in zip(results, gold))
        nll = sum(x['nll_sum'] for x in results) / sum(x['answer_tokens'] for x in results)
        text[model] = {'rows': len(results), 'exact_correct': sum(x['exact_numeric'] for x in results),
                       'exact_accuracy': sum(x['exact_numeric'] for x in results)/len(results), 'answer_token_nll': nll,
                       'length_cap_rate': sum(x['hit_length_limit'] for x in results)/len(results)}
    summary = {'at': now(), 'status': 'complete', 'protocol_sha256': protocol_sha, 'metrics': metrics,
               'paired_differences': diffs, 'text64': text, 'overlap': {k: audit[k] for k in ['rows', 'unique_rgb_images', 'exact_match_rows', 'conservative_match_rows', 'exact_free_rows', 'conservative_free_rows', 'scope']},
               'gold_letter_counts': dict(collections.Counter(x['answer'] for x in test)),
               'official_rule_parity': official_parities,
               'interpretation': 'Restricted-letter scores are a separate forced-choice diagnostic, never interchangeable with generated-answer accuracy. No foundation-model contamination or general language retention guarantee.'}
    dump(HERE / 'summary.json', summary)
    lines = ['# MiniMind-V 独立评测报告', '', f'完成时间（UTC）：{summary["at"]}', '',
             '所有测试规则在本次推理前固定。完整外部测试集为 MMStar 1,500 题、1,430 张不同 RGB 图片；四个模型使用同一源码、处理器、FP32 推理和 greedy16。初始模型使用官方语言权重与 seed42 随机视觉投影。', '',
             '## 完整测试集', '', '| 模型 | 严格生成正确率 | 官方规则正确率 | A–D 概率诊断 | 单字母合规率 |', '|---|---:|---:|---:|---:|']
    labels = {'initial': '官方语言初始权重', 'pretrain': '本次预训练', 'sft': '本次 SFT', 'official': '官方发布 VLM'}
    for model in MODELS:
        item = metrics['full']['jobs'][f'{model}_correct']
        lines.append(f'| {labels[model]} | {item["strict"]["accuracy"]:.2%} | {item["official"]["accuracy"]:.2%} | {item["restricted"]["accuracy"]:.2%} | {item["strict_format_rate"]:.2%} |')
    lines += ['', '严格生成只接受完整单字母（允许括号、句点和空白）；其他回答一律计错。官方解析逐项与固定版本的原始 MMStar_eval 函数执行结果一致。A–D 概率诊断强制从首位置四个字母分数选一个，随机均匀选择的期望为 25%；该列不能作为生成能力或排行榜成绩。', '',
              '## 本次 SFT 的图片依赖检查', '', '| 条件 | 严格生成 | 官方规则 | A–D 概率诊断 |', '|---|---:|---:|---:|']
    for condition in ['correct', 'shuffled', 'none']:
        item = metrics['full']['jobs'][f'sft_{condition}']
        lines.append(f'| {condition} | {item["strict"]["accuracy"]:.2%} | {item["official"]["accuracy"]:.2%} | {item["restricted"]["accuracy"]:.2%} |')
    lines += ['', '差值以百分点表示，括号为按 RGB 图片聚类的 2,000 次配对 bootstrap 95% 区间；只描述测试图片抽样波动，不代表多次训练重复的方差。', '']
    for name, item in diffs['full'].items():
        lines.append(f'- {name}：' + '；'.join(f'{metric} {v["difference_pp"]:+.2f} [{v["ci95_image_cluster_pp"][0]:+.2f}, {v["ci95_image_cluster_pp"][1]:+.2f}]' for metric, v in item.items()))
    lines += ['', '## 训练图片重合核查', '',
              f'重新核查两份实际全量训练文件，共 {sum(x["rows"] for x in audit["sources"]):,} 行。完全相同编码或 RGB 图片涉及 {audit["exact_match_rows"]} 道题；加入 dHash≤3 的保守近似候选后涉及 {audit["conservative_match_rows"]} 道题。近似候选可能误报，不能等同于人工确认的重复。', '',
              '| 子集 | 题数 | SFT 严格生成 | SFT 官方规则 | SFT A–D 概率诊断 |', '|---|---:|---:|---:|---:|']
    for subset in subsets:
        item = metrics[subset]['jobs']['sft_correct']
        lines.append(f'| {subset} | {metrics[subset]["rows"]} | {item["strict"]["accuracy"]:.2%} | {item["official"]["accuracy"]:.2%} | {item["restricted"]["accuracy"]:.2%} |')
    lines += ['', '该核查只覆盖此次视觉训练的原始数据。初始语言模型、SigLIP 和官方参考模型更早的预训练污染无法排除，因此不能把任何子集称为“已证明完全无污染”。没有复用旧项目的测试切分或旧模型预测。', '',
              '## 中文文本诊断', '', '| 模型 | 64 条算术严格正确数 | 答案 token NLL |', '|---|---:|---:|']
    for model, item in text.items(): lines.append(f'| {labels[model]} | {item["exact_correct"]}/64 | {item["answer_token_nll"]:.4f} |')
    lines += ['', '这是固定中文算术小样本诊断，只能发现该任务上的变化；不能据此声称一般语言能力保持或丧失。NLL 只统计标准数字答案，不含 EOS。', '',
              '## 复核与限制', '',
              '原始预测保存在 predictions/，完整分领域结果、区间和字母偏置保存在 summary.json。protocol.json 固定输入、模型与脚本哈希；artifacts/ 保存执行回执、完整训练源图片重合核查和进度。每个模型的四种条件各验证两条输出与原生 generate token 完全一致。', '',
              '本报告是本次复现模型的外部评测，不是独立第三方认证。英文高难度基准与小模型的中文图文训练存在明显领域差异；单一基准不足以代表中文开放式图片问答体验。没有根据这些结果追加训练、筛题、选 checkpoint 或调提示。', '',
              '来源：[MMStar 官方数据](https://huggingface.co/datasets/Lin-Chen/MMStar/tree/bc98d668301da7b14f648724866e57302778ab27)、[固定版本官方评分代码](https://github.com/MMStar-Benchmark/MMStar/blob/88f243ab4a39cb339530085c33aecb22819881a1/eval/vlmeval/evaluate/mmstar.py)。', '']
    (HERE / 'REPORT.md').write_text('\n'.join(lines))
    dump(HERE / 'artifacts/delivery_verification.json', {'at': now(), 'status': 'verified', 'jobs': 12,
        'mmstar_predictions': 18000, 'text_predictions': 256, 'official_parser_parities': len(official_parities),
        'upstream_greedy_token_parities': 32, 'summary_sha256': sha(HERE / 'summary.json'), 'report_sha256': sha(HERE / 'REPORT.md'),
        'prediction_hashes': {p.name: sha(p) for p in sorted((HERE / 'predictions').glob('*.jsonl'))}})
    print('REPORT_COMPLETE', flush=True)

if __name__ == '__main__': main()
