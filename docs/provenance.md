# Provenance and archive scope

The upstream MiniMind-V source, tokenizer, and bundled demo examples are retained with the Apache 2.0 license. The pin in the repository README is the architecture and tokenizer baseline. Initial external checkpoints, including SigLIP2 and the official MiniMind-V release, are referenced at their original sources. Author-created training outputs are available in the private checkpoint release.

- Upstream source and licenses. [MiniMind-V](https://github.com/jingyaogong/minimind-v) · [Apache-2.0](../LICENSE)
- Language, vision, and training data revisions. [Pinned training configuration](../training/config.json) · [official model card](https://huggingface.co/jingyaogong/minimind-3v) · [training dataset card](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset)
- COCO images retain individual source terms. Manifest files keep source IDs and transformation provenance for reconstruction. The archive includes rendered image hashes, labels and donor selections. [COCO usage terms](https://cocodataset.org/#termsofuse) · [balanced data](../balanced_sft/data/manifest.json)
- MMStar is an external diagnostic. Model predictions, pinned benchmark question metadata, and computed results retain original scope. [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) · [evaluation protocol](../evaluation_independent/protocol.json)

Bulk source image data, transient optimizer states, and credentials are kept outside the archive. The included reports retain the per-stage data counts, optimizer update counts, resume audit, checkpoint hashes and archived scores. Historical server path strings were normalized in selected files. The [sanitization inventory](../evidence/path-sanitization.json) maps changed file hashes to their historical hashes. The frozen receipt hashes still describe the original runs. The portability wrapper verifies archive bytes separately and the scoring replay verifies the stored conclusions.
