---
base_model: google/gemma-2-2b
library_name: transformers
model_name: GEMMA-2-2B-FT-ORPO-ISAERFT_run-gemma-2-2b-lr3.6e-05-beta0.2-20250228-2228
tags:
- generated_from_trainer
- smol-course
- module_1
- isaerft
- lr_3.568480459462587e-05
- beta_0.2
licence: license
---

# Model Card for GEMMA-2-2B-FT-ORPO-ISAERFT_run-gemma-2-2b-lr3.6e-05-beta0.2-20250228-2228

This model is a fine-tuned version of [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="AMindToThink/GEMMA-2-2B-FT-ORPO-ISAERFT_run-gemma-2-2b-lr3.6e-05-beta0.2-20250228-2228", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/matthewkhoriaty-northwestern-university/orpo-isaerft-sweep/runs/byuwqkyu) 


This model was trained with ORPO, a method introduced in [ORPO: Monolithic Preference Optimization without Reference Model](https://huggingface.co/papers/2403.07691).

### Framework versions

- TRL: 0.15.1
- Transformers: 4.49.0
- Pytorch: 2.6.0
- Datasets: 2.21.0
- Tokenizers: 0.21.0

## Citations

Cite ORPO as:

```bibtex
@article{hong2024orpo,
    title        = {{ORPO: Monolithic Preference Optimization without Reference Model}},
    author       = {Jiwoo Hong and Noah Lee and James Thorne},
    year         = 2024,
    eprint       = {arXiv:2403.07691}
}
```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallouédec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```