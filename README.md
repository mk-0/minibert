# Minimalistic BERT in Jax

DIY project to teach myself JAX. Currently work in progress, the plan is to get original BERT performance with clean architecture built from scratch and low (personal) budget for training.

- Training on English Wikipedia (might add samples from books3). Original BookCorpus is not a good dataset (see below)
- Finetuning on GLUE tasks (MNLI currently)
- Tokenizer trained from scratch

## Checkpoints & training logs
 Raw and finetuned checkpoints can be downloaded with git LFS.

 Current **MNLI scores (m/mm): 80.7 / 80.0**, as reported by the [GLUE dashboard](https://gluebenchmark.com/leaderboard).

Training overview can be found in the [training log](log.md).


## Architectural changes
- No NSP loss, as it does not help performance (see papers below)
- Training only for 1 epoch, so no dropout and very little regularization in general
- Pre-LayerNorm transformer for training stability
- No masking, only using contiguous chunks of fixed length to increase per-step efficiency and simplify code
- Gradient accumulation to fit larger batch sizes, linear batch-size schedule (rationale: gradient noise scale, see below)
- One Cycle learning rate schedule for efficient training under constrained budget
- Mixed precision with loss scaling
- Trianing the tokenizer using HuggingFace WordPiece training algorithm, which is actually just BPE. Inference rules are still from WordPiece


## Tech stack
Jax, optax, Equinox. HuggingFace datasets and tokenizers, Omegaconf for config, Weights&Biases for monitoring, numpy memmap for disk access,  Pytest for extensive unit testing.


## Main reference papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [How to Train BERT with an Academic Budget](https://arxiv.org/abs/2104.07705)
- [Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [How AI training scales](https://openai.com/research/how-ai-training-scales)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)
- [What the BookCorpus?](https://gist.github.com/alvations/4d2278e5a5fbcf2e07f49315c4ec1110)
- [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461)
- [A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426)
