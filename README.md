# Minimalistic BERT in Jax

DIY project to teach myself JAX. Currently work in progress, the plan is to get original BERT performance with clean architecture built from scratch and low (personal) budget for training.

- Training on English Wikipedia (might add samples from books3). BookCorpus is bad (see below)
- Finetuning on GLUE tasks (TBD)
- Tokenizer trained from scratch

## Checkpoints & training logs
 TBD


## Architectural changes
- No NSP loss, as it does not help performance (see papers below)
- Training only for 1 epoch, so no dropout and very little regularization in general
- Pre-LayerNorm transformer for training stability
- No masking, only using contiguous chunks of fixed length to increase per-step efficiency and simplify code
- Gradient accumulation to fit larger batch sizes, linear batch-size schedule (rationale: gradient noise scale, see below)
- One Cycle learning rate schedule for efficient training under constrained budget
- Mixed precision with loss scaling
- Trining the tokenizer using HuggingFace WordPiece training algorithm, which is actually just BPE. Inference rules are still from WordPiece


## Tech stack
Jax, optax, Equinox. HuggingFace datasets and tokenizers, Omegaconf for config, Weights&Biases for monitoring, numpy memmap for disk access,  Pytest for extensive unit testing.

While working on this, I have found and [fixed a small bug](https://github.com/patrick-kidger/equinox/pull/288) in Equinox. I have also found a bug in Optax, but it's [been known](https://github.com/deepmind/optax/issues/377) for months already.


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
