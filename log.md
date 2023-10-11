# Run 1 (24.09 Sun)
* **Differences:** Baseline run, based on [Cramming](https://arxiv.org/abs/2212.14034)
* **Commit**: 61f2df2
* **Run**: https://wandb.ai/mkitov/bert/runs/9o9a7xcc
* **Data**: English Wikipedia (3,851,197,952 tokens)
* **Result**: Converged after 10 hours.
* **Final loss**: ~=2.
* **MNLI m/mm**: 77.0 / 76.1 
* Problems:
    * MNLI accuracy too low. Little progress in terms of MNLI loss in the second half of the training.
* Hypothesis:
    * Considerably less data than in baseline, therefore batch accumulation schedule (especially together with learning rate cycle) is too agressive. Increasing dataset size.
* Other:
    * Due to a bug in processing linebreaks were ignored by tokenizer. Adding a special token and a test case.

## Sanity check demo

### Raw
| Input | Output |
--|--|
| "I **[MASK]** taking **[MASK]** nap." | "I **am** taking **a** nap. |
| George Washington was the first **[MASK]** of the United **[MASK]** of America | george washington was the first **president** of the united **states** of america |
| China is the most **[MASK]** country in the world. | china is the most **populous** country in the world. |
| Thermodynamics is an area of **[MASK]**. It studies **[MASK]** and related processes | thermodynamics is an area of **mathematics**. it studies **processes** and related processes |
| Two plus **[MASK]** equals four. Same statement in mathematical notation: 2 + 2 = **[MASK]** | two plus **two** equals four. same statement in mathematical notation : 2 + 2 = **1** |
| Rainbow consits of the following colors: red, **[MASK]**, yellow, **[MASK]**, **[MASK]**, violet. | rainbow consits of the following colors : red, **green**, yellow, **yellow**, **violet**, violet. |

### MNLI finetune
| Sentence 1 | Sentence 2 | Relation |
--|--|--|
| London is the capital of the empire | Paris is the capital of the empire | contradiction |
| London is the capital of the empire | The empire has a capital | entailment |
| London is the capital of the empire | London is a coastal city | contradiction |
| London is the capital of the empire | London was founded in 10th century | neutral |



# Run 2 (07.10 Sun)
* **Differences:** New dataset (and fresh tokenizer)
* **Commit**: head
* **Run**: https://wandb.ai/mkitov/bert/runs/pra81pkn?workspace=user-mkitov
* **Data**: 30% of books3 (7,334,400,000 tokens)
* **Result**: Converged after 19 hours.
* **Final loss**: ~=2.
* **MNLI m/mm**: 80.7 / 80.0
* Result:
    * Actually a healthy performance for 19 hours of single GPU training. Still less than full BERT-base (which has ~ 83%)

## Sanity check demo
### Raw
| Input | Output |
--|--|
| "I **[MASK]** taking **[MASK]** nap." | "I **am** taking **a** nap. |
| George Washington was the first **[MASK]** of the United **[MASK]** of America | george washington was the first **president** of the united **states** of america |
| China is the most **[MASK]** country in the world. | china is the most **powerful** country in the world. |
| Thermodynamics is an area of **[MASK]**. It studies **[MASK]** and related processes | thermodynamics is an area of **study**. it studies **processes** and related processes |
| Two plus **[MASK]** equals four. Same statement in mathematical notation: 2 + 2 = **[MASK]** | two plus **two** equals four. same statement in mathematical notation : 2 + 2 = **2** |
| Rainbow consits of the following colors: red, **[MASK]**, yellow, **[MASK]**, **[MASK]**, violet. | rainbow consits of the following colors : red, **blue**, yellow, **blue**, **blue**, violet. |


### MNLI finetune
| Sentence 1 | Sentence 2 | Relation |
--|--|--|
| London is the capital of the empire | Paris is the capital of the empire | contradiction |
| London is the capital of the empire | The empire has a capital | entailment |
| London is the capital of the empire | London is a coastal city | neutral |
| London is the capital of the empire | London was founded in 10th century | neutral |
