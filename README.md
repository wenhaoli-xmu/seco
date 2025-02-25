![img](docs/main.jpg)

## Overview

<!-- 我们提出了SeCO和SpaCO，用于在memory constrained scenarios下训练LLM。 -->
We propose SeCO and SpaCO for training LLMs under memory-constrained scenarios.

### Sequential Chunk-wise Optimization (SeCO)

* Employs a step-by-step strategy to execute forward propagation and localized backward propagation in chunks, with only one computational graph stored in GPU memory at any given time.

* Enables exact gradient computation, achieving gradient accuracy up to **12 decimal places** when using fp64 precision.

* Maintains near-native training speed when the chunk size is efficiently large, with no significant slowdown compared to conventional gradient checkpointing.

### Sparse Chunk-wise Optimization (SpaCO)

* Extends SeCO by introducing sparsification during backward propagation.

* Gradually aligns training costs with inference costs as context length increases.

* While unable to compute exact gradients, the resulting trade-offs remain practically acceptable for most applications.

Compared to mainstream training approaches, SeCO and SpaCO demonstrate substantial efficiency advantages:

![img](docs/efficiency.png)


