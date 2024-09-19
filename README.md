# QKV Transformers are RNN models with extra steps and memory capacity

**Author: Eugene Cheah ( @picocreator )**

## Abstract

The information stored into a transformer model KV cache, represents not just the existing token information. But the model embedding state generated recurrently from the previous tokens, mixed in. We Prove this by prompting the model with a critical piece of information “the needle”, in a multi-step/chain-of-thought format, while avoiding repeating the critical information. We subsequently modify the KV cache, to remove the needle and prove that the subsequent embedding had sufficient information kept recurrently, to answer the question.
