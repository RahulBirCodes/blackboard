Jax + TPU/GPU scaling work.

All exercises done by hand - goal was to understand the scaling aspect of training LLMs.

Work includes: bidirectional collective matmuls written for different sharding schemes (with profiled speedups), MoE routing + expert parallelism. Sharding + collective matmuls were run + profiled on a v5e-8 slice.

The chinchilla-exercise folder contains the "final project" of this self-learning sprint. Ona single v5e I fit all 3 chinchilla fitting approaches and compare between MoE and dense models. However, due to how easy the problem is, N_opt flattens to ~= 0 with all the compute allocation going to data.
Afterwards, I write a custom pallas kernel completely fusing the SwiGLU MLP block - giving up to a 1.7x speedup compared to XLA's compiled version for just the forward pass. I use custom_vjp and fuse the MLP's backward pass in another custom pallas kernel - giving up to a 1.77x speedup for an entire training step.
