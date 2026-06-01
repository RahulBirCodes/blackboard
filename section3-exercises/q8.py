import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P

"""
Allocate a 2d matrix and shard
across a 2d hardware grid.

Compare benchmarked time to ideal comm time.
"""

assert len(jax.devices()) == 8
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('x', 'y'))
jax.set_mesh(mesh)

print("Benchmark AllGather and compare to theoretical time.")

"""
Shard our array A so that each
device has 1 element.

A[I_x, J_y]
local shape: [1000, 100]
"""
A = jax.device_put(jnp.arange(4 * 1000 * 2 * 100, dtype="bfloat16").reshape(4 * 1000, 2 * 100), P('x', 'y'))
print("Original array sharding:")
jax.debug.visualize_array_sharding(A)


"""
Note: v5e-8 slice has no wraparound connection, so assume W_ici = unidirectional bandwidth.

AllGather_x(A[I_x, J_y]) = A[I, Jy]

Latency-bound:
Assuming T_min = 1us

T_total = T_min * (x - 1) = 3us.

Bandwidth-bound:

T_total = 2 * (I / x) * (J / y) * (x - 1) / W_ici
= 2 * 4 * 1000 / 4 * 2 * 100 / 2 * 3 / 4.5e10
= 13us

This load is bandwidth-bound and we should expect it to take around 13us.
"""

@jax.shard_map(in_specs=(P('x', 'y')), out_specs=P('x', 'y'))
def f1(x_block):
    return jax.lax.all_gather(x_block, 'x', tiled=True)

print("Above all gather function time:")
f1_jit = jax.jit(f1)
f1_jit(A).block_until_ready() # initial jit compilation
%timeit f1_jit(A).block_until_ready()  # measure actual time-ish (probably still including some other overhead as well)






print("\n\nBenchmark AllReduce and compare to theoretical time.")

"""
Shard our array A so that rows are sharded along x
and duplicated along y.

A[I_x, J]
local shape: [1000, 100]
"""
A = jax.device_put(jnp.arange(4 * 1000 * 2 * 100, dtype="bfloat16").reshape(4 * 1000, 2 * 100), P('x', None))
print("Original array sharding:")
jax.debug.visualize_array_sharding(A)


"""
AllReduce_y(A[I_x, J]{U_y}) = A[I_x, J]

Example decomposition into AG and RS:
ReduceScatter_yJ(A[I_x, J]{U_y}) = A[I_x, J_y]
AllGather_y(A[I_x, J_y]) = A[I_x, J]

Latency-bound:
T_total = 2 * 1us * (y - 1) = 2us

Bandwidth-bound:
T_total = 2 * (2 * (I/x) * (J/y)) * (y - 1) / W_ici
= 2 * 2 * (4000 / 4) * (200 / 2) * 1 / 9e10
= 4000 * 400 / 9e10 = 17us

Above is bandwidth bound.
"""
@jax.shard_map(in_specs=(P('x', None)), out_specs=P('x', None))
def f2(x_block):
    return jax.lax.psum(x_block, 'y')

print("Above all reduce function time")
f2_jit = jax.jit(f2)
f2_jit(A).block_until_ready()
%timeit f2_jit(A).block_until_ready()






print("\n\nBenchmark ReduceScatter and compare to theoretical time.")

"""
Shard A rows along x and duplicate along y.

A[I_x, J]
local shape: [4 * 1000 / 4, 2 * 100]
"""
A = jax.device_put(jnp.arange(4 * 1000 * 2 * 100, dtype="bfloat16").reshape(4 * 1000, 2 * 100), P('x', None))
print("Original array sharding:")
jax.debug.visualize_array_sharding(A)

"""
ReduceScatter_yJ(A[I_x, J]{U_y}) = A[I_x, J_y]

Latency-bound:
T_total = 1us * (y-1) = 1us

Bandwidth-bound:
T_total = 2 * (I / x) * (J / y) * (y - 1) / W_ici
= 2 * 4000 / 4 * 200 / 2 * 1 / 4.5e10
~= 4us

The above comm is bandwidth bounded.
"""
@jax.shard_map(in_specs=(P('x', None)), out_specs=P('x', 'y'))
def f3(x_block):
    return jax.lax.psum_scatter(x_block, 'y', scatter_dimension=1, tiled=True)

print("Above reduce scatter function time")
f3_jit = jax.jit(f3)
f3_jit(A).block_until_ready()
%timeit f3_jit(A).block_until_ready()






print("\n\nBenchmark AllToAll and compare to theoretical time.")

"""
Shard A rows along x and duplicate along y.

A[I_x, J]
local shape: [4 * 1000 / 4, 2 * 100]
"""
# A = jax.device_put(jnp.arange(4 * 1000 * 2 * 100, dtype="bfloat16").reshape(4 * 1000, 2 * 100), P('x', None))
A = jax.device_put(jnp.arange(4 * 4, dtype="bfloat16").reshape(4, 4), P('x', None))
print("Original array sharding:")
jax.debug.visualize_array_sharding(A)

"""
AllToAll_xJ(A[I_x, J]) = A[I, J_x]

Latency-bound:
T_total = 1us * (y-1) = 1us


Bandwidth-bound:
T_total = (1/4) * 2 * (I / x) * J * (x - 1) / W_ici
= 1/2 * 4000 / 4 * 200 * 3 / 4.5e10 = 6us
"""

@jax.shard_map(in_specs=(P('x', None)), out_specs=P(None, 'x'))
def f4(x_block):
    return jax.lax.all_to_all(x_block, 'x', split_axis=1, concat_axis=0, tiled=True)

print("Above all to all function time")
f4_jit = jax.jit(f4)
f4_jit(A).block_until_ready()
%timeit f4_jit(A).block_until_ready()
