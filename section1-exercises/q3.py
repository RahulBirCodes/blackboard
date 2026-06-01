import matplotlib.pyplot as plt
import numpy as np

batches = np.arange(1, 500)
BANDWIDTH = 8.2e11
FLOPS_S = 1.97e14


def calc_peak_flops(f, d):
    def solve(b):
        flops = 2 * b * f * d
        bytes_moved = (2 * b * d) + (d * f) + (2 * b * f)
        ai = flops / bytes_moved
        peak_flops_s = np.minimum(ai * BANDWIDTH, FLOPS_S)

        return peak_flops_s

    return solve


f_d_4096 = calc_peak_flops(4096, 4096)
f_d_1024 = calc_peak_flops(1024, 1024)

plt.plot(batches, f_d_4096(batches))
plt.plot(batches, f_d_1024(batches))
plt.xlabel("batch size")
plt.ylabel("peak flops/s")
plt.show()

