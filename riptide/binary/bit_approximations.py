import numpy as np
import os

# Uses least squared approximation to compute the best true binary approximations
# for HWGQ binarization.

def get_binary_repr(value, bits):
    output = np.zeros(shape=bits)
    for bit in reversed(range(bits)):
        bit_set = int(value / 2**bit) != 0
        output[bit] = bit_set
        if bit_set:
            value -= 2**bit
    output =  np.flip(output)
    return output

def approximate_bits(num_bits, values):
    # Compose matrix A, the bits for each value, returns
    # list with bias at index 0 then bit values.
    A = []
    for i in range(len(values)):
        A.append(get_binary_repr(i, num_bits))
    A = np.asarray(A)
    output, _, _, _ = np.linalg.lstsq(A, values)
    output = np.flip(output)
    return output

def compute_approximate_clusters(bits):
    num_bits = len(bits)
    output = []
    for i in range(2**num_bits):
        bit_rep = get_binary_repr(i, num_bits)
        bit_rep = np.flip(bit_rep)
        val_sum = 0
        for j in range(num_bits):
            val_sum += bits[j] * bit_rep[j]
        output.append(val_sum)
    return np.asarray(output)

def load_clusters(bits, path="/root/Riptide/riptide/binary/HWGQ_clusters"):
    file_path = os.path.join(path, "lstsq_clusters_%d_bit.npy" % bits)
    return np.load(file_path).astype(np.float32)

def load_bits(bits, path="/root/Riptide/riptide/binary/HWGQ_clusters"):
    file_path = os.path.join(path, "lstsq_bit_values_%d_bit.npy" % bits)
    return np.load(file_path).astype(np.float32)

# Example computation
# bits = 4
# clusters = load_cluster(bits, binarizable=False).asnumpy()
# app_bits = approximate_bits(bits, clusters)
# compute_approximate_clusters(app_bits)
