import numba
import numpy as np
import time


@numba.njit
def sum2d(arr: np.ndarray) -> float:
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i, j]
    return result


if __name__ == '__main__':
    array = np.random.rand(10000, 10000)
    tic = time.time()
    a = sum2d(array)
    tac = time.time()
    print(f'time taken: {tac - tic}')
    print('result: ', a)
