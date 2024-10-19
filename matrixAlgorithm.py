from memory_profiler import memory_usage
import numpy as np
import time
import psutil
import os

np.random.seed(27)

def matrixMultiplication(A, B):
    if A.shape[1] == B.shape[0]:
        C = np.zeros((A.shape[0], B.shape[1]), dtype=int)
        for row in range(A.shape[0]):
            for col in range(B.shape[1]):
                for i in range(A.shape[1]):
                    C[row, col] += A[row, i] * B[i, col]
        return C
    else:
        return "Matrix Multiplication is not possible"

def benchmark(A, B):
    start = time.time()
    result = matrixMultiplication(A, B)
    end = time.time()
    execution_time = (end - start) * 1000
    return execution_time

def write_to_csv(language, size, exec_time, mem_usage, cpu_usage):
    file_exists = os.path.isfile('../benchmark_results.csv')

    with open('../benchmark_results.csv', 'a') as f:
        if not file_exists:
            f.write("Language,Matrix Size,Execution Time,Memory Use,CPU use\n")

        f.write(f"{language},{size}x{size},{exec_time:.3f},{max(mem_usage):.2f},{cpu_usage:.3f}\n")

if __name__ == '__main__':
    sizes = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1024]
    for size in sizes:
        A = np.random.randint(1, 10, (size, size))
        B = np.random.randint(1, 10, (size, size))

        mem_usage = memory_usage((benchmark, (A, B)), interval=0.1)
        execution_time = benchmark(A, B)
        cpu_usage = psutil.cpu_percent(interval=1)

        print(f"Matrix size: {size}x{size}")
        print(f"Execution time: {execution_time:.3f} milliseconds")
        print(f"Peak memory usage: {max(mem_usage):.2f} MiB")
        print(f"CPU usage: {cpu_usage:.3f} %")
        print("------------------------")

        write_to_csv("Python", size, execution_time, mem_usage, cpu_usage)
