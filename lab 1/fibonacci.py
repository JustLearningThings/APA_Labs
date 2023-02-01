import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit

OKGREEN = '\033[92m'
WARNING = '\033[93m'
BOLD = '\033[1m'
ENDC = '\033[0m'

running_time = dict()
values = [1, 5, 10, 15, 20, 25, 30, 35]

def exec_time(name):
    def real_decorator(func):
        result = None

        def wrapper(n):
            t_start = timeit.default_timer()
            result = func(n)
            t_end = timeit.default_timer()

            elapsed_time = round((t_end - t_start) * 10**6, 4)
            print(f'Elapsed time with input {WARNING}{n}{ENDC} and result {BOLD}{result}{ENDC}: {OKGREEN}{elapsed_time}{ENDC} µs.')
            
            # save results for graphing
            if name in running_time.keys():
                running_time[name].append(elapsed_time)
            else:
                running_time[name] = list()
                running_time[name].append(elapsed_time)

            return result
        return wrapper
    return real_decorator

def exec_time_recursive(func, name, *args, **kwargs):
    t_start = timeit.default_timer()
    func(args[0])
    t_end = timeit.default_timer()

    elapsed_time = round((t_end - t_start) * 10**6, 4)
    print(f'Elapsed time with input {WARNING}{args[0]}{ENDC}: {OKGREEN}{elapsed_time}{ENDC} µs.')

    # save results for graphing
    if name in running_time.keys():
        running_time[name].append(elapsed_time)
    else:
        running_time[name] = list()
        running_time[name].append(elapsed_time)

def plot_result():
    names = np.array(list(running_time.keys()))
    times = np.array(list(running_time.values()))
    algorithm_num = 0

    plt.title('Running times per algorithm')
    plt.xlabel('Values')
    plt.ylabel('Time (µs)')

    for history in times:
        x_axis = np.array(values)
        y_axis = times[algorithm_num]

        plt.plot(x_axis, y_axis, label=names[algorithm_num])

        algorithm_num += 1

    plt.legend()
    plt.grid()
    plt.show()

def recursive_fib(n) -> int:
    return n if n < 2 else recursive_fib(n - 1) + recursive_fib(n - 2)

@exec_time('iterative')
def iterative_fib(n: int) -> int:
    i, j = 0, 0

    for k in range(n):
        j = i + j
        i = j - i
    
    return j

@exec_time('iterative_memoization')
def iterative_fib_with_memoization(n: int) -> int:
    fib = [0, 1]

    for i in range(2, n + 1):
        fib.append(fib[i-1] + fib[i - 2])
    
    return fib[n]

@exec_time('eigen')
def eigen_fib(n: int) -> int:
    f1 = np.array(([1, 1], [1, 0]))
    eigenvalues, eigenvectors = np.linalg.eig(f1)
    fn = eigenvectors @ np.diag(eigenvalues ** n) @ eigenvectors.T

    return int(np.rint(fn[0, 1]))

# with implicit exponentiation
@exec_time('eigen_optimized')
def eigen_fib_optimized(n: int) -> int:
    multiply = lambda a, b, x, y: (x*(a + b) + a*y, a*x + b*y)
    square = lambda a, b: ((a*a) + ((a*b) << 1), a*a + b*b)
    def power(a, b, m):
        n = 2

        if m == 0:
            return (0, 1)
        elif m == 1:
            return (a, b)
        else:
            x, y = a, b
            
            while n <= m:
                x, y = square(x, y)
                n = n*2
            
            a, b = power(a, b, m-n//2)

            return multiply(x, y, a, b)
    
    res, _ = power(1, 0, n)

    return res

@exec_time('golden_ratio')
def golden_ratio_fib(n: int) -> int:
    # using Binet's formula
    SQRT_OF_FIVE = 2.23606797749979 # cached result (calculated with np.sqrt(5))
    PHI = (1 + SQRT_OF_FIVE) / 2
    INV_PHI = (1 - SQRT_OF_FIVE) / 2

    return int((PHI**n - INV_PHI**n) / SQRT_OF_FIVE)

print('1. Recursive:')
for value in values:
    exec_time_recursive(recursive_fib, 'recursive', value)

print('\n2. Iterative:')
for value in values:
    iterative_fib(value)

print('\n3. Iterative with memoization:')
for value in values:
    iterative_fib_with_memoization(value)

print('\n4. Using eigenvectors:')
for value in values:
    eigen_fib(value)

print('\n5. Using eigenvectors (optimized):')
for value in values:    
    eigen_fib_optimized(value)

print('\n6. Golden Ratio:')
for value in values:
    golden_ratio_fib(value)

# find the best one
mean_times = running_time.copy()

for key in mean_times.keys():
    mean_times[key] = np.mean(np.array(mean_times[key]))

mean_times = {k: v for k, v in sorted(mean_times.items(), key=lambda item: item[1])}

print('\nBest algorithms (for fibbonacci 1-35):')
for i, key in enumerate(mean_times.keys()):
    print(f'{i + 1}: {BOLD}{key}{ENDC} with an average of {OKGREEN}{mean_times[key]}{ENDC} µs.')

# graphs
plot_result()
del running_time['recursive']
plot_result()
del running_time['eigen']
plot_result()