import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit

OKGREEN = '\033[92m'
WARNING = '\033[93m'
BOLD = '\033[1m'
ENDC = '\033[0m'

running_time = dict()

def exec_time(name, print_data=False):
    def real_decorator(func):
        result = None

        def wrapper(n):
            t_start = timeit.default_timer()
            result = func(n)
            t_end = timeit.default_timer()

            elapsed_time = round((t_end - t_start) * 10**6, 4)
            
            result = f'and result {BOLD}{result}{ENDC}' if print_data else ''
            input_str = f'with input {WARNING}{n}{ENDC}' if print_data else ''

            print(f'Elapsed time {input_str} {result}: {OKGREEN}{elapsed_time}{ENDC} µs.')
            
            # save results for graphing
            if name in running_time.keys():
                running_time[name].append(elapsed_time)
            else:
                running_time[name] = list()
                running_time[name].append(elapsed_time)

            return result
        return wrapper
    return real_decorator

def plot_result(input: list):    
    names = np.array(list(running_time.keys()))
    times = np.array(list(running_time.values()))
    algorithm_num = 0

    plt.title('Running times per algorithm')
    plt.xlabel('Values')
    plt.ylabel('Time (µs)')

    for history in times:
        x_axis = np.array(input)
        y_axis = times[algorithm_num]

        plt.plot(x_axis, y_axis, label=names[algorithm_num])

        algorithm_num += 1

    plt.legend()
    plt.grid()
    plt.show()

@exec_time('quick_sort')
def quick_sort(arr: list) -> list:
    def partition(arr: list, left: int, right: int) -> int:
        pivot = arr[right]
        i = left - 1

        for j in range(left, right):
            if arr[j] <= pivot:
                i += 1

                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[right] = arr[right], arr[i + 1]

        return i + 1

    def qs(arr: list, left: int, right: int) -> None:
        pivot = partition(arr, left, right)

        qs(arr, left, pivot - 1)
        qs(arr, pivot + 1, right)

@exec_time('heap_sort')
def heap_sort(arr: list) -> list:
    def heapify(arr: list, n: int, i: int) -> None:
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and arr[i] < arr[l]:
            largest = l
        
        if r < n and arr[largest] < arr[r]:
            largest = r
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]

            heapify(arr, n ,largest)

    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]

        heapify(arr, i, 0)
    
    return arr

@exec_time('merge_sort')
def merge_sort(arr: list) -> list:
    def ms(arr: list) -> None:
        if len(arr) > 1:
            mid = len(arr) // 2
            left, right = arr[:mid], arr[mid:]

            ms(left)
            ms(right)

            i = j = k = 0

            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    arr[k] = left[i]
                    
                    i += 1
                else:
                    arr[k] = right[j]

                    j += 1
                
                k += 1
            
            while i < len(left):
                arr[k] = left[i]

                i += 1
                k += 1
            
            while j < len(right):
                arr[k] = right[j]

                j += 1
                k += 1

    ms(arr)

    return arr

@exec_time('insertion_sort')
def insertion_sort(arr: list) -> list:
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1

        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key

    return arr

# test_sizes = [10, 100, 1000, 10_000, 100_000, 1_000_000]
test_sizes = sorted(np.random.randint(1, 1_000, size=20))

def generate_inputs(size: int) -> tuple[list]:
    return (
        # np.arange(0, size, 1).tolist() + [size - 2], # just one switch required to sort
        np.random.randint(0, size, size=size).tolist(), # randomly arranged
        # list(reversed(np.arange(0, size, 1).tolist()))
    )

algorithms = {
    'quick_sort': quick_sort,
    'merge_sort': merge_sort,
    'heap_sort': heap_sort,
    # 'insertion_sort': insertion_sort,
}

def test(algorithms: dict['function']) -> None:
    for algo_name, algo in algorithms.items():
        print('\n')
        for test_case in test_sizes:
            print(f'Testing {BOLD}{algo_name}{ENDC} with array size {WARNING}{test_case}{ENDC}:')
            for array in generate_inputs(test_case):
                algo(array)

# print(insertion_sort([9, 2, 3, 4, 6, 5, 1]))

test(algorithms)
plot_result(test_sizes)