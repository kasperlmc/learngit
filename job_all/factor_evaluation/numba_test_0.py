from numba import jit,int64,float64
import numpy as np
import time
import sys


class TailRecurseException(Exception):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


def tail_call_optimized(g):
    """
    This function decorates a function with tail call
    optimization. It does this by throwing an exception
    if it is it's own grandparent, and catConnected to pydev debugger (build 183.4284.139)

Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)ching such
    exceptions to fake the tail call optimization.

    This function fails if the decorated
    function recurses in a non-tail context.
    """

    def func(*args, **kwargs):
        f = sys._getframe()
        # 为什么是grandparent, 函数默认的第一层递归是父调用,
        # 对于尾递归, 不希望产生新的函数调用(即:祖父调用),
        # 所以这里抛出异常, 拿到参数, 退出被修饰函数的递归调用栈!(后面有动图分析)
        if f.f_back and f.f_back.f_back and f.f_back.f_back.f_code == f.f_code:
            # 抛出异常
            raise TailRecurseException(args, kwargs)
        else:
            while 1:
                try:
                    return g(*args, **kwargs)
                except TailRecurseException as e:
                    args = e.args
                    kwargs = e.kwargs
    func.__doc__ = g.__doc__
    return func


@jit()
def test_func(n):
    array_list = np.zeros(100)
    for i in range(n):
        temp = i*20
        array_list[i] = temp
    return array_list


@jit(nopython=True)
def test_func_1(n):
    array_list = np.zeros(n)
    for i in range(n):
        temp = i**0.5+i*(i-1)
        array_list[i] = temp
    return array_list


@jit()
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n*factorial(n-1)

@tail_call_optimized
# @jit()
def test_func_2(n,array_list=np.zeros(10000)):
    if n==10000:
        return array_list
    array_list[n] = n**0.5+n*(n-1)
    return test_func_2(n+1,array_list)


@jit(nopython=True)
def go_fast(a):  # Function is compiled and runs in machine code
    trace = 0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

#
# x = np.arange(100).reshape(10, 10)
# # print(x)
# start = time.time()
# # print(test_func_1(100000000))
# print(test_func_2(1)[-1])
# end = time.time()
# print("Elapsed (with compilation) = %s" % (end - start))


print(max(1,2))



print(0.5+10/200)


if 0 or (1 and 0):
    print("y")
else:
    print("no")

a=[0,1,2,3,4]
a=np.array(a)
print(np.append(a,1))
a=np.append(a,1)
print(a)
a=np.append(a,3)
print(a)
