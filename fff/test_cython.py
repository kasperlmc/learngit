import pyximport; pyximport.install()

import test_cython_1

print(test_cython_1.fib(100))
