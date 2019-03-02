from __future__ import print_function
import numpy as np

from numpile import autojit


@autojit
def dot(a, b):
    c = 0
    n = a.shape[0]
    for i in range(n):
       c += a[i] * b[i]
    return c


a = np.array(range(100, 200), dtype='int32')
b = np.array(range(300, 400), dtype='int32')

print('Result:', dot(a, b))
