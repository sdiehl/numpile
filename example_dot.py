import numpy as np
from numpile import autojit


@autojit
def dot(a, b):
    c = 0
    n = a.shape[0]
    for i in range(n):
        c += a[i] * b[i]
    return c


a = np.arange(100, 200, dtype="int32")
b = np.arange(300, 400, dtype="int32")
result = dot(a, b)
print(result)
