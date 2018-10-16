from numpile import autojit
import numpy as np

@autojit
def dot(a, b):
    c = 0
    n = a.shape[0]
    for i in range(n):
       c += a[i]*b[i]
    return c

a = np.array(range(1000,2000), dtype='int32')
b = np.array(range(3000,4000), dtype='int32')

print(dot(a,b))

