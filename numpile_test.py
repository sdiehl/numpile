#!/usr/bin/env python
#coding=utf-8

import numpy as np

from numpile import autojit

@autojit
def dot_vectorize(a, b):
    c = 0
    n = a.shape[0]
    for i in range(n):
       c += a[i]*b[i]
    return c

a = np.arange(1000,2000, dtype='int64')
b = np.arange(3000,4000, dtype='int64')

assert dot_vectorize(a,b) == np.dot(a,b)
