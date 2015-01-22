Numpile
-------

A tiny 1000 line LLVM-based numeric specializer for scientific Python code.

You really shouldn't use this for anything serious, it's just to demonstrate how
you might build one of these things from scratch. There's a lot of untapped
potential and low hanging fruit around *selective embedded JIT specialization*
for array expression languages in the SciPython space.

Installing
----------

Just install [Anaconda][Anaconda](https://store.continuum.io/cshop/anaconda/)
and it will work out of box.

Or if you like suffering then install LLVM from your package manager and
then have fun installing ``llvmpy`` and ``numpy`` from source.

```bash
$ pip install llvmpy
$ pip install numpy
```

Usage
-----

```python
from numpile import autojit

@autojit
def dot(a, b):
    c = 0
    n = a.shape[0]
    for i in range(n):
       c += a[i]*b[i]
    return c

a = np.array(range(1000,2000), dtype='int32')
b = np.array(range(3000,4000), dtype='int32')

print dot(a,b)
```

License
-------

Released under the MIT License.
Copyright (c) 2015, Stephen Diehl
