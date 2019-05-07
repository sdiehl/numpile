from numpile import autojit


@autojit
def add(a, b):
    return a + b


a = 3.1415926
b = 2.7182818
result = add(a, b)
print(result)
