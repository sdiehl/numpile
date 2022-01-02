from numpile import autojit


@autojit
def test_const():
    return 114514


result = test_const()
print(result)
