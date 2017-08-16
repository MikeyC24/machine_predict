def test_math(a,b):
	y = a+b
	z = a-b
	return y,z

vars = test_math(10,5)
print(vars)
a = vars
v = vars[0]

print(a)
print(v)