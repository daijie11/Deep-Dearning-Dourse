Python 3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import math
>>> def quadratic(a,b,c):
	a = float(a)
	b = float(b)
	c = float(c)
	if a == 0:
		x = -c/b
		return x
	elif b*b-4*a*c < 0:
		return None
	else:
		x1 = math.sqrt(b*b-4*a*c)/2*a - b/2*a
		x2 = -math.sqrt(b*b-4*a*c)/2*a - b/2*a
		return x1,x2
	quadratic(1,3,2)
	(-1.0,-2.0)
	quadratic(1,3,-4)
	(1.0,4.0)
	quadratic(0,2,4)
	-2.0
