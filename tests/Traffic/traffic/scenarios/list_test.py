class A:
	def __init__(self,a):
		self.a = a

a=A(1)
b=A(2)
c=A(3)

abc = [a,b,c]

for i in abc[1:]:
	if i.a>2:
		i.a=0
		abc.remove(i)

for i in abc:
	print(i.a)