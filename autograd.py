# class Value:
# 	def __init__(self, data, _children=(), _op=''):
# 		self.data = data
# 		self.grad = 0
# 		self._backward = lambda: None
# 		self._prev = set(_children)
# 		self._op = _op
		
# 	def __repr__(self):
# 		return Value(data={self.data})

# 	def __add__(self, other):
# 		return Value(data={self.data + other})

# 	def __mul__(self, other):
# 		return Value(data={self.data * other.data})

# 	def relu(self):
# 		pass

# 	def backward(self):
# 		# Implement backward pass here
# 		pass
	


# a = Value(2)
# b = Value(-3)
# c = Value(10)
# d = a + b * c
# # e = d.relu()
# # e.backward()
# print(a, b, c, d)