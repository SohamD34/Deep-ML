# Autograd is short for "automatic differentiation", a technique to automatically compute gradients (derivatives) of functions, especially those represented as computational graphs.
# 
# In deep learning, models are often represented as a sequence of operations (addition, multiplication, activation functions, etc.) forming a computation graph.
# Each node in the graph represents a value and tracks how it was computed from its inputs (children).
# 
# The key idea is to use the chain rule from calculus to propagate gradients backward through the graph.
# For a function f(x, y, ...), the gradient of the output with respect to each input is computed by recursively applying the chain rule:
# 
#   If z = f(x, y), then:
#     dz/dx = ∂f/∂x
#     dz/dy = ∂f/∂y
# 
# When combining multiple operations, the chain rule allows us to compute the derivative of the output with respect to any input by multiplying the local derivatives along the path from output to input.
# 
# In code, this is implemented by:
#   1. Building a computation graph as operations are performed.
#   2. Storing how each value was computed (its "parents" and the operation).
#   3. When backward() is called, traversing the graph in reverse topological order and applying the chain rule to accumulate gradients.
# 
# This enables efficient and accurate gradient computation for complex models, which is essential for training neural networks using gradient-based optimization.



class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out



    def __mul__(self, other):

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out


    def relu(self):

        out = Value(self.data if self.data > 0 else 0, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out


    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()


# Example usage
a = Value(2)
b = Value(-3)
c = Value(10)
d = a + b * c
e = d.relu()
e.backward()
print(a, b, c, d, e)