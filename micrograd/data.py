import math
from typing import Text

class Data:
    def __init__(self, value, _children=(), _operator: Text = '', label: Text = ''):
        self.value = value
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._operator = _operator
        self.label = label
    
    def __repr__(self):
        return f"Data(value={self.value})"
      
    def __add__(self, other):
        other = other if isinstance(other, Data) else Data(other)
        result = Data(value = self.value + other.value, _children = (self, other), _operator = '+')
        
        def _backward():
          self.grad += 1.0 * result.grad
          other.grad += 1.0 * result.grad
        result._backward = _backward
        
        return result
    
    def __mul__(self, other):
        other = other if isinstance(other, Data) else Data(other)
        result = Data(value = self.value * other.value, _children = (self, other), _operator = '*')
        
        def _backward():
          self.grad += other.value * result.grad
          other.grad += self.value * result.grad
        result._backward = _backward
          
        return result

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        result = Data(self.value**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.value ** (other - 1)) * result.grad

        result._backward = _backward
    
        return result
  
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other

    def tanh(self):
        x = self.value
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        result = Data(t, (self, ), 'tanh')
    
        def _backward():
            self.grad += (1 - t**2) * result.grad
        result._backward = _backward
    
        return result

    def exp(self):
        x = self.value
        result = Data(math.exp(x), (self, ), 'exp')
        
        def _backward():
          self.grad += result.value * result.grad

        result._backward = _backward
        
        return result

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
        
        self.grad = 1.0
        for node in reversed(topo):
          node._backward()