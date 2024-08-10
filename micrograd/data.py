from typing import Text

class Data:  
  def __init__(self, value, _children=(), _operator: Text = '', label: Text = ''):
    self.value = value
    self.grad = 0.0
    self._prev = set(_children)
    self._operator = _operator
    self.label = label

  def __repr__(self):
    return f"Data(value={self.value})"
  
  def __add__(self, other):
    # other = other if isinstance(other, Value) else Value(other)
    result = Data(value = self.value + other.value, _children = (self, other), _operator = '+')
    
    # def _backward():
    #   self.grad += 1.0 * out.grad
    #   other.grad += 1.0 * out.grad
    # out._backward = _backward
    
    return result

  def __mul__(self, other):
    # other = other if isinstance(other, Value) else Value(other)
    result = Data(value = self.value * other.value, _children = (self, other), _operator = '*')
    
    # def _backward():
    #   self.grad += other.data * out.grad
    #   other.grad += self.data * out.grad
    # out._backward = _backward
      
    return result