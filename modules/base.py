import numpy as np

class Module:
    """Base class for all neural network modules"""
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")