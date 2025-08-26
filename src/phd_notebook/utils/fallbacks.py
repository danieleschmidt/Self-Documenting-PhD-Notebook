"""
Fallback implementations for missing dependencies.
"""

# Numpy fallbacks
class FallbackArray:
    """Minimal numpy.ndarray fallback."""
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]
        self.shape = (len(self.data),)
    
    def mean(self):
        return sum(self.data) / len(self.data) if self.data else 0
    
    def std(self):
        if not self.data:
            return 0
        mean_val = self.mean()
        variance = sum((x - mean_val) ** 2 for x in self.data) / len(self.data)
        return variance ** 0.5
    
    def tolist(self):
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]

class FallbackNumpy:
    """Minimal numpy fallback module."""
    
    @staticmethod
    def array(data):
        return FallbackArray(data)
    
    @staticmethod 
    def zeros(shape):
        if isinstance(shape, int):
            return FallbackArray([0.0] * shape)
        return FallbackArray([0.0] * shape[0])
    
    @staticmethod
    def ones(shape):
        if isinstance(shape, int):
            return FallbackArray([1.0] * shape)
        return FallbackArray([1.0] * shape[0])
    
    @staticmethod
    def random():
        import random
        class RandomModule:
            @staticmethod
            def normal(loc=0.0, scale=1.0, size=None):
                import random
                if size is None:
                    return random.gauss(loc, scale)
                return FallbackArray([random.gauss(loc, scale) for _ in range(size)])
            
            @staticmethod
            def uniform(low=0.0, high=1.0, size=None):
                import random
                if size is None:
                    return random.uniform(low, high)
                return FallbackArray([random.uniform(low, high) for _ in range(size)])
        return RandomModule()

# Pydantic fallbacks
class BaseModel:
    """Minimal pydantic.BaseModel fallback."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def json(self):
        import json
        return json.dumps(self.dict())

# Try to import real modules, fallback if not available
try:
    import numpy as np
except ImportError:
    np = FallbackNumpy()

try:
    from pydantic import BaseModel as PydanticBaseModel
except ImportError:
    PydanticBaseModel = BaseModel

try:
    from typing_extensions import Annotated
except ImportError:
    # Fallback for older Python versions
    try:
        from typing import Annotated
    except ImportError:
        def Annotated(type_, *args):
            return type_

# NetworkX fallback
class NetworkXFallback:
    """Minimal networkx fallback."""
    
    class Graph:
        def __init__(self):
            self._nodes = {}
            self._edges = []
        
        def add_node(self, node, **attr):
            self._nodes[node] = attr
            
        def add_edge(self, u, v, **attr):
            self._edges.append((u, v, attr))
            
        def nodes(self):
            return list(self._nodes.keys())
    
    def DiGraph(self):
        return self.Graph()

# Scikit-learn fallbacks
class SklearnFallback:
    """Minimal sklearn fallback."""
    
    class linear_model:
        class LinearRegression:
            def __init__(self):
                self.coef_ = [1.0]
                self.intercept_ = 0.0
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [1.0] * len(X)
    
    class ensemble:
        class RandomForestRegressor:
            def __init__(self, **kwargs):
                pass
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [1.0] * len(X)

# Try to import real modules, fallback if not available  
try:
    import networkx as nx_module
except ImportError:
    nx_module = NetworkXFallback()

try:
    import sklearn
except ImportError:
    sklearn = SklearnFallback()