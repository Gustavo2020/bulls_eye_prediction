import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


import cudf
import cuml
import cupy as cp

# Test cuDF
import pandas as pd
df = cudf.DataFrame({'a': range(100000), 'b': range(100000)})
print('cuDF OK:', df.shape)

# Test cuML
from cuml.cluster import KMeans
X = cp.random.rand(10000, 10).astype(cp.float32)
km = KMeans(n_clusters=5)
km.fit(X)
print('cuML KMeans OK')

print('Todo listo!')
