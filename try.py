import numpy as np

data = np.array([[1,2,3,4,5], [6,7,8,9,10], [2,4,6,8,10]])

cent = np.array([[1,2,3,4,5],
              [2,3,4,9,9]])

res = []
for i in range(len(cent)):
    a = np.square(data-cent[i])
    b = np.sum(a,axis=1)
    res.append(np.sqrt(b))
res = np.transpose(np.array(res))
print(res)

clustering = np.zeros(data.shape[0])

clustering = res.argmin(axis=1)
print(clustering)