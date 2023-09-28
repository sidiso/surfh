import numpy as np
import itertools

image = np.arange(400*400*400, dtype=np.int32).reshape((400,400,400))
im_size = 400*400
One_size = 400

a = np.arange(3*5).reshape(3,5)
b = np.arange(3*5).reshape(3,5)*2
aa = np.arange(3*5).reshape(3,5)*10
bb = np.arange(3*5).reshape(3,5)*20

c = zip(a,bb)
cc = zip(aa,b)
hyper = itertools.product(*zip(c, cc))
cube1 = np.zeros((8,5), dtype=np.int32)
cube2 = np.zeros_like(cube1, dtype=np.int32)
i=0
cube_weight1 = np.zeros((8,5), dtype=np.int32)
cube_weight2 = np.zeros((8,5), dtype=np.int32)


value1 = np.zeros(5)
value2 = np.zeros(5)

for h in hyper:
    edge_indices, weights = zip(*h)
    weight = np.array([1.])
    for w in weights:
        weight = weight * w
    ref = image[edge_indices]
    i+=1
    term = ref*weight
    print(f"Pyt : {ref} x {weight} = {term}")
    value1 += term

ravel_a = a.ravel()
ravel_aa = aa.ravel()
ravel_b = b.ravel()
ravel_bb = bb.ravel()

new_a = np.concatenate((ravel_a, ravel_aa))
new_b = np.concatenate((ravel_bb, ravel_b))
size_j = a.shape[1]


for h in reversed(range(8)):
    weight = np.ones(size_j)
    id0 = (h&0b001) >> 0
    id1 = (h&0b010) >> 1
    id2 = (h&0b100) >> 2

    for j in range(size_j):
        weight[j] = new_b[id0*3*size_j + j] *\
                    new_b[id1*3*size_j + size_j + j] *\
                    new_b[id2*3*size_j + 2*size_j + j]
    
    ref2 = np.zeros(5, dtype=image.dtype)
    for j in range(size_j):
        idx =   new_a[id0*3*size_j + j]*im_size +\
                new_a[id1*3*size_j + size_j + j]*One_size +\
                new_a[id2*3*size_j + 2*size_j + j]

        ref2[j] = image.ravel()[idx]
    term = ref2*weight
    print(f"Cyt : {ref2} x {weight} = {term}")
    value2 += term


print(value1 == value2)