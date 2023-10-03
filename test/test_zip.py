import numpy as np
import itertools

image = np.arange(3*400*400, dtype=np.int32).reshape((3,400,400))
im_size = 400*400
One_size = 400

a = np.arange(3*15).reshape(3,15)
b = np.arange(3*15).reshape(3,15)*2
aa = np.arange(3*15).reshape(3,15)*10
bb = np.arange(3*15).reshape(3,15)*20

nWave = 3
vec_wave = np.repeat(np.arange(nWave), 5)
vec_wave = vec_wave[np.newaxis:]
vec_x_values = np.tile(np.arange(5), [1,3]) +5
vec_y_values = np.tile(np.arange(5), [1, 3]) +10

test = np.zeros((3,15))

a[0] = vec_wave
aa[0] = vec_wave
b[0] = vec_wave
bb[0] = vec_wave

a[1] = vec_x_values
aa[1] =vec_x_values*10
b[1] = vec_x_values*2
bb[1] =vec_x_values*20

a[2] = vec_y_values
aa[2] =vec_y_values*10
b[2] = vec_y_values*2
bb[2] =vec_y_values*20



###########################################
############# Basic  Scipy ################
###########################################
c = zip(a,bb)
cc = zip(aa,b)
hyper = itertools.product(*zip(c, cc))
cube1 = np.zeros((8,15), dtype=np.int32)
cube2 = np.zeros_like(cube1, dtype=np.int32)
i=0
cube_weight1 = np.zeros((8,15), dtype=np.int32)
cube_weight2 = np.zeros((8,15), dtype=np.int32)

value1 = np.zeros(15)
value2 = np.zeros(15)
value3 = np.zeros(15)

for h in hyper:
    edge_indices, weights = zip(*h)
    weight = np.array([1.])
    for w in weights:
        weight = weight * w
        if i == 7:
            print(w)
    ref = image[edge_indices]
    i+=1
    term = ref*weight
    #print(f"Pyt : {ref} x {weight} = {term}")
    value1 += term




###########################################
############# First Optim  ################
###########################################
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
    print("Weight 2 = ", weight)
    
    ref2 = np.zeros(15, dtype=image.dtype)
    for j in range(size_j):
        idx =   new_a[id0*3*size_j + j]*im_size +\
                new_a[id1*3*size_j + size_j + j]*One_size +\
                new_a[id2*3*size_j + 2*size_j + j]

        ref2[j] = image.ravel()[idx]
    term = ref2*weight
    #print(f"Cyt : {ref2} x {weight} = {term}")
    value2 += term



###########################################
############# Second Optim ################
###########################################
na = np.arange(2*5).reshape(2,5)
nb = np.arange(2*5).reshape(2,5)*2
naa = np.arange(2*5).reshape(2,5)*10
nbb = np.arange(2*5).reshape(2,5)*20

nWave = 3
vec_x_values = np.arange(5) +5
vec_y_values = np.arange(5) +10

na[0] = vec_x_values
nb[0] = vec_x_values*2
naa[0] = vec_x_values*10
nbb[0] = vec_y_values*20

na[1] = vec_y_values
nb[1] = vec_y_values*2
naa[1] = vec_y_values*10
nbb[1] = vec_y_values*20

#####
nravel_a = na.ravel()
nravel_aa = naa.ravel()
nravel_b = nb.ravel()
nravel_bb = nbb.ravel()

newnew_a = np.concatenate((nravel_a, nravel_aa))
newnew_b = np.concatenate((nravel_bb, nravel_b))
nsize_j = na.shape[1]


for h in reversed(range(4)):
    weight = np.ones(nsize_j)
    id0 = (h&0b001) >> 0
    id1 = (h&0b010) >> 1
    id2 = (h&0b100) >> 2

    for j in range(nsize_j):
        weight[j] = newnew_b[id0*2*nsize_j + j] *\
                    newnew_b[id1*2*nsize_j + nsize_j + j]

    print("Weight 3 = ", weight)


    ref3 = np.zeros((nWave,nsize_j), dtype=image.dtype)
    for j in range(nsize_j):
        idx =   newnew_a[id0*2*nsize_j + j]*One_size +\
                newnew_a[id1*2*nsize_j + nsize_j + j]
        
        ref3[:,j] = image[:,newnew_a[id0*2*nsize_j + j], newnew_a[id1*2*nsize_j + nsize_j + j]]
    #term3 = np.array([0.])
    term3 = ref3.ravel()*np.tile(weight, nWave)
    #print(f"Cyt : {ref2} x {weight} = {term}")
    value3 += term3


print(value1 == value2)