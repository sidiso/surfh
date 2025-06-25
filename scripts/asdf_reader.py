import numpy as np
import asdf 
import matplotlib.pyplot as plt

asdf_file = '/home/nmonnier/Downloads/jwst_miri_regions_0137.asdf'

with asdf.open(asdf_file) as af:
    af.info()
    a = af["meta"]["exposure"]
    print(a)
    print(af["meta"]["instrument"])
    b = af["regions"]
    print(b.shape)
    plt.imshow(b[1]-101, cmap='gray')
    plt.colorbar()
    plt.show()
