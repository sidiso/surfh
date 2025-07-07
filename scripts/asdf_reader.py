import numpy as np
import asdf 
import matplotlib.pyplot as plt

asdf_file = '/home/nmonnier/Downloads/MAST_2025-06-30T12_04_42.790Z/JWST/jwst_miri_regions_0136.asdf'
with asdf.open(asdf_file) as af:
    print(f"Opening ASDF file: {asdf_file}")
    af.info()
    a = af["meta"]["exposure"]
    print(a)
    print(af["meta"]["instrument"])
    b = af["regions"]
    print(b.shape)
    plt.imshow(b[1]-301, cmap='gray')
    plt.colorbar()
    plt.show()



# asdf_file = '/home/nmonnier/Downloads/MAST_2025-06-30T12_04_42.790Z/JWST/jwst_miri_distortion_0047.asdf'
# with asdf.open(asdf_file) as af:
#     print(f"Opening ASDF file: {asdf_file}")
#     af.info()
#     # a = af["meta"]["exposure"]
#     # print(a)
#     # print(af["meta"]["instrument"])
#     # b = af["regions"]
#     # print(b.shape)
#     # plt.imshow(b[1]-101, cmap='gray')
#     # plt.colorbar()
#     # plt.show()
#     a = af['meta']['exposure']
#     print(a)
#     print("-----------------")
#     print(af['meta']['model_type'])
#     print("-----------------")
#     # print(af['model'])

# print("#################################")

# asdf_file = '/home/nmonnier/Downloads/MAST_2025-06-30T12_04_42.790Z/JWST/jwst_miri_specwcs_0136.asdf'
# with asdf.open(asdf_file) as af:
#     print(f"Opening ASDF file: {asdf_file}")
#     af.info()
#     print(af['model'])
#     print("-----------------")
# print("##################################")

# asdf_file = '/home/nmonnier/Downloads/MAST_2025-06-30T12_04_42.790Z/JWST/jwst_miri_filteroffset_0008.asdf'
# with asdf.open(asdf_file) as af:
#     print(f"Opening ASDF file: {asdf_file}")
#     af.info()
#     print(af['meta'])