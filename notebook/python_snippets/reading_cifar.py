# data_batch_1 = rds.read_cifar_dataset("data\cifar\data_batch_1")
# print("Length db1: {0}".format(len(data_batch_1)))
# print("Keys: {0}".format(data_batch_1.keys()))
# print("Db1[b'data']: {0}".format(data_batch_1[b'data']))
# print("Shape: {0}".format(data_batch_1[b'data'].shape))
# print("First image: {0}".format(data_batch_1[b'data'][0]))

# each image loaded as 1x3072 as [1024xR][1024xG][1024xB]
#                                   32      32      32      
# reshape to 32x32
# i.e stack rgb intop and trasnpose value to RGB pixel data
# transpose to rgb 32x32z3
#image = data_batch_1[b'data'][2]

# 32x32 for r
# ...       g
#image32x32GridsOfRGB = image.reshape(3,32,32)
#imageRgbPixelData = image32x32GridsOfRGB.transpose(1,2,0)

#plt.imshow(imageRgbPixelData)
#plt.show()
#reshape the whole dataset
# data = data_batch_1[b'data']
# data = data.reshape(len(data), 3, 32, 32).transpose(0,2,3,1)
# print("Whole ds shape: ", data.shape)

# #visualize multiple images
# for i in range(16):
#     plt.subplot(4, 4, i+1)
#     plt.imshow(data[i])
# plt.show()

# columns, rows = 5, 4
# fig = plt.figure(figsize=(8,8))
# for i in range(1, columns*rows + 1):
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(data[i])
# plt.show()
