import h5py

#HDF5的读取：  
f = h5py.File('TCR_encoder_30.h5','r')   #打开h5文件
#可以查看所有的主键  
for key in f.keys():      
	print(f[key].name)      
	#print(f[key].shape)
	print(f[key].value)
