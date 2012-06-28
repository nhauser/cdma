from matplotlib import pyplot
import numpy
import cdma
import h5py

h5 = h5py.File("demo.nxs")
h5data = h5["/D1A_016_D1A/image#20/data"][...]
print h5data.dtype
h5.close()

dataset = cdma.open_dataset("file:demo.nxs")

print "scan group..."
scan_group = dataset["D1A_016_D1A"]
print "attribute size: ",scan_group.attrs["name"].size
print "attribute name: ",scan_group.attrs["name"].name
print "attribute type: ",scan_group.attrs["name"].type
print "attribute data: ",scan_group.attrs["name"][...]

print "read data ..."
print scan_group["duration"][...]

print "image group ..."
image_group = scan_group["image#20"]

print "data item ..."
data= image_group["data"]
print data.shape
print data
print data.type

pyplot.subplot(121)
pyplot.imshow(numpy.log10(data[...]))
pyplot.subplot(122)
pyplot.imshow(numpy.log10(h5data))
pyplot.show()
