from matplotlib import pyplot
import numpy
import cdma


dataset = cdma.open_dataset("file:../data/demo.nxs")
rg = dataset.root_group

print "scan group..."
scan_group = rg["D1A_016_D1A"]
for i in scan_group.items:
    print i.name,i.type,i.size

print scan_group.attrs["name"]

print scan_group["start_time"][...]
print scan_group["end_time"][...]

print "read data ..."
#print scan_group["duration"][...]

print "image group ..."
image_group = scan_group["image#20"]

print "data item ..."
data= image_group["data"]
print data

for a in data.attrs:
    print a

pyplot.figure()
pyplot.contourf(numpy.log10(data[...]),40)
pyplot.figure()
pyplot.contourf(numpy.log10(data[512:,:512]),40)
pyplot.show()
