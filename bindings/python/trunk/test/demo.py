from matplotlib import pyplot
import numpy
import cdma

def print_attribute(attr):
    ostr = "attribute ["+attr.name + "]:"
    ostr += " size=%i," %(attr.size)
    ostr += " type=%s," %(attr.type)
    ostr += " shape=%s," %(attr.shape.__str__())
    ostr += " value=%s" %(attr[...].__str__())

    print ostr 

def print_item(item):
    ostr = "dataitem ["+item.name+"]:"
    ostr += " type=%s," %(item.type)
    ostr += " rank=%i," %(item.rank)
    ostr += " shape=%s," %(item.shape.__str__())
    ostr += "\ndata = \n%s" %(item[...].__str__())

    print ostr

dataset = cdma.open_dataset("file:../data/demo.nxs")

print "scan group..."
scan_group = dataset["D1A_016_D1A"]
for i in scan_group.items:
    print i.name,i.type,i.size

print_attribute(scan_group.attrs["name"])

print scan_group.dims

print "read data ..."
#print scan_group["duration"][...]

print "image group ..."
image_group = scan_group["image#20"]

print "data item ..."
data= image_group["data"]
print_item(data)
#print data[0:100:2,50:1000:20]

for a in data.attrs:
    print a

pyplot.figure()
pyplot.imshow(numpy.log10(data[...]))
pyplot.figure()
pyplot.imshow(numpy.log10(data[:,512:]))
pyplot.figure()
pyplot.imshow(numpy.log10(data[:,:512]))
pyplot.show()
