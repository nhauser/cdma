from matplotlib import pyplot
import cdma

dataset = cdma.open_dataset("file:demo.nxs")

print "scan group..."
scan_group = dataset["D1A_016_D1A"]
print scan_group.parent.location
print scan_group.location
print scan_group.name

print "read data ..."
print scan_group["duration"][...]

print "image group ..."
image_group = scan_group["image#50"]
print image_group.location
print image_group.name
print image_group.childs

print "data item ..."
field = image_group["data"]
print field.location
print field.is_group
print field.unit
print field.type
print field.shape
print field.size
data = field[...]
print data
