#!/usr/bin/env python
#simple example for ploting multidimensional data

import cdma
from matplotlib import pyplot
import numpy

#open the dataset
ds = cdma.open_dataset("file:demo.nxs")
dg = ds.root_group["D1A_016_D1A"]
expid = dg["experiment_identifier"][...]

ig = dg["image#20"]
data = ig["data"][...]
timestamp = ig["timestamp"].attrs["timestamp"][...]

pyplot.figure()
pyplot.contourf(numpy.log10(data),40)
pyplot.title(expid + "@" +timestamp)
pyplot.savefig("test.png",dpi=300)

