#!/usr/bin/env python
#simple example for simple data reduction

import cdma
from matplotlib import pyplot
import numpy

#open the dataset
ds = cdma.open_dataset("file:demo.nxs")
dg = ds.root_group["D1A_016_D1A"]
expid = dg["experiment_identifier"][...]

itot = []
time   = []
t0 = 0

for i in range(60):
    gn = "image#%i" %(i)
    print "processing image ",gn,"..."
    ig = dg[gn]
    itot.append(ig["data"][840:960,320:440].sum())
    if not i: t0 = ig["timestamp"][...]
    time.append(ig["timestamp"][...])


pyplot.figure()
pyplot.plot(numpy.array(time)-t0,numpy.array(itot))
pyplot.title(expid + " total intensity")
pyplot.xlabel("time in (%s)" %(ig["timestamp"].unit))
pyplot.show()
