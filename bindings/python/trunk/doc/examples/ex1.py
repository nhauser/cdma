#!/usr/bin/env python
#simple example for opening a dataset

import cdma

#open the dataset
ds = cdma.open_dataset("file:demo.nxs")

#iterate over all group below the root group
for g in ds.root_group:

    #iterate over all groups
    for group in cdma.get_groups(g):
        print group

    #iterate over all items
    for item in cdma.get_dataitems(g):
        print item

    #iterate over all dimensions
    for dimension in cdma.get_dimensions(g):
        print dimension

#get some basic information about the recorded data
g = ds.root_group["D1A_016_D1A"]
print g["experiment_identifier"][...]
print g["duration"][...]
print g["start_time"][...]
print g["end_time"][...]

