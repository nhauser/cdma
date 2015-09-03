# Introduction #

This section is dedicated to existing CDMA engines for the Java version. It shortly presents what data format they are able to support.

Today some CDMA engines already exist:
  * [NetCDF](#NetCDF_engine.md)
  * [NeXus](#NeXus_engine.md)

# Engines #

## NetCDF engine ##

Based on the [NetCDF API](http://www.unidata.ucar.edu/software/netcdf/) this engine supports various file format.

### Benefits ###

This API is fully java implemented. No dependencies to any native code is required. It supports a large number of file formats.

## NeXus engine ##

Based on the [NeXus API](http://www.nexusformat.org/) this engine supports the NeXus format.

### Benefits ###

Some enhancements have been added to the original NeXus API. The native API has been wrapped in a higher library. The code developed by SOLEIL is contained in the package _fr.soleil.nexus_.

#### Thread safety ####

The underlying NeXus API wasn't thread safe due to the use of native C++ library. It has been encapsulated in a higher library that prevents multiple access at a time.

The **CDMA NeXus engine is thread safe**. All read / write operations are canonical and synchronized using a semaphore.

#### Navigation optimizations ####

The navigation system uses a buffer. The original API had some slowness troubles with very populated nodes. For example a group having hundreds of children, was long to answer each time a request was asked. The cache system used in the enhanced API permits to avoid it. The first request might be a bit long, but all others will be processed almost instantly without accessing the NeXus file.

#### Data loading optimizations ####

The data loading systems has been enhanced too. Only the requested portion of a matrix is loaded at a time. It means if you are interested about an image, of 1024 x 1024 in a 2D stack lets say 150 x 400, that image only will be loaded and not the whole matrix of 150 x 400 x 1024 x 1024.

The loading is done at the request of the first storage's bit. You can have the full description of an array without having loaded a single bit from the matrix.

A soft reference mechanism is also used permiting to limit the use of the Java heap space. Each array can reload its matrix if it has been cleared due to high memory usage.

### Download ###

Project is available [here](http://code.google.com/p/cdma/source/browse/#svn%2Fjava%2Fbranches%2Fsoleil%2Forg.gumtree.data.engine.jnexus).