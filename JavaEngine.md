# Content #

  * [Introduction](#Introduction.md)
  * [Main parts of an engine](#Main_parts_of_an_engine.md)
    * [Handle the data source](#Handle_the_data_source.md)
    * [Navigation](#Navigation.md)
    * [Data description and manipulation](#Data_description_and_manipulation.md)
      * [Data description](#Data_description.md)
      * [Data manipulation](#Data_manipulation.md)

# Introduction #

---

A CDMA engine is used to access on a particular data format like NeXus, NetCDF, a data base or what ever. Indeed it manages data source handle, navigation, and access to data. It can't be instantiated, as it is, by the Core API: it's not a plug-in at all. The engine only cares about:
  * handling the data source (opening and closing its target URI)
  * navigation
  * data loading

The interest of developing engines is that several plug-ins can share the same one. It reduces a lot the development cost. To prevent the plug-in reimplementing every methods of each class, items created by the engine must respect the CDMA interfaces. Indeed a plug-in above an engine can then directly drive the engine or even return engine's objects.

The engine developer should focus on efficiency and memory usage.

# Main parts of an engine #

---


All the things an engine should implement are stored in the Core package _interfaces_: `org.cdma.interfaces`.

Notice that one particular interface shouldn't be implemented directly because it is inherited yet from other interfaces : _IContainer_. Its role is to give the property of a physical node of the data source to prevent rewriting many methods many times.

Below an example of the CDMA NeXus engine structure:

![http://cdma.googlecode.com/svn/wiki/images/java_engine_nexus_structure.png](http://cdma.googlecode.com/svn/wiki/images/java_engine_nexus_structure.png)

The engine is split among 3 main packages:
  * _fr.soleil.nexus_ which is the NeXus API by this engine (it enhances array loading, memory and disk usage)
  * _navigation_ that regroups all concerning the navigation and data source handling
  * _array_ that manages matrices loading and description

## Handle the data source ##

The data source is manipulated using a CDMA engine. According so that's the engine whose handles it and ask for read / write access. The data source handler is done by implementing the _IDataset_ interface. It provides open and close methods, a generic write methods.

Above all, it's the _IDataset_ interface which give access to the navigation part using the _getRootGroup_ method. That call returns the entry point of the data source which is mandatory a _IGroup_ implementation.


## Navigation ##

The data source is considered as a tree. But it's not mandatory: if you can do the big things, you can do the little things as well.

The aim is to be able to browse the data source and get nodes' informations, location, name, properties...

Here 5 classes need to be implemented: `IDataset, IGroup, IDataItem, IAttribute`.

  * `IDataset` is the one that handles the data source it permits to have a location on it, to open  or close an access on it,
  * `IGroup` which is a container for IDataItem or other `IGroup`. It's the one that let you exploring the tree
  * `IDataItem` which are considered as the leaves of our tree. They wear the physical data (more often a matrix), so it should be able to give some informations on it like type, size...
  * `IAttribute` are attribute or contextual data of nodes (`IGroup` or `IDataItem`). For instance, an `IDataItem` has timestamps which tells at what time it was recorded.
  * `IDimension` they are the axis of a data item. For instance, it should store motor positions when a detector that produced a stack of spectrum has been translated while acquiring it.

With such a structure we should be able to explore almost all (to not say all) data sources.

**Take care to share references on object.**

Example 1:
```
IDataset dataset = Factory.createDatasetInstance(URI);
IGroup root = dataset.getRootGroup();
IDataItem item = root.findDataItem( key );
IGroup root2 = item.getRootGroup();
```
The code above should lead to “root” and “root2” to be 2 references of the same instance.

Example 2:

Let suppose the following structure the root group has 1 group item called my\_group. And my\_group as one data item called my\_data. Then:
```
IDataset dataset = Factory.openDataset(URI);
IGroup root = dataset.getRootGroup();
IDataItem item = root.findDataItem( "my_data" );
IGroup group1 = item.getParent();
IGroup group2 = root.getGroup( "my_group" );
IGroup group3 = root.findGroup( "my_group" );
```
Same as before, the above code should lead to have `group1`, `group2` and `group3` to be 3 references of the same instance. Only one call to the `IGroup` constructor should be done.

## Data description and manipulation ##

### Data description ###

Physical data are seen as matrices. Even when they are a scalar, they are always described using the IArray interface. Below the main methods provided by the array interface:
```
public interface IArray extends IModelObject {
	IArrayUtils getArrayUtils();
	IArrayMath getArrayMath();

	Class<?> getElementType();
	int getRank();
	int[] getShape();
	IIndex getIndex();
	ISliceIterator getSliceIterator(int rank);
	IArrayIterator getIterator();
        IArrayIterator getRegionIterator(int[] reference, int[] range);
...
}
```

The `IArray` interface is the entry point of data access and manipulation, it give all necessary informations to process, copy... matrices

Classes used to described matrices are the following:
  * `IArray` that holds and manages the memory storage
  * `IIndex` defines the viewable part of an `IArray`, “independently” of it's storage. More exactly it does all cell index calculation when accessing the storage. For instance, when changing the IIndex of an array without reloading data, we can be able to iterate over it by scanning one cell on two.
  * `IRange` defines each dimension of the `IIndex`. It means its length, offset, stride. It tells on one dimension of the storage how many cells should be avoided between two cells that are seen as contiguous by the user.

They work as following: `IArray` uses an `IIndex` to do all its calculation of cell offset and position. `IIndex` uses `IRange` to define each dimension of the represented array.

Below how they are structured in the NeXus engine:
```
public class NxsArray implements IArray {
	private IIndex m_index;   // Viewable part of the array
	private Object m_oData;   // Array of primitive
	private int[]  m_shape;   // Shape of the backing storage
...
}

public class NxsIndex implements IIndex {
	private int   m_rank;        // rank of the view
	private int[] m_iCurPos;     // Current position
	private NxsRange[] m_ranges; // View in each dimension
...
}

public class NxsRange implements IRange {
    private long      m_last;     // number of elements
    private long      m_first;    // first value in range
    private long      m_stride;   // stride, must be >= 1
    private String    m_name;     // optional name
    private boolean   m_reduced;  // range reduced or not i.e is the range visible outside the index
...
}
```

At this point the engine should be able to browse and load data from the data source.

### Data manipulation ###

The manipulation of matrices is processed with following items:
  * `IArrayIterator` permits to iterate over the array and get each value according to the defined view (of the `IArray`).
  * `ISliceIterator` permits to slice the IArray into several parts and to get a new `IArray` (smaller than the original one) that shares the same backing storage.
  * `IArrayMath` is implemented by default in the “Core” but it uses `IArray` methods to apply its algorithms so it isn't the most efficient: see class `ArrayMath` in math package.
  * `IArrayUtils` is implemented by default in the “Core” but it uses `IArray` methods to apply its algorithms so it isn't the most efficient: see class `ArrayUtils` in utils package.