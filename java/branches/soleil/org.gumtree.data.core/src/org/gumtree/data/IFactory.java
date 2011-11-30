package org.gumtree.data;

import java.io.IOException;
import java.net.URI;

import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.dictionary.IPathParamResolver;
import org.gumtree.data.dictionary.IPathParameter;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.exception.InvalidArrayTypeException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.interfaces.IKey;
import org.gumtree.data.utils.Utilities.ParameterType;

public interface IFactory {

	/**
	 * Retrieve the dataset referenced by the URI.
	 * 
	 * @param uri
	 *            URI object
	 * @return GDM Dataset
	 * @throws FileAccessException
	 *             Created on 18/06/2008
	 */
	public IDataset openDataset(final URI uri) throws FileAccessException;

	public IDictionary openDictionary(final URI uri) throws FileAccessException;
	
	public IDictionary openDictionary(final String filepath) throws FileAccessException;
	
	/**
	 * Create an empty Array with a certain data type and certain shape.
	 * 
	 * @param clazz
	 *            Class type
	 * @param shape
	 *            java array of integer
	 * @return GDM Array Created on 18/06/2008
	 */
	public IArray createArray(final Class<?> clazz, final int[] shape);

	/**
	 * Create an Array with a given data type, shape and data storage.
	 * 
	 * @param clazz
	 *            in Class type
	 * @param shape
	 *            java array of integer
	 * @param storage
	 *            a 1D java array in the type reference by clazz
	 * @return GDM Array Created on 18/06/2008
	 */
	public IArray createArray(final Class<?> clazz, final int[] shape,
			final Object storage);

	/**
	 * Create an Array from a java array. A new 1D java array storage will be
	 * created. The new GDM Array will be in the same type and same shape as the
	 * java array. The storage of the new array will be a COPY of the supplied
	 * java array.
	 * 
	 * @param javaArray
	 *            one to many dimensional java array
	 * @return GDM Array Created on 18/06/2008
	 */
	public IArray createArray(final Object javaArray);

	/**
	 * Create an Array of String storage. The rank of the new Array will be 2
	 * because it treat the Array as 2D char array.
	 * 
	 * @param string
	 *            String value
	 * @return new Array object
	 */
	public IArray createStringArray(final String string);

	/**
	 * Create a double type Array with a given single dimensional java double
	 * storage. The rank of the generated Array object will be 1.
	 * 
	 * @param javaArray
	 *            java double array in one dimension
	 * @return new Array object Created on 10/11/2008
	 */
	public IArray createDoubleArray(final double[] javaArray);

	/**
	 * Create a double type Array with a given java double storage and shape.
	 * 
	 * @param javaArray
	 *            java double array in one dimension
	 * @param shape
	 *            java integer array
	 * @return new Array object Created on 10/11/2008
	 */
	public IArray createDoubleArray(final double[] javaArray, final int[] shape);

	/**
	 * Create an IArray from a java array. A new 1D java array storage will be
	 * created. The new GDM Array will be in the same type and same shape as the
	 * java array. The storage of the new array will be the supplied java array.
	 * 
	 * @param javaArray
	 *            java primary array
	 * @return GDM array Created on 28/10/2008
	 */
	public IArray createArrayNoCopy(final Object javaArray);

	/**
	 * Create a DataItem with a given GDM parent Group, name and GDM Array data.
	 * If the parent Group is null, it will generate a temporary Group as the
	 * parent group.
	 * 
	 * @param parent
	 *            GDM Group
	 * @param shortName
	 *            in String type
	 * @param array
	 *            GDM Array
	 * @return GDM IDataItem
	 * @throws InvalidArrayTypeException
	 *             Created on 18/06/2008
	 */
	public IDataItem createDataItem(final IGroup parent,
			final String shortName, final IArray array)
			throws InvalidArrayTypeException;

	/**
	 * Create a GDM Group with a given parent GDM Group, name, and a boolean
	 * initiate parameter telling the factory if the new group will be put in
	 * the list of children of the parent. Group.
	 * 
	 * @param parent
	 *            GDM Group
	 * @param shortName
	 *            in String type
	 * @param updateParent
	 *            if the parent will be updated
	 * @return GDM Group Created on 18/06/2008
	 */
	public IGroup createGroup(final IGroup parent, final String shortName,
			final boolean updateParent);

	/**
	 * Create an empty GDM Group with a given name. The factory will create an
	 * empty GDM Dataset first, and create the new Group under the root group of
	 * the Dataset.
	 * 
	 * @param shortName
	 *            in String type
	 * @return GDM Group
	 * @throws IOException
	 *             Created on 18/06/2008
	 */
	public IGroup createGroup(final String shortName) throws IOException;

	/**
	 * Create an empty GDM Logical Group with a given key. 
	 * 
	 * @param dataset
	 *            an IDataset that this group will belong to
	 * @param key
	 *            an IKey that this group will correspond to
	 * @return GDM Logical Group
	 * @throws IOException
	 *             Created on 18/06/2008
	 */
	public ILogicalGroup createLogicalGroup(IDataset dataset, IKey key);
	
	/**
	 * Create a GDM Attribute with given name and value.
	 * 
	 * @param name
	 *            in String type
	 * @param value
	 *            in String type
	 * @return GDM Attribute Created on 18/06/2008
	 */
	public IAttribute createAttribute(final String name, final Object value);

	/**
	 * Create a GDM Dataset with a URI reference. If the file exists, it will
	 * 
	 * @param uri
	 *            URI object
	 * @return GDM Dataset
	 * @throws Exception
	 *             Created on 18/06/2008
	 */
	public IDataset createDatasetInstance(final URI uri) throws Exception;

	/**
	 * Create a GDM Dataset in memory only. The dataset is not open yet. It is
	 * necessary to call dataset.open() to access the root of the dataset.
	 * 
	 * @return a GDM Dataset
	 * @throws IOException
	 *             I/O error Created on 18/06/2008
	 */
	public IDataset createEmptyDatasetInstance() throws IOException;

	public IKey createKey(String keyName);
	
	public IPath createPath( String path );
	
	public IPathParameter createPathParameter(ParameterType type, String name, Object value);
	
	public IPathParamResolver createPathParamResolver(IPath path);
	
	/**
	 * Return the symbol used by the plug-in to separate nodes in a string path
	 * @return 
	 * @note <b>EXPERIMENTAL METHOD</b> do note use/implements
	 */
	public String getPathSeparator();
	
	/**
	 * The factory has a unique name that identifies it.
	 * @return the factory's name
	 */
	public String getName();
	
	/**
	 * The plug-in has a label, which describe the institute it comes from
	 * and / or the data source it is supposed to read / write: a human friendly
	 * information of which plug-in is working.
	 * @return the plug-in's label
	 */
	public String getPluginLabel();
	
	/**
	 * Returns the URI detector of the instantiated plug-in. 
	 * @return IPluginURIDetector
	 */
	public IDatasource getPluginURIDetector();

	public IDictionary createDictionary();
}
