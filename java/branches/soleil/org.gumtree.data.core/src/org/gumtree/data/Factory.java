/*******************************************************************************
 * Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 *    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 ******************************************************************************/
package org.gumtree.data;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.exception.InvalidArrayTypeException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.interfaces.IIndex;
import org.gumtree.data.interfaces.IKey;
import org.gumtree.data.utils.FactoryManager;
import org.gumtree.data.utils.IFactoryManager;
import org.gumtree.data.utils.Utilities;

/**
 * The Factory class in gumtree data model is a tools to create GDM objects. The
 * factory can take a URI as a parameter to read data in as GDM object, or it
 * can create an empty GDM object to hold data in a future time. <br>
 * Abbreviation: Gumtree data model -- GDM
 * 
 * @author nxi
 * @version 1.1
 */
public final class Factory {

//	private static final Logger logger = LoggerFactory.getLogger(Factory.class);
	
	private static volatile IFactoryManager manager;
	
	public static final String DICO_PATH_PROP = "CDM_DICTIONARY_PATH";
	
	private static String CDM_EXPERIMENT = "";
	
	/**
	 * Retrieve the dataset referenced by the URI.
	 * 
	 * @param uri
	 *            URI object
	 * @return GDM Dataset
	 * @throws FileAccessException
	 *             Created on 18/06/2008
	 */
	public static IDataset openDataset(URI uri)
			throws FileAccessException {
		Object rootGroup = Utilities.findObject(uri, null);
		if (rootGroup != null) {
			return ((IGroup) rootGroup).getDataset();
		} else {
			return null;
		}
	}
	
	public static void setActiveView(String experiment) {
		CDM_EXPERIMENT = experiment;
	}
	
	public static String getActiveView() {
		return CDM_EXPERIMENT;
	}
	
	/**
	 * According to the currently defined experiment, this method will return the path
	 * to reach the declarative dictionary. It means the file where
	 * is defined what should be found in a IDataset that fits the experiment.
	 * It's a descriptive file.
	 * 
	 * @return the path to the standard declarative file
	 */
	public static String getKeyDictionaryPath() {
		String sDict = getDictionariesFolder();
		String sFile = ( getActiveView() + "_view.xml" ).toLowerCase();
		
		return sDict + "/" + sFile;
	}
	
	/**
	 * According to the given factory this method will return the path to reach
	 * the folder containing mapping dictionaries. This file associate entry 
	 * keys to paths that are plug-in dependent.
	 * 
	 * @param factory of the plug-in instance from which we want to load the dictionary
	 * @return the path to the plug-in's mapping dictionaries folder  
	 */
	public static String getMappingDictionaryFolder(IFactory factory) {
		String sDict = getDictionariesFolder();

		return sDict + "/" + factory.getName() + "/";
	}
	
	/**
	 * Set the folder path where to search for key dictionary files.
	 * This folder should contains all dictionaries that the above application needs.
	 * @param path targeting a folder
	 */
	public static void setDictionariesFolder(String path) {
		System.setProperty(DICO_PATH_PROP, path);
	}
	
	public static String getDictionariesFolder() {
		return System.getProperty(DICO_PATH_PROP, System.getenv(DICO_PATH_PROP));
	}

	/**
	 * Create an index of Array by given a shape of the Array.
	 * 
	 * @param shape
	 *            java array of integer
	 * @return GDM Array Index
	 * @deprecated it is recommended to use Array.getIndex() instead.
	 * @see IArray#getIndex()
	 */
	public static IIndex createIndex(int[] shape) {
		throw new UnsupportedOperationException();
	}

	/**
	 * Create an empty Array with a certain data type and certain shape.
	 * 
	 * @param clazz
	 *            Class type
	 * @param shape
	 *            java array of integer
	 * @return GDM Array Created on 18/06/2008
	 */
	public static IArray createArray(Class<?> clazz, int[] shape) {
		return getFactory().createArray(clazz, shape);
	}

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
	public static IArray createArray(Class<?> clazz, int[] shape,
			final Object storage) {
		return getFactory().createArray(clazz, shape, storage);
	}

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
	public static IArray createArray(final Object javaArray) {
		return getFactory().createArray(javaArray);
	}

	/**
	 * Create an Array of String storage. The rank of the new Array will be 2
	 * because it treat the Array as 2D char array.
	 * 
	 * @param string
	 *            String value
	 * @return new Array object
	 */
	public static IArray createStringArray(String string) {
		return createArray(String.class, new int[] { 1 }, new String[] { string
				.toString() });
	}

	/**
	 * Create a double type Array with a given single dimensional java double
	 * storage. The rank of the generated Array object will be 1.
	 * 
	 * @param javaArray
	 *            java double array in one dimension
	 * @return new Array object Created on 10/11/2008
	 */
	public static IArray createDoubleArray(double[] javaArray) {
		return createArrayNoCopy(javaArray);
	}

	/**
	 * Create a double type Array with a given java double storage and shape.
	 * 
	 * @param javaArray
	 *            java double array in one dimension
	 * @param shape
	 *            java integer array
	 * @return new Array object Created on 10/11/2008
	 */
	public static IArray createDoubleArray(double[] javaArray,
			final int[] shape) {
		return createArray(Double.TYPE, shape, javaArray);
	}

	/**
	 * Create an IArray from a java array. A new 1D java array storage will be
	 * created. The new GDM Array will be in the same type and same shape as the
	 * java array. The storage of the new array will be the supplied java array.
	 * 
	 * @param javaArray
	 *            java primary array
	 * @return GDM array Created on 28/10/2008
	 */
	public static IArray createArrayNoCopy(Object javaArray) {
		int rank = 0;
		Class<?> componentType = javaArray.getClass();
		while (componentType.isArray()) {
			rank++;
			componentType = componentType.getComponentType();
		}

		// get the shape
		int count = 0;
		int[] shape = new int[rank];
		Object jArray = javaArray;
		Class<?> cType = jArray.getClass();
		while (cType.isArray()) {
			shape[count++] = java.lang.reflect.Array.getLength(jArray);
			jArray = java.lang.reflect.Array.get(jArray, 0);
			cType = jArray.getClass();
		}

		// create the Array
		return createArray(componentType, shape, javaArray);
	}

	/**
	 * Create a GDM DataItem with a given parent Group, Dataset, name and GDM
	 * Array data.
	 * 
	 * @param dataset
	 *            GDM Dataset
	 * @param parent
	 *            GDM Group
	 * @param shortName
	 *            in String type
	 * @param array
	 *            GDM Array
	 * @return GDM IDataItem
	 * @throws InvalidArrayTypeException
	 *             wrong type
	 * @deprecated use {@link #createDataItem(IGroup, String, IArray)} instead
	 * @since 18/06/2008
	 */
	public static IDataItem createDataItem(IDataset dataset,
			IGroup parent, String shortName, IArray array)
			throws InvalidArrayTypeException {
		throw new UnsupportedOperationException();
	}

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
	public static IDataItem createDataItem(IGroup parent,
			String shortName, IArray array)
			throws InvalidArrayTypeException {
		return getFactory().createDataItem(parent, shortName, array);
	}

	/**
	 * Create a GDM Group with given Dataset, parent GDM Group, name. A boolean
	 * initiate parameter tells the factory if the new group will be put in the
	 * list of children of the parent Group.
	 * 
	 * @param dataset
	 *            GDM Dataset
	 * @param parent
	 *            GDM Group
	 * @param shortName
	 *            in String type
	 * @param init
	 *            boolean type
	 * @return GDM Group
	 * @deprecated use {@link #createGroup(IGroup, String, boolean)} instead
	 *             Created on 18/06/2008
	 */
	public static IGroup createGroup(IDataset dataset,
			IGroup parent, String shortName, boolean init) {
		throw new UnsupportedOperationException();
	}

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
	public static IGroup createGroup(IGroup parent,
			String shortName, boolean updateParent) {
		return getFactory().createGroup(parent, shortName, updateParent);
	}

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
	public static IGroup createGroup(String shortName)
	throws IOException {
		return getFactory().createGroup(shortName);
	}

	/**
	 * Create an empty GDM Logical Group with a given key. 
	 * 
	 * @param parent
	 *            an ILogicalGroup
	 * @param key
	 *            an IKey that this group will correspond
	 * @return GDM Logical Group
	 * @throws IOException
	 *             Created on 18/06/2008
	 */
	public static ILogicalGroup createLogicalGroup(IDataset dataset, IKey key) {
		return getFactory().createLogicalGroup(dataset, key);
	}

	/**
	 * Create a GDM Attribute with given name and value.
	 * 
	 * @param name
	 *            in String type
	 * @param value
	 *            in String type
	 * @return GDM Attribute Created on 18/06/2008
	 */
	public static IAttribute createAttribute(String name, Object value) {
		return getFactory().createAttribute(name, value);
	}

	/**
	 * Create a GDM Dataset with a URI reference. If the file exists, it will
	 * 
	 * @param uri
	 *            URI object
	 * @return GDM Dataset
	 * @throws Exception
	 *             Created on 18/06/2008
	 */
	public static IDataset createDatasetInstance(URI uri)
			throws Exception {
		return getFactory().createDatasetInstance(uri);
	}

	/**
	 * Create a GDM Dataset in memory only. The dataset is not open yet. It is
	 * necessary to call dataset.open() to access the root of the dataset.
	 * 
	 * @return a GDM Dataset
	 * @throws IOException
	 *             I/O error Created on 18/06/2008
	 */
	public static IDataset createEmptyDatasetInstance() throws IOException {
		IDataset dataset = getFactory().createEmptyDatasetInstance();
		return dataset;
	}
	
	public static IKey createKey(String keyName) {
		return getFactory().createKey(keyName);
	}
	
	public static IPath createPath(String path) {
		return getFactory().createPath(path);
	}
	
	/**
	 * Create an empty GDM IDictionary
	 * 
	 * @return a GDM IDictionary
	 * Created on 24/03/2010
	 */
	public static IDictionary createDictionary() {
		return getFactory().createDictionary();
	}
	
	public static IFactoryManager getManager() {
		if (manager == null) {
			synchronized (Factory.class) {
				if (manager == null) {
					manager = new FactoryManager();
				}
			}
		}
		return manager;
	}
	
	public static IFactory getFactory() {
		return getManager().getFactory();
	}
	
	public static IFactory getFactory(String name) {
		return getManager().getFactory(name);
	}
	
	public static IFactory getFactory(URI uri) {
		return detectPlugin(uri);
	}
	
	public static IFactory detectPlugin(URI uri) {
		ArrayList<IFactory> reader = new ArrayList<IFactory>();
		IFactory result = null;
		IFactory plugin;
		IDatasource detector;
		IFactoryManager mngr = getManager();
		
		
		Map<String, IFactory> registry = mngr.getFactoryRegistry();
		
		for( Entry<String, IFactory> entry : registry.entrySet() ) {
			plugin   = entry.getValue();
			detector = plugin.getPluginURIDetector();
			
			if( detector.isReadable(uri) ) {
				reader.add( plugin );
				if( detector.isProducer(uri) ) {
					result = plugin;
					break;
				}
			}
		}
		
		if( result == null && reader.size() > 0 ) {
			result = reader.get(0);
		}
		
		return result;
	}
	
	/**
	 * Hide default constructor.
	 */
	private Factory() {
	}
	

}
