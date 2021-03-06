/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
//
// Contributors:
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
//    Tony Lam (nxi@Bragg Institute) - initial API and implementation
//    Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
// ****************************************************************************
package org.cdma;

import java.io.IOException;
import java.net.URI;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDatasource;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.utils.FactoryManager;

/// @cond pluginAPIclientAPI

/**
 * @brief The plug-in factory is the entry point of the CDMA plug-in.
 * 
 *        The IFactory interface is implemented by each plug-in. It permits to instantiate the IDataset and
 *        all the CDMA plug-in's object that will be used during the process.
 */

public interface IFactory {

    /**
     * Retrieve the dataset referenced by the URI.
     * 
     * @param uri URI object
     * @return CDMA Dataset
     * @throws FileAccessException
     */
    public IDataset openDataset(final URI uri) throws FileAccessException;

    /**
     * Instantiate a IDictionary with the given URI. Its loading can be done later.
     * 
     * @param uri of the dictionary
     * @return a new instance of the dictionary
     * @throws FileAccessException
     */
    @Deprecated
    public org.cdma.interfaces.IDictionary openDictionary(final URI uri) throws FileAccessException;

    /**
     * Instantiate a IDictionary with the given file path. Its loading can be done later.
     * 
     * @param path of the dictionary file
     * @return a new instance of the dictionary
     * @throws FileAccessException
     */
    @Deprecated
    public org.cdma.interfaces.IDictionary openDictionary(final String filepath) throws FileAccessException;

    /**
     * Create an empty IArray with a certain data type and certain shape.
     * 
     * @param clazz Class type
     * @param shape java array of integer
     * @return CDMA IArray
     */
    public IArray createArray(final Class<?> clazz, final int[] shape);

    /**
     * Create an IArray with a given data type, shape and data storage.
     * 
     * @param clazz in Class type
     * @param shape java array of integer
     * @param storage a 1D java array in the type reference by clazz
     * @return CDMA IArray
     */
    public IArray createArray(final Class<?> clazz, final int[] shape, final Object storage);

    /**
     * Create an IArray from a java array. A new 1D java array storage will be
     * created. The new CDMA IArray will be in the same type and same shape as the
     * java array. The storage of the new array will be a COPY of the supplied
     * java array.
     * 
     * @param javaArray one to many dimensional java array
     * @return CDMA IArray
     */
    public IArray createArray(final Object javaArray);

    /**
     * Create an IArray of String storage. The rank of the new IArray will be 2
     * because it treat the IArray as 2D char array.
     * 
     * @param string String value
     * @return new IArray object
     */
    public IArray createStringArray(final String string);

    /**
     * Create a double type IArray with a given single dimensional java double
     * storage. The rank of the generated IArray object will be 1.
     * 
     * @param javaArray java double array in one dimension
     * @return new IArray object
     */
    public IArray createDoubleArray(final double[] javaArray);

    /**
     * Create a double type IArray with a given java double storage and shape.
     * 
     * @param javaArray java double array in one dimension
     * @param shape java integer array
     * @return new IArray object
     */
    public IArray createDoubleArray(final double[] javaArray, final int[] shape);

    /**
     * Create an IArray from a java array. A new 1D java array storage will be
     * created. The new CDMA IArray will be in the same type and same shape as the
     * java array. The storage of the new array will be the supplied java array.
     * 
     * @param javaArray java primary array
     * @return CDMA array
     */
    public IArray createArrayNoCopy(final Object javaArray);

    /**
     * Create a IDataItem with a given CDMA parent Group, name and CDMA IArray data.
     * If the parent Group is null, it will generate a temporary Group as the
     * parent group.
     * 
     * @param parent CDMA Group
     * @param shortName in String type
     * @param array CDMA IArray
     * @return CDMA IDataItem
     * @throws InvalidArrayTypeException
     */
    public IDataItem createDataItem(final IGroup parent, final String shortName, final IArray array)
            throws InvalidArrayTypeException;

    /**
     * Create a CDMA Group with a given parent CDMA Group and a name.
     * 
     * @param parent CDMA Group
     * @param shortName in String type
     * @return CDMA Group
     */
    public IGroup createGroup(final IGroup parent, final String shortName);

    /**
     * Create an empty CDMA Group with a given name. The factory will create an
     * empty CDMA Dataset first, and create the new Group under the root group of
     * the Dataset.
     * 
     * @param shortName in String type
     * @return CDMA Group
     * @throws IOException
     */
    public IGroup createGroup(final String shortName) throws IOException;

    /**
     * Create an empty CDMA Logical Group with a given key.
     * 
     * @param dataset an IDataset that this group will belong to
     * @param key an IKey that this group will correspond to
     * @return CDMA Logical Group
     * @throws IOException
     */
    public LogicalGroup createLogicalGroup(IDataset dataset, IKey key);

    /**
     * Create a CDMA Attribute with given name and value.
     * 
     * @param name in String type
     * @param value in String type
     * @return CDMA Attribute
     */
    public IAttribute createAttribute(final String name, final Object value);

    /**
     * Create a CDMA Dataset with a URI reference. If the file exists it will open
     * it, else it will be created
     * 
     * @param uri URI object
     * @return CDMA Dataset
     * @throws Exception
     */
    public IDataset createDatasetInstance(final URI uri) throws Exception;

    /**
     * Create a CDMA Dataset in memory only. The dataset is not open yet. It is
     * necessary to call dataset.open() to access the root of the dataset.
     * 
     * @return a CDMA Dataset
     * @throws IOException
     *             I/O error
     */
    public IDataset createEmptyDatasetInstance() throws IOException;

    /**
     * Create a IKey having the given name.
     * 
     * @param name of the key
     * @return a new IKey
     */
    public IKey createKey(String name);

    /**
     * Create a IPath having the given value.
     * 
     * @param path interpreted by the plug-in
     * @return a new IPath
     */
    public Path createPath(String path);

    /**
     * The factory has a unique name that identifies it.
     * 
     * @return the factory's name
     */
    public String getName();

    /**
     * The plug-in has a label, which describe the institute it comes from
     * and / or the data source it is supposed to read / write: a human friendly
     * information of which plug-in is working.
     * 
     * @return the plug-in's label
     */
    public String getPluginLabel();

    /**
     * Shortly describes the specificities of the plug-in for instance the underlying format,
     * managed file's extension, or protocol.
     * 
     * @return a plug-in's description
     */
    public String getPluginDescription();

    /**
     * Returns the URI detector of the instantiated plug-in.
     * 
     * @return IPluginURIDetector
     */
    public IDatasource getPluginURIDetector();

    /**
     * Create an empty CDMA IDictionary
     * 
     * @return a CDMA IDictionary
     */
    @Deprecated
    public org.cdma.interfaces.IDictionary createDictionary();

    /**
     * Returns the plug-ins version number with under the form
     * of 3 digits: X_Y_Z
     * 
     * @return String representation of the plug-in's version
     */
    String getPluginVersion();

    /**
     * Returns the CDMA core's version this plug-in is expected to work with.
     * The version will be represented under the form of 3 digits:
     * X_Y_Z
     * 
     * @return String representation of the CDMA core's version
     */
    String getCDMAVersion();

    /**
     * This method is called by the main API when the plug-in is loaded during the plug-in
     * discovering process.<br/>
     * The method should check that all requirement are present to make the plug-in fully
     * functional. If it's not the case it should use the {@link FactoryManager.unregisterFactory} method.
     */
    void processPostRecording();

    /**
     * Tells if this plug-in can use or not the Dictionary mechanism is available
     * 
     * @return true if the Dictionary mechanism can be enabled
     */
    boolean isLogicalModeAvailable();
}

/// @endcond pluginAPIclientAPI
