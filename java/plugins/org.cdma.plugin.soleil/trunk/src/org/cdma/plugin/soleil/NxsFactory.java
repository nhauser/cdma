//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.soleil;

import java.io.IOException;
import java.lang.reflect.Array;
import java.net.URI;

import org.cdma.IFactory;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.engine.nexus.navigation.NexusAttribute;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDatasource;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.math.IArrayMath;
import org.cdma.plugin.soleil.array.NxsArray;
import org.cdma.plugin.soleil.dictionary.NxsLogicalGroup;
import org.cdma.plugin.soleil.navigation.NxsDataset;
import org.cdma.plugin.soleil.navigation.NxsGroup;
import org.cdma.plugin.soleil.utils.NxsArrayMath;
import org.cdma.plugin.soleil.utils.NxsArrayUtils;
import org.cdma.utils.IArrayUtils;

import fr.soleil.nexus.DataItem;
import fr.soleil.nexus.PathGroup;
import fr.soleil.nexus.PathNexus;

public final class NxsFactory implements IFactory {
    private static NxsFactory factory;
    private static NxsDatasource detector;
    public static final String NAME        = "SoleilNeXus";
    public static final String LABEL       = "SOLEIL's NeXus plug-in";
    public static final String DEBUG_INF   = "CDMA_DEBUG";
    public static final String CONFIG_FILE = "cdma_nexussoleil_config.xml";

    public static final String ERR_NOT_SUPPORTED = "not supported yet in plug-in!";

    public NxsFactory() {
    }

    public static NxsFactory getInstance() {
        synchronized (NxsFactory.class ) {
            if( factory == null ) {
                factory  = new NxsFactory();
                detector = NxsDatasource.getInstance();
            }
        }
        return factory;
    }

    @Override
    public IArray createArray(Class<?> clazz, int[] shape) {
        Object o = java.lang.reflect.Array.newInstance(clazz, shape);
        return new NxsArray( o, shape);
    }

    @Override
    public IArray createArray(Class<?> clazz, int[] shape, Object storage) {
        IArray result = null;
        if( storage instanceof IArray[] ) {
            result = new NxsArray( (IArray[]) storage );
        }
        else if( storage instanceof DataItem ) {
            result = new NxsArray( (DataItem) storage );
        }
        else {
            result = new NxsArray( storage, shape);
        }
        return result;
    }

    @Override
    public IArray createArray(Object javaArray) {
        IArray result = null;
        // [ANSTO][Tony][2011-08-31] testing isArray may be slow
        // [SOLEIL][clement][2012-04-18] as the supported array is a primitive type the "instanceof" won't be correct 
        if (javaArray != null && javaArray.getClass().isArray()) {
            int size = Array.getLength(javaArray);
            result = new NxsArray(javaArray, new int[] { size });
        }
        return result;
    }

    @Override
    public IArray createArrayNoCopy(Object array) {
        IArray result = null;
        if( array instanceof IArray[] ) {
            result = new NxsArray( (IArray[]) array);
        }
        else if( array instanceof DataItem ) {
            result = new NxsArray( (DataItem) array);
        }
        else {
            DataItem dataset = null;
            try {
                dataset = new DataItem(array);
                result = new NxsArray(dataset);
            } catch( Exception e ) {
                result = null;
            }

        }
        return result;
    }

    @Override
    public IAttribute createAttribute(String name, Object value) {
        return new NexusAttribute(NAME ,name, value);
    }

    @Override
    public IDataItem createDataItem(IGroup parent, String shortName, IArray array) throws InvalidArrayTypeException {
        throw new InvalidArrayTypeException(ERR_NOT_SUPPORTED);
    }

    @Override
    public IDataset createDatasetInstance(URI uri) throws Exception {
        return NxsDataset.instanciate(uri);
    }

    @Override
    public IArray createDoubleArray(double[] javaArray) {
        DataItem data;
        try {
            data = new DataItem(javaArray);
        } catch( Exception e ) {
            data = null;
        }
        return new NxsArray(data);
    }

    @Override
    public IArray createDoubleArray(double[] javaArray, int[] shape) {
        DataItem data;
        try {
            data = new DataItem(javaArray);
        } catch( Exception e ) {
            data = null;
        }
        return new NxsArray(data);
    }

    @Override
    public IDataset createEmptyDatasetInstance() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public IGroup createGroup(IGroup parent, String shortName) {
        String path_val = parent.getLocation();
        PathGroup path = new PathGroup(PathNexus.splitStringPath(path_val));
        NxsGroup group = new NxsGroup( parent, (PathNexus) path, (NxsDataset) parent.getDataset());

        return group;
    }

    @Override
    public IGroup createGroup(String shortName) throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public IArray createStringArray(String string) {
        throw new NotImplementedException();
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public String getPluginLabel() {
        return LABEL;
    }

    @Override
    public IDatasource getPluginURIDetector() {
        synchronized (NxsDatasource.class ) {
            if( detector == null ) {
                detector = NxsDatasource.getInstance();
            }
        }
        return detector;
    }

    @Override
    public IDataset openDataset(URI uri) throws FileAccessException {
        IDataset ds = null;
        try {
            ds = NxsDataset.instanciate(uri);
        }
        catch( NoResultException e) {
            throw new FileAccessException(e);
        }
        return ds;
    }

    @Override
    public IKey createKey(String keyName) {
        return new Key(this, keyName);
    }

    @Override
    public LogicalGroup createLogicalGroup(IDataset dataset, IKey key) {
        return new NxsLogicalGroup(dataset, new Key(this, key.getName()) );
    }

    @Override
    public Path createPath(String path) {
        return new Path(this, path);
    }

    @Deprecated
    @Override
    public IDictionary openDictionary(URI uri) throws FileAccessException {
        throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public IDictionary openDictionary(String filepath)
            throws FileAccessException {
        throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public IDictionary createDictionary() {
        throw new UnsupportedOperationException();
    }

    public static IArrayUtils createArrayUtils(NxsArray array) {
        return new NxsArrayUtils(array);
    }

    public static IArrayMath createArrayMath(NxsArray array) {
        return new NxsArrayMath(array);
    }
}
