/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.soleil.nexus;

import java.io.IOException;
import java.lang.reflect.Array;
import java.net.URI;
import java.util.logging.Level;

import ncsa.hdf.object.h5.H5ScalarDS;

import org.cdma.AbstractFactory;
import org.cdma.Factory;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.engine.hdf.navigation.HdfAttribute;
import org.cdma.engine.hdf.navigation.HdfDataset;
import org.cdma.engine.hdf.utils.HdfObjectUtils;
import org.cdma.exception.CDMAException;
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
import org.cdma.plugin.soleil.nexus.array.NxsArray;
import org.cdma.plugin.soleil.nexus.dictionary.NxsLogicalGroup;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataItem;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataset;
import org.cdma.plugin.soleil.nexus.navigation.NxsGroup;
import org.cdma.plugin.soleil.nexus.utils.NxsArrayMath;

import fr.soleil.lib.project.math.ArrayUtils;

public final class NxsFactory extends AbstractFactory {
    private static NxsFactory factory;
    private static NxsDatasource detector;
    public static final String NAME = "SoleilNeXus2";
    public static final String LABEL = "SOLEIL's NeXus plug-in v2";
    public static final String DEBUG_INF = "CDMA_DEBUG";
    public static final String CONFIG_FILE = "cdma_nexussoleil_config.xml";
    private static final String CDMA_VERSION = "3_2_0";
    private static final String PLUG_VERSION = "2.0.0";
    private static final String DESC = "This plug-in manages NeXus data files (having 'nxs' for extension).";

    public NxsFactory() {
    }

    public static NxsFactory getInstance() {
        synchronized (NxsFactory.class) {
            if (factory == null) {
                factory = new NxsFactory();
                detector = NxsDatasource.getInstance();
            }
        }
        return factory;
    }

    @Override
    public IArray createArray(final Class<?> clazz, final int[] shape) {
        Object o = java.lang.reflect.Array.newInstance(clazz, shape);
        IArray result = null;
        try {
            result = new NxsArray(o, shape);
        } catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        }
        return result;
    }

    @Override
    public IArray createArray(final Class<?> clazz, final int[] shape, final Object storage) {
        IArray result = null;
        if (storage instanceof IArray[]) {
            result = new NxsArray((IArray[]) storage);
        } else if (storage instanceof H5ScalarDS) {
            try {
                result = new NxsArray(((H5ScalarDS) storage).getData(), shape);
            } catch (Exception e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
            }

        } else {
            try {
                result = new NxsArray(storage, shape);
            } catch (InvalidArrayTypeException e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
            }
        }
        return result;
    }

    @Override
    public IArray createArray(final Object value) {
        IArray result = null;

        // Default value is for non array value.
        int[] shape = { 1 };

        Object inlineArray;
        if (value != null) {
            if (value.getClass().isArray()) {
                shape = ArrayUtils.recoverShape(value);
                inlineArray = ArrayUtils.convertArrayDimensionFromNTo1(value);
            } else {
                inlineArray = Array.newInstance(value.getClass(), 1);
                Array.set(inlineArray, 0, value);
            }
            try {
                result = new NxsArray(inlineArray, shape);
            } catch (InvalidArrayTypeException e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
            }
        }
        return result;
    }

    @Override
    public IArray createArrayNoCopy(final Object array) {
        IArray result = null;
        if (array instanceof IArray[]) {
            result = new NxsArray((IArray[]) array);
        } else if (array instanceof H5ScalarDS) {

            H5ScalarDS hdfItem = (H5ScalarDS) array;
            try {
                result = new NxsArray(hdfItem.getData(), HdfObjectUtils.convertLongToInt(hdfItem.getDims()));
            } catch (Exception e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
            }
        } else {
            int size = Array.getLength(array);
            try {
                result = new NxsArray(array, new int[] { size });
            } catch (Exception e) {
                result = null;
            }

        }
        return result;
    }

    @Override
    public IAttribute createAttribute(final String name, final Object value) {
        return new HdfAttribute(NAME, name, value);
    }

    @Override
    public IDataItem createDataItem(final IGroup parent, final String shortName, final IArray array)
            throws InvalidArrayTypeException {
        IDataItem result = new NxsDataItem(shortName, (NxsDataset) parent.getDataset());
        result.setCachedData(array, false);
        parent.addDataItem(result);
        return result;
    }

    @Override
    public IDataItem createDataItem(IGroup parent, String shortName, Object value) throws CDMAException {
        NxsDataItem result = null;
        if (value instanceof NxsDataItem) {
            NxsDataItem itemToLinkTo = (NxsDataItem) value;
            result = new NxsDataItem(shortName, (NxsDataset) parent.getDataset());
            result.linkTo(itemToLinkTo);
        } else {
            IArray array = createArray(value);
            result = (NxsDataItem) createDataItem(parent, shortName, array);
        }
        parent.addDataItem(result);
        return result;
    }

    @Override
    public IDataset createDatasetInstance(final URI uri) throws NoResultException {
        return NxsDataset.instanciate(uri);
    }

    @Override
    public IDataset createDatasetInstance(URI uri, boolean withWriteAccess) throws CDMAException {
        return NxsDataset.instanciate(uri, withWriteAccess);
    }

    @Override
    public IArray createDoubleArray(final double[] javaArray) {
        IArray result = null;
        try {
            int[] shape = new int[1];
            shape[0] = javaArray.length;
            result = new NxsArray(javaArray, shape);
        } catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        }
        return result;
    }

    @Override
    public IArray createDoubleArray(final double[] javaArray, final int[] shape) {

        IArray result = null;
        try {
            result = new NxsArray(javaArray, shape);
        } catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        }
        return result;
    }

    @Override
    public IDataset createEmptyDatasetInstance() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public IGroup createGroup(final IGroup parent, final String shortName) {
        String path_val = parent.getLocation();
        NxsGroup group = new NxsGroup((NxsDataset) parent.getDataset(), shortName, path_val, (NxsGroup) parent);
        parent.addSubgroup(group);
        return group;
    }

    @Override
    public IGroup createGroup(final String shortName) throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public IArray createStringArray(final String string) {
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
        synchronized (NxsDatasource.class) {
            if (detector == null) {
                detector = NxsDatasource.getInstance();
            }
        }
        return detector;
    }

    @Override
    public IDataset openDataset(final URI uri) throws FileAccessException {
        IDataset ds = null;
        try {
            ds = NxsDataset.instanciate(uri);
        } catch (NoResultException e) {
            throw new FileAccessException(e);
        }
        return ds;
    }

    @Override
    public IKey createKey(final String keyName) {
        return new Key(this, keyName);
    }

    @Override
    public LogicalGroup createLogicalGroup(final IDataset dataset, final IKey key) {
        return new NxsLogicalGroup(dataset, new Key(this, key.getName()));
    }

    @Override
    public Path createPath(final String path) {
        return new Path(this, path);
    }

    @Deprecated
    @Override
    public IDictionary openDictionary(final URI uri) throws FileAccessException {
        throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public IDictionary openDictionary(final String filepath) throws FileAccessException {
        throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public IDictionary createDictionary() {
        throw new UnsupportedOperationException();
    }

    public static IArrayMath createArrayMath(final NxsArray array) {
        return new NxsArrayMath(array);
    }

    @Override
    public String getPluginVersion() {
        return PLUG_VERSION;
    }

    @Override
    public String getCDMAVersion() {
        return CDMA_VERSION;
    }

    @Override
    public String getPluginDescription() {
        return NxsFactory.DESC;
    }

    @Override
    public void processPostRecording() {
        synchronized (NxsFactory.class) {
            boolean checkHdfAPI = HdfDataset.checkHdfAPI();
            if (!checkHdfAPI) {
                Factory.getManager().unregisterFactory(NAME);
                factory = null;
            }
        }
    }

    @Override
    public boolean isLogicalModeAvailable() {
        String dictPath = Factory.getDictionariesFolder();
        return (dictPath != null && !dictPath.isEmpty());
    }

}
