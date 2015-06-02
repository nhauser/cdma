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
 * St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.soleil;

import java.io.IOException;
import java.lang.reflect.Array;
import java.net.URI;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.engine.nexus.array.NexusArray;
import org.cdma.engine.nexus.navigation.NexusAttribute;
import org.cdma.engine.nexus.navigation.NexusDataItem;
import org.cdma.engine.nexus.navigation.NexusDataset;
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
import org.cdma.plugin.soleil.navigation.NxsDataItem;
import org.cdma.plugin.soleil.navigation.NxsDataset;
import org.cdma.plugin.soleil.navigation.NxsGroup;
import org.cdma.plugin.soleil.utils.NxsArrayMath;
import org.nexusformat.NexusException;

import fr.soleil.nexus.DataItem;
import fr.soleil.nexus.PathData;
import fr.soleil.nexus.PathGroup;
import fr.soleil.nexus.PathNexus;

public final class NxsFactory implements IFactory {
    private static NxsFactory factory;
    private static NxsDatasource detector;
    public static final String NAME = "SoleilNeXus";
    public static final String LABEL = "SOLEIL's NeXus plug-in";
    public static final String DEBUG_INF = "CDMA_DEBUG";
    public static final String CONFIG_FILE = "cdma_nexussoleil_config.xml";
    private static final String CDMA_VERSION = "3.2.5";
    private static final String PLUG_VERSION = "1.4.13";
    private static final String DESC = "Manages NeXus data files (with �nxs� extension)";

    public NxsFactory() {
    }

    public static NxsFactory getInstance() {
        synchronized (NxsFactory.class) {
            boolean checkNeXusAPI = NexusDataset.checkNeXusAPI();
            if (factory == null && checkNeXusAPI) {
                factory = new NxsFactory();
                detector = NxsDatasource.getInstance();
            }

            if (!checkNeXusAPI) {
                Factory.getManager().unregisterFactory(NAME);
            }
        }
        return factory;
    }

    @Override
    public IArray createArray(Class<?> clazz, int[] shape) {
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
    public IArray createArray(Class<?> clazz, int[] shape, Object storage) {
        IArray result = null;
        if (storage instanceof IArray[]) {
            result = new NxsArray((IArray[]) storage);
        } else if (storage instanceof DataItem) {
            try {
                result = new NxsArray((DataItem) storage);
            } catch (InvalidArrayTypeException e) {
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
    public IArray createArray(Object javaArray) {
        IArray result = null;
        // [ANSTO][Tony][2011-08-31] testing isArray may be slow
        // [SOLEIL][clement][2012-04-18] as the supported array is a primitive type the "instanceof" won't be correct
        if (javaArray != null && javaArray.getClass().isArray()) {
            int size = Array.getLength(javaArray);
            try {
                result = new NxsArray(javaArray, new int[] { size });
            } catch (InvalidArrayTypeException e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
            }
        }
        return result;
    }

    @Override
    public IArray createArrayNoCopy(Object array) {
        IArray result = null;
        if (array instanceof IArray[]) {
            result = new NxsArray((IArray[]) array);
        } else if (array instanceof DataItem) {
            try {
                result = new NxsArray((DataItem) array);
            } catch (InvalidArrayTypeException e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
            }
        } else {
            DataItem dataset = null;
            try {
                dataset = new DataItem(array);
                result = new NxsArray(dataset);
            } catch (Exception e) {
                result = null;
            }

        }
        return result;
    }

    @Override
    public IAttribute createAttribute(String name, Object value) {
        return new NexusAttribute(NAME, name, value);
    }

    @Override
    public IDataItem createDataItem(IGroup parent, String shortName, IArray array) throws InvalidArrayTypeException {
        NxsDataItem result = null;
        if ((shortName == null) || shortName.trim().isEmpty()) {
            throw new InvalidArrayTypeException("Invalid shortName");
        } else if (array instanceof NxsArray) {
            NxsArray nxsArray = (NxsArray) array;
            IArray[] parts = nxsArray.getArrayParts();
            if (parts == null) {
                throw new InvalidArrayTypeException("Invalid array parts");
            } else {
                DataItem item = null;
                for (IArray tmp : parts) {
                    if (tmp instanceof NexusArray) {
                        item = ((NexusArray) tmp).getDataItem();
                        break;
                    }
                }
                if (item == null) {
                    item = new DataItem();
                    try {
                        item.initFromData(array.getStorage());
                    } catch (NexusException e) {
                        throw new InvalidArrayTypeException(e);
                    }
                }
                if (parent instanceof NxsGroup) {
                    NxsGroup group = (NxsGroup) parent;
                    if ((item.getPath() == null) || !shortName.equals(item.getPath().getDataItemName())) {
                        PathGroup pathGroup;
                        PathNexus pathNexus = group.getPathNexus();
                        if (pathNexus instanceof PathGroup) {
                            pathGroup = (PathGroup) pathNexus;
                        } else {
                            pathGroup = new PathGroup(pathNexus);
                        }
                        item.setPath(new PathData(pathGroup, shortName));
                    }
                    NxsDataset dataset = (NxsDataset) group.getDataset();
                    result = new NxsDataItem(new NexusDataItem(nxsArray.getFactoryName(), item, dataset == null ? null
                            : dataset.getNexusDataset()), parent, dataset);
                } else {
                    throw new InvalidArrayTypeException("Invalid parent group");
                }
            }
        } else {
            throw new InvalidArrayTypeException("Invalid IArray class");
        }
        return result;
    }

    @Override
    public IDataset createDatasetInstance(URI uri) throws Exception {
        return NxsDataset.instanciate(uri);
    }

    @Override
    public IArray createDoubleArray(double[] javaArray) {
        DataItem data;
        IArray result = null;
        try {
            data = new DataItem(javaArray);
            result = new NxsArray(data);
        } catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        } catch (NexusException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        }
        return result;
    }

    @Override
    public IArray createDoubleArray(double[] javaArray, int[] shape) {
        DataItem data;
        IArray result = null;
        try {
            data = new DataItem(javaArray);
            result = new NxsArray(data);
        } catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        } catch (NexusException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        }
        return result;
    }

    @Override
    public IDataset createEmptyDatasetInstance() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public IGroup createGroup(IGroup parent, String shortName) {
        String path_val = parent.getLocation();
        PathGroup path = new PathGroup(PathNexus.splitStringPath(path_val));
        NxsGroup group = new NxsGroup(parent, (PathNexus) path, (NxsDataset) parent.getDataset());

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
        synchronized (NxsDatasource.class) {
            if (detector == null) {
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
        } catch (NoResultException e) {
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
        return new NxsLogicalGroup(dataset, new Key(this, key.getName()));
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
    public IDictionary openDictionary(String filepath) throws FileAccessException {
        throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public IDictionary createDictionary() {
        throw new UnsupportedOperationException();
    }

    public static IArrayMath createArrayMath(NxsArray array) {
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
            boolean checkNeXusAPI = NexusDataset.checkNeXusAPI();
            if (!checkNeXusAPI) {
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
