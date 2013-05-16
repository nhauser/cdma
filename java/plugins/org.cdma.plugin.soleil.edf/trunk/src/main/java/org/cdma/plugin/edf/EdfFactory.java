package org.cdma.plugin.edf;

import java.io.File;
import java.io.IOException;
import java.net.URI;

import org.cdma.IFactory;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDatasource;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.plugin.edf.array.BasicArray;
import org.cdma.plugin.edf.navigation.EdfAttribute;
import org.cdma.plugin.edf.navigation.EdfDataItem;
import org.cdma.plugin.edf.navigation.EdfDataset;
import org.cdma.plugin.edf.navigation.EdfGroup;
import org.cdma.plugin.edf.navigation.EdfKey;

public class EdfFactory implements IFactory {

    public static final String NAME = "EDF";
    public static final String LABEL = "EDF plug-in";
    public static final String DEBUG_INF = "CDMA_DEBUG";
    private static final String CDMA_VERSION = "3.2.5";
    private static final String PLUG_VERSION = "1.4.13";
    private static final String DESC = "Manages EDF data files";
    private static EdfFactory factory;

    public EdfFactory() {
    }

    public static EdfFactory getInstance() {
        synchronized (EdfFactory.class) {
            if (factory == null) {
                factory = new EdfFactory();
            }
        }
        return factory;
    }

    @Override
    public IDataset openDataset(URI uri) throws FileAccessException {
        EdfDataset dataset = new EdfDataset(uri.getPath());
        try {
            dataset.open();
        }
        catch (IOException e) {
            throw new FileAccessException(e.getMessage(), e);
        }
        return dataset;
    }

    @Override
    public IArray createArray(Class<?> clazz, int[] shape) {
        IArray result = null;
        Object o = java.lang.reflect.Array.newInstance(clazz, shape);
        try {
            result = new BasicArray(o, shape);
        }
        catch (InvalidArrayTypeException e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public IArray createArray(Class<?> clazz, int[] shape, Object storage) {
        IArray result = null;
        try {
            result = new BasicArray(storage, shape);
        }
        catch (InvalidArrayTypeException e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public IArray createArray(Object javaArray) {
        if (javaArray instanceof IArray) {
            return (IArray) javaArray;
        }
        // TODO
        return null;
    }

    @Override
    public IArray createStringArray(String string) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IArray createDoubleArray(double[] javaArray) {
        IArray result = null;
        try {
            result = new BasicArray(javaArray, new int[] { javaArray.length });
        }
        catch (InvalidArrayTypeException e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public IArray createDoubleArray(double[] javaArray, int[] shape) {
        IArray result = null;
        try {
            result = new BasicArray(javaArray, shape);
        }
        catch (InvalidArrayTypeException e) {

            e.printStackTrace();
        }
        return result;
    }

    @Override
    public IArray createArrayNoCopy(Object javaArray) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IDataItem createDataItem(IGroup parent, String shortName, IArray array)
            throws InvalidArrayTypeException {
        EdfDataItem dataitem = new EdfDataItem(shortName, array);
        dataitem.setParent(parent);
        return dataitem;
    }

    @Override
    public IGroup createGroup(String shortName) throws IOException {
        EdfGroup grp = new EdfGroup(new File(shortName));
        return grp;
    }

    @Override
    public IAttribute createAttribute(String name, Object value) {
        return new EdfAttribute(name, createArray(value));
    }

    @Override
    public IDataset createDatasetInstance(URI uri) throws Exception {
        return new EdfDataset(uri.getPath());
    }

    @Override
    public IDataset createEmptyDatasetInstance() throws IOException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IKey createKey(String keyName) {
        return new EdfKey(keyName);
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    @Deprecated
    public IDictionary openDictionary(URI uri) throws FileAccessException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    @Deprecated
    public IDictionary openDictionary(String filepath) throws FileAccessException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IGroup createGroup(IGroup parent, String shortName) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public LogicalGroup createLogicalGroup(IDataset dataset, IKey key) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Path createPath(String path) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public String getPluginLabel() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IDatasource getPluginURIDetector() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    @Deprecated
    public IDictionary createDictionary() {
        // TODO Auto-generated method stub
        return null;
    }


    @Override
    public void processPostRecording() {
        // TODO Auto-generated method stub

    }

    @Override
    public boolean isLogicalModeAvailable() {
        // TODO Auto-generated method stub
        return false;
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
        return EdfFactory.DESC;
    }

}
