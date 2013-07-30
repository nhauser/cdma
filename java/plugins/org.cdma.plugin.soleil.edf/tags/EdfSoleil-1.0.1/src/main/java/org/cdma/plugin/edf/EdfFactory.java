package org.cdma.plugin.edf;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.arrays.DefaultArrayMatrix;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDatasource;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.plugin.edf.navigation.EdfAttribute;
import org.cdma.plugin.edf.navigation.EdfDataItem;
import org.cdma.plugin.edf.navigation.EdfDataset;
import org.cdma.plugin.edf.navigation.EdfGroup;

public class EdfFactory implements IFactory {

    public static final String NAME = "SoleilEDF";
    public static final String LABEL = "SOLEIL's EDF plug-in";
    public static final String DEBUG_INF = "CDMA_DEBUG";
    private static final String CDMA_VERSION = "3.2.5";
    private static final String PLUG_VERSION = "1.0.0";
    private static final String DESC = "Manages EDF data files";
    private static EdfFactory factory;
    private EdfDatasource detector;

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
            result = new DefaultArrayMatrix(EdfFactory.NAME, o);
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
            result = new DefaultArrayMatrix(EdfFactory.NAME, storage);
        }
        catch (InvalidArrayTypeException e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public IArray createArray(Object javaArray) {
        IArray result = null;
        if (javaArray != null && javaArray.getClass().isArray()) {
            try {
                result = new DefaultArrayMatrix(EdfFactory.NAME, javaArray);
            }
            catch (InvalidArrayTypeException e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
            }
        }

        return result;
    }

    @Override
    public IArray createStringArray(String string) {
        throw new NotImplementedException();
    }

    @Override
    public IArray createDoubleArray(double[] javaArray) {
        IArray result = null;
        try {
            result = new DefaultArrayMatrix(EdfFactory.NAME, javaArray);
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
            result = new DefaultArrayMatrix(EdfFactory.NAME, javaArray);
        }
        catch (InvalidArrayTypeException e) {

            e.printStackTrace();
        }
        return result;
    }

    @Override
    public IArray createArrayNoCopy(Object javaArray) {
        throw new NotImplementedException();
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
        return new EdfAttribute(name, value);
    }

    @Override
    public IDataset createDatasetInstance(URI uri) throws Exception {
        return new EdfDataset(uri.getPath());
    }

    @Override
    public IDataset createEmptyDatasetInstance() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public IKey createKey(String keyName) {
        return new Key(EdfFactory.getInstance(), keyName);
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    @Deprecated
    public IDictionary openDictionary(URI uri) throws FileAccessException {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public IDictionary openDictionary(String filepath) throws FileAccessException {
        throw new NotImplementedException();
    }

    @Override
    public IGroup createGroup(IGroup parent, String shortName) {
        throw new NotImplementedException();
    }

    @Override
    public LogicalGroup createLogicalGroup(IDataset dataset, IKey key) {
        throw new NotImplementedException();
    }

    @Override
    public Path createPath(String path) {
        return new Path(this, path);
    }

    @Override
    public String getPluginLabel() {
        return LABEL;
    }

    @Override
    public IDatasource getPluginURIDetector() {
        synchronized (EdfDatasource.class) {
            if (detector == null) {
                detector = EdfDatasource.getInstance();
            }
        }
        return detector;
    }

    @Override
    @Deprecated
    public IDictionary createDictionary() {
        throw new NotImplementedException();
    }


    @Override
    public void processPostRecording() {
        // NOTHING TO DO
    }

    @Override
    public boolean isLogicalModeAvailable() {
        return true;
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

