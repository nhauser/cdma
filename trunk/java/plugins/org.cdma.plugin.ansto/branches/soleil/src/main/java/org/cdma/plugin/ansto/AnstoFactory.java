package org.cdma.plugin.ansto;

import java.io.File;
import java.io.IOException;
import java.net.URI;

import org.cdma.IFactory;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.engine.netcdf.array.NcArray;
import org.cdma.engine.netcdf.navigation.NcAttribute;
import org.cdma.engine.netcdf.navigation.NcDataItem;
import org.cdma.engine.netcdf.navigation.NcDictionary;
import org.cdma.engine.netcdf.navigation.NcGroup;
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

public class AnstoFactory implements IFactory {
    public static final String NAME  = "AnstoNetCDF";
    public static final String LABEL = "ANSTO's NetCDF plug-in";
    
    @Override
    public IDataset openDataset(URI uri) throws FileAccessException {
        return AnstoDataset.instantiate(uri);
    }

    @Override
    public IDictionary openDictionary(URI uri) throws FileAccessException {
        IDictionary dict = new NcDictionary(AnstoFactory.NAME);
        String file = uri.getPath();
        dict.readEntries(file);
        return dict;
    }

    @Override
    public IDictionary openDictionary(String filepath) throws FileAccessException {
        IDictionary dict = new NcDictionary(AnstoFactory.NAME);
        dict.readEntries(filepath);
        return dict;
    }

    @Override
    public IArray createArray(Class<?> clazz, int[] shape) {
        NcArray array = new NcArray(ucar.ma2.Array.factory(clazz, shape), AnstoFactory.NAME);
        return array;
    }

    @Override
    public IArray createArray(Class<?> clazz, int[] shape, Object storage) {
        NcArray array = new NcArray(ucar.ma2.Array.factory(clazz, shape, storage), AnstoFactory.NAME);
        return array;
    }

    @Override
    public IArray createArray(Object javaArray) {
        NcArray array = new NcArray(ucar.ma2.Array.factory(javaArray), AnstoFactory.NAME);
        return array;

    }

    @Override
    public IArray createStringArray(String string) {
        return createArray(String.class, new int[] { 1 }, new String[] { string });
    }

    @Override
    public IArray createDoubleArray(double[] javaArray) {
        return createArrayNoCopy(javaArray);
    }

    @Override
    public IArray createDoubleArray(double[] javaArray, int[] shape) {
        return createArray(Double.TYPE, shape, javaArray);
    }

    @Override
    public IArray createArrayNoCopy(Object javaArray) {
        int rank = 0;
        Class<?> componentType = javaArray.getClass();
        while (componentType.isArray()) {
                rank++;
                componentType = componentType.getComponentType();
        }
        /*
         * if( rank_ == 0) throw new
         * IllegalArgumentException("Array.factory: not an array"); if(
         * !componentType.isPrimitive()) throw new
         * UnsupportedOperationException(
         * "Array.factory: not a primitive array");
         */

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

    @Override
    public IDataItem createDataItem(IGroup parent, String shortName, IArray array) throws InvalidArrayTypeException {
        NcDataItem item = null;
        if( parent instanceof NcGroup ) {
            item = new NcDataItem((NcGroup) parent, shortName, array);
        }
        return item;
    }

    @Override
    public IGroup createGroup(IGroup parent, String shortName) {
        NcGroup group = null;
        if( parent instanceof NcGroup ) {
            group = new NcGroup((NcGroup) parent, shortName);
        }
        return group;
    }

    @Override
    public IGroup createGroup(String shortName) throws IOException {
        return createGroup(null, shortName);
    }

    @Override
    public LogicalGroup createLogicalGroup(IDataset dataset, IKey key) {
        return new LogicalGroup(key, dataset);
    }

    @Override
    public IAttribute createAttribute(String name, Object value) {
        IAttribute attribute = null;
        
        if( value instanceof Number ) {
            attribute = new NcAttribute(name, (Number) value, AnstoFactory.NAME);
        }
        else if( value instanceof String ) {
            attribute = new NcAttribute(name, (String) value, AnstoFactory.NAME);
        }
        else if( value instanceof Boolean ) {
            attribute = new NcAttribute(name, (Boolean) value, AnstoFactory.NAME);
        }
        
        return attribute;
    }

    @Override
    public IDataset createDatasetInstance(URI uri) throws Exception {
        String path = uri.getPath();
        File file = new File(path);
        
        if( !file.exists() && !file.isDirectory() ) {
            return new AnstoDataset( path );
        }
        else {
            return AnstoDataset.instantiate(uri);
        }
    }

    @Override
    public IDataset createEmptyDatasetInstance() throws IOException {
        return new AnstoDataset(null);
    }

    @Override
    public IKey createKey(String name) {
        return new Key(this, name);
    }

    @Override
    public Path createPath(String path) {
        return new Path(this, path);
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
        return AnstoDataSource.getInstance();
    }

    @Override
    public IDictionary createDictionary() {
        return new NcDictionary(AnstoFactory.NAME);
    }

}
