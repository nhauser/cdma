package org.gumtree.data.engine.jnexus;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.net.URI;

import org.gumtree.data.IDatasource;
import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.dictionary.IPathParamResolver;
import org.gumtree.data.dictionary.IPathParameter;
import org.gumtree.data.dictionary.impl.Key;
import org.gumtree.data.dictionary.impl.Path;
import org.gumtree.data.dictionary.impl.PathParameter;
import org.gumtree.data.engine.jnexus.array.NexusArray;
import org.gumtree.data.engine.jnexus.navigation.NexusAttribute;
import org.gumtree.data.engine.jnexus.navigation.NexusDataset;
import org.gumtree.data.engine.jnexus.navigation.NexusGroup;
import org.gumtree.data.engine.jnexus.utils.NexusArrayMath;
import org.gumtree.data.engine.jnexus.utils.NexusArrayUtils;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.exception.InvalidArrayTypeException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.interfaces.IKey;
import org.gumtree.data.math.IArrayMath;
import org.gumtree.data.utils.IArrayUtils;
import org.gumtree.data.utils.Utilities.ParameterType;

import fr.soleil.nexus4tango.DataItem;
import fr.soleil.nexus4tango.PathGroup;
import fr.soleil.nexus4tango.PathNexus;

public final class NexusFactory implements IFactory {
    private static NexusFactory factory;
    private static NexusDatasource detector;
    public static final String NAME = "org.gumtree.data.engine.nexus";
    public static final String LABEL = "NeXus engine plug-in";
    public static final String DEBUG_INF = "CDMA_DEBUG_NXS";
    public static final String ERR_NOT_SUPPORTED = "Method not supported yet in this plug-in!";
    
    public NexusFactory() {
    }
    
    public static NexusFactory getInstance() {
        synchronized (NexusFactory.class ) {
            if( factory == null ) {
                factory  = new NexusFactory();
                detector = new NexusDatasource();
            }
        }
        return factory;
    }

    @Override
	public IArray createArray(Class<?> clazz, int[] shape) {
    	Object o = java.lang.reflect.Array.newInstance(clazz, shape);
		return new NexusArray( o, shape);
	}

    @Override
	public IArray createArray(Class<?> clazz, int[] shape, Object storage) {
    	IArray result = null;
    	if( DataItem.class.equals(storage.getClass()) ) {
    		result = new NexusArray( (DataItem) storage );
    	}
    	else {
    		result = new NexusArray( storage, shape);
    	}
		return result;
	}

    @Override
	public IArray createArray(Object javaArray) {
    	IArray result = null;
    	// [ANSTO][Tony][2011-08-31] testing isArray may be slow
    	// http://stackoverflow.com/questions/219881/java-array-reflection-isarray-vs-instanceof
    	if (javaArray != null && javaArray.getClass().isArray()) {
    		int size = Array.getLength(javaArray);
    		result = new NexusArray(javaArray, new int[] { size });
    	}
		return result;
	}

    @Override
	public IArray createArrayNoCopy(Object array) {
    	IArray result = null;
    	if( array instanceof DataItem ) {
    		result = new NexusArray( (DataItem) array);
    	}
    	else {
    		DataItem dataset = null;
    		try {
        		dataset = new DataItem(array);
        		result = new NexusArray(dataset);
        	} catch( Exception e ) {}
        	
    	}
		return result;
	}

    @Override
	public IAttribute createAttribute(String name, Object value) {
		return new NexusAttribute(name, value);
	}

    @Override
    public IDataItem createDataItem(IGroup parent, String shortName, IArray array) throws InvalidArrayTypeException {
		// TODO Auto-generated method stub
    	throw new InvalidArrayTypeException(ERR_NOT_SUPPORTED);
	}

    @Override
    public IDataset createDatasetInstance(URI uri) throws Exception {
		return new NexusDataset(new File(uri));
	}

    @Override
    public IArray createDoubleArray(double[] javaArray) {
    	IArray array = null;
    	try {
    		DataItem dataset = new DataItem(javaArray);
    		array = new NexusArray(dataset);
    	} catch( Exception e ) {
    	}
		return array;
	}

    @Override
    public IArray createDoubleArray(double[] javaArray, int[] shape) {
    	IArray array = null;
    	try {
    		DataItem dataset = new DataItem(javaArray);
    		array = new NexusArray(dataset);
    	} catch( Exception e ) {
    	}
		return array;
	}

    @Override
	public IDataset createEmptyDatasetInstance() throws IOException {
		// TODO Auto-generated method stub
    	throw new IOException(ERR_NOT_SUPPORTED);
	}

    @Override
    public IGroup createGroup(IGroup parent, String shortName, boolean updateParent) {
    	String path_val = parent.getLocation();
    	PathGroup path = new PathGroup(PathNexus.splitStringPath(path_val));
		NexusGroup group = new NexusGroup( (NexusGroup) parent, (PathNexus) path, (NexusDataset) parent.getDataset());
		
		return group;
    }

    @Override
    public IGroup createGroup(String shortName) throws IOException {
		// TODO Auto-generated method stub
    	throw new IOException(ERR_NOT_SUPPORTED);
	}

    @Override
    public IArray createStringArray(String string) {
		// TODO Auto-generated method stub
    	return null;
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
		synchronized (NexusDatasource.class ) {
			if( detector == null ) {
				detector = new NexusDatasource();
			}
		}
		return detector;
	}
	
    @Override
    public IDataset openDataset(URI uri) throws FileAccessException {
		// TODO Auto-generated method stub
    	throw new FileAccessException(ERR_NOT_SUPPORTED);
	}

	@Override
	public IKey createKey(String keyName) {
		return new Key(this, keyName);
	}

	@Override
	public IPathParameter createPathParameter(ParameterType type, String name, Object value) {
		return new PathParameter(this, type, name, value);
	}

	@Override
	public ILogicalGroup createLogicalGroup(IDataset dataset, IKey key) {
		return null;
	}

	@Override
	public IPath createPath(String path) {
		return new Path(this, path);
	}

	@Override
	public String getPathSeparator() {
		return "/";
	}

	@Override
	public IPathParamResolver createPathParamResolver(IPath path) {
		return null;
	}

	@Override
	public IDictionary openDictionary(URI uri) throws FileAccessException {
		throw new UnsupportedOperationException();
	}

	@Override
	public IDictionary openDictionary(String filepath)
			throws FileAccessException {
		throw new UnsupportedOperationException();
	}

	@Override
	public IDictionary createDictionary() {
		throw new UnsupportedOperationException();
	}
	
	public static IArrayUtils createArrayUtils(NexusArray array) {
		return new NexusArrayUtils(array);
	}
	
	public static IArrayMath createArrayMath(NexusArray array) {
		return new NexusArrayMath(array);
	}
}
