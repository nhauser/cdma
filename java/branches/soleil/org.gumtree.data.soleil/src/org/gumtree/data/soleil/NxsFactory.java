package org.gumtree.data.soleil;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.net.URI;

import org.gumtree.data.IFactory;
import org.gumtree.data.IDatasource;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.dictionary.IPathParamResolver;
import org.gumtree.data.dictionary.IPathParameter;
import org.gumtree.data.dictionary.impl.Key;
import org.gumtree.data.dictionary.impl.Path;
import org.gumtree.data.dictionary.impl.PathParameter;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.exception.InvalidArrayTypeException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.interfaces.IKey;
import org.gumtree.data.soleil.array.NxsArray;
import org.gumtree.data.soleil.array.NxsArrayMatrix;
import org.gumtree.data.soleil.dictionary.NxsLogicalGroup;
import org.gumtree.data.soleil.dictionary.NxsPathParamResolver;
import org.gumtree.data.soleil.navigation.NxsAttribute;
import org.gumtree.data.soleil.navigation.NxsDataset;
import org.gumtree.data.soleil.navigation.NxsDatasetFile;
import org.gumtree.data.soleil.navigation.NxsGroup;
import org.gumtree.data.utils.Utilities.ParameterType;

import fr.soleil.nexus4tango.DataItem;
import fr.soleil.nexus4tango.PathGroup;
import fr.soleil.nexus4tango.PathNexus;

public class NxsFactory implements IFactory {
    private static NxsFactory factory;
    private static NxsDatasource detector;
    public final static String NAME = "org.gumtree.data.soleil.NxsFactory";
    public final static String LABEL = "SOLEIL's NeXus plug-in";
    public final static String DEBUG_INF = "CDMA_DEBUG_NXS";
    public NxsFactory() {
    }
    
    public static NxsFactory getInstance() {
        if( factory == null ) {
            synchronized (NxsFactory.class ) {
                if( factory == null ) {
                    factory  = new NxsFactory();
                    detector = new NxsDatasource();
                }
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
    		result = new NxsArrayMatrix( (IArray[]) storage );
    	}
    	else if( DataItem.class.equals(storage.getClass()) ) {
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
    	// http://stackoverflow.com/questions/219881/java-array-reflection-isarray-vs-instanceof
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
    		result = new NxsArrayMatrix( (IArray[]) array);
    	}
    	else if( array instanceof DataItem ) {
    		result = new NxsArray( (DataItem) array);
    	}
    	else {
    		DataItem dataset = null;
    		try {
        		dataset = new DataItem(array);
        	} catch( Exception e ) {}
        	result = new NxsArray(dataset);
    	}
		return result;
	}

    @Override
	public IAttribute createAttribute(String name, Object value) {
		return new NxsAttribute(name, value);
	}

    @Override
    public IDataItem createDataItem(IGroup parent, String shortName, IArray array) throws InvalidArrayTypeException {
		// TODO Auto-generated method stub
    	throw new InvalidArrayTypeException("not supported yet in plug-in!");
	}

    @Override
    public IDataset createDatasetInstance(URI uri) throws Exception {
		return NxsDataset.instanciate(new File(uri).getAbsolutePath());
	}

    @Override
    public IArray createDoubleArray(double[] javaArray) {
    	DataItem dataset;
    	try {
    		dataset = new DataItem(javaArray);
    	} catch( Exception e ) {
    		dataset = null;
    	}
		return new NxsArray(dataset);
	}

    @Override
    public IArray createDoubleArray(double[] javaArray, int[] shape) {
    	DataItem dataset;
    	try {
    		dataset = new DataItem(javaArray);
    	} catch( Exception e ) {
    		dataset = null;
    	}
		return new NxsArray(dataset);
	}

    @Override
	public IDataset createEmptyDatasetInstance() throws IOException {
		// TODO Auto-generated method stub
    	throw new IOException("not supported yet in plug-in!");
	}

    @Override
    public IGroup createGroup(IGroup parent, String shortName, boolean updateParent) {
    	String path_val = parent.getLocation();
    	PathGroup path = new PathGroup(PathNexus.splitStringPath(path_val));
		NxsGroup group = new NxsGroup( (NxsGroup) parent, (PathNexus) path, (NxsDatasetFile) parent.getDataset());
		
		return group;
    }

    @Override
    public IGroup createGroup(String shortName) throws IOException {
		// TODO Auto-generated method stub
    	throw new IOException("not supported yet in plug-in!");
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
		if( detector == null ) {
			detector = new NxsDatasource();
		}
		return detector;
	}
	
    @Override
    public IDataset openDataset(URI uri) throws FileAccessException {
		// TODO Auto-generated method stub
    	throw new FileAccessException("not supported yet in plug-in!");
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
		return new NxsLogicalGroup(dataset, key);
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
		IPathParamResolver result;
		if( path instanceof Path ) {
			result = new NxsPathParamResolver(this, (Path) path);
		}
		else {
			result = new NxsPathParamResolver( this, new Path(this, path.getValue() ) );
		}
		
		return result;
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
}
