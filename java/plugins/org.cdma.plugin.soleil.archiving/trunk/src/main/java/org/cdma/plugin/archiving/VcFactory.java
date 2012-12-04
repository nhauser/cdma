package org.cdma.plugin.archiving;

import java.io.IOException;
import java.lang.reflect.Array;
import java.net.URI;

import org.cdma.IFactory;
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
import org.cdma.plugin.archiving.array.VcArray;
import org.cdma.plugin.archiving.navigation.VcDataItem;
import org.cdma.plugin.archiving.navigation.VcDataset;

public class VcFactory implements IFactory {

	public static final String NAME = "SoleilArchiving";
    public static final String LABEL = "SOLEIL's Archiving plug-in";
	public static final String API_VERS = "3.2.1";
	public static final String PLUG_VERS = "1.0.0";

	@Override
	public IDataset openDataset(URI uri) {
		VcDataset dataset = new VcDataset(uri);
		dataset.open();
		return dataset;
	}

	@Override
	public IDictionary openDictionary(URI uri) throws FileAccessException {
		throw new NotImplementedException();
	}

	@Override
	public IDictionary openDictionary(String filepath)
			throws FileAccessException {
		throw new NotImplementedException();
	}

	@Override
	public IArray createArray(Class<?> clazz, int[] shape) {
		Object o = java.lang.reflect.Array.newInstance(clazz, shape);
		return new VcArray(o, shape);
	}

	@Override
	public IArray createArray(Class<?> clazz, int[] shape, Object storage) {
		throw new NotImplementedException();
	}

	@Override
	public IArray createArray(Object javaArray) {
		IArray result = null;
		if (javaArray != null && javaArray.getClass().isArray()) {
			int size = Array.getLength(javaArray);
			result = new VcArray(javaArray, new int[] { size });
		}
		return result;
	}

	@Override
	public IArray createStringArray(String string) {
		throw new NotImplementedException();
	}

	@Override
	public IArray createDoubleArray(double[] javaArray) {
		throw new NotImplementedException();
	}

	@Override
	public IArray createDoubleArray(double[] javaArray, int[] shape) {
		throw new NotImplementedException();
	}

	@Override
	public IArray createArrayNoCopy(Object javaArray) {
		throw new NotImplementedException();
	}

	@Override
	public IDataItem createDataItem(IGroup parent, String shortName, IArray array) throws InvalidArrayTypeException {
		IDataset dataset = parent != null ? parent.getDataset() : null;
		IDataItem result = new VcDataItem(shortName, dataset, parent, null);
		return result;
	}

	@Override
	public IGroup createGroup(String shortName) throws IOException {
		throw new NotImplementedException();
	}

	@Override
	public LogicalGroup createLogicalGroup(IDataset dataset, IKey key) {
		return new LogicalGroup(key, dataset);
	}

	@Override
	public IAttribute createAttribute(String name, Object value) {
		throw new NotImplementedException();
	}

	@Override
	public IDataset createDatasetInstance(URI uri) throws Exception {
		return new VcDataset(uri);
	}

	@Override
	public IDataset createEmptyDatasetInstance() throws IOException {
		throw new NotImplementedException();
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
		return VcFactory.NAME;
	}

	@Override
	public String getPluginLabel() {
		return LABEL;
	}

	@Override
	public IDatasource getPluginURIDetector() {
		return VcDataSource.getInstance();
	}

	@Override
	public IDictionary createDictionary() {
		throw new NotImplementedException();
	}

	@Override
	public IGroup createGroup(IGroup parent, String shortName) {
		throw new NotImplementedException();
	}

	@Override
	public String getPluginVersion() {
		return PLUG_VERS;
	}

	@Override
	public String getCDMAVersion() {
		return API_VERS;
	}

}
