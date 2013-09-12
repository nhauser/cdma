package org.cdma.engine.sql;

import java.io.IOException;
import java.net.URI;
import java.util.List;

import org.cdma.IFactory;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.engine.sql.navigation.SqlDataItem;
import org.cdma.engine.sql.navigation.SqlDataset;
import org.cdma.engine.sql.navigation.SqlQueryDataset;
import org.cdma.engine.sql.utils.SqlCdmaCursor;
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

public final class SqlFactory implements IFactory {

	private static SqlFactory factory;
	public static final String NAME = "SqlEngine";
	public static final String LABEL = "SQL Engine";
	public static final String DEBUG_INF = "CDMA_DEBUG";
	private static final String CDMA_VERSION = "3.2.5";
	private static final String PLUG_VERSION = "1.4.13";
	private static final String DESC = "Manages SQL Queries";

	public SqlFactory() {
	}

	public static SqlFactory getInstance() {
		synchronized (SqlFactory.class) {
			if (factory == null) {
				factory = new SqlFactory();
			}
		}
		return factory;
	}

	@Override
	public IDataset openDataset(URI uri) throws FileAccessException {
		IDataset dataset = null;
		String userInfo = uri.getUserInfo();
		String[] infos = userInfo.split(":");
		if (infos.length == 2) {
			dataset = new SqlDataset(NAME, uri.getHost(), infos[0], infos[1]);
		} else {
			throw new FileAccessException();
		}
		return dataset;
	}

	@Override
	@Deprecated
	public IDictionary openDictionary(URI uri) throws FileAccessException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	@Deprecated
	public IDictionary openDictionary(String filepath)
			throws FileAccessException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArray createArray(Class<?> clazz, int[] shape) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArray createArray(Class<?> clazz, int[] shape, Object storage) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArray createArray(Object javaArray) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArray createStringArray(String string) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArray createDoubleArray(double[] javaArray) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArray createDoubleArray(double[] javaArray, int[] shape) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArray createArrayNoCopy(Object javaArray) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IDataItem createDataItem(IGroup parent, String shortName,
			IArray array) throws InvalidArrayTypeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IGroup createGroup(IGroup parent, String shortName) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IGroup createGroup(String shortName) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public LogicalGroup createLogicalGroup(IDataset dataset, IKey key) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IAttribute createAttribute(String name, Object value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IDataset createDatasetInstance(URI uri) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IDataset createEmptyDatasetInstance() throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IKey createKey(String name) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Path createPath(String path) {
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
	public String getPluginDescription() {
		return DESC;
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
	public String getPluginVersion() {
		return PLUG_VERSION;
	}

	@Override
	public String getCDMAVersion() {
		return CDMA_VERSION;
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

	public static void main(String args[]) throws Exception {
		String uriStr = "jdbc:oracle:thin:@LUTIN:1521:TEST11";

		SqlQueryDataset dataset = new SqlQueryDataset(NAME, uriStr, "HDB",
				"HDB", "select * from ADT");
		IGroup root = dataset.getRootGroup();
		if (root == null) {
			System.out.println("Something is Wrong. root is null");
		}
		List<IDataItem> items = root.getDataItemList();
		for (IDataItem sqlDataItem : items) {
			System.out.println(sqlDataItem.getName());
		}
	}

}
