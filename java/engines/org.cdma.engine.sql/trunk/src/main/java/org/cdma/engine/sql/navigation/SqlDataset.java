package org.cdma.engine.sql.navigation;

import java.io.IOException;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.sql.internal.SqlConnector;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;

public final class SqlDataset implements IDataset {
	private SqlConnector mConnector;
	private String mFactory;

	
	public SqlDataset( String factoryName, String host, String user, String password ) {
		mFactory = factoryName;
		mConnector = new SqlConnector(host, user, password);
	}
	
	@Override
	public String getFactoryName() {
		return mFactory;
	}

	@Override
	public void close() throws IOException {
		if( mConnector != null ) {
			mConnector.close();
		}
	}

	@Override
	public IGroup getRootGroup() {
		return new SqlGroup(this, "", null);
	}

	@Override
	public LogicalGroup getLogicalRoot() {
		throw new NotImplementedException();
	}

	@Override
	public String getLocation() {
		return mConnector.getHost();
	}

	@Override
	public String getTitle() {
		String title = "";
		try {
			return mConnector.getConnection().getMetaData().getURL();
		} catch (SQLException e) {
			Factory.getLogger().log(Level.WARNING, "Unable to get dataset's title", e);
		} catch (IOException e) {
			Factory.getLogger().log(Level.WARNING, "Unable to get dataset's title", e);
		}
		return title;
	}

	@Override
	public void setLocation(String location) {
		throw new NotImplementedException();
	}

	@Override
	public void setTitle(String title) {
		throw new NotImplementedException();
	}

	@Override
	public boolean sync() throws IOException {
		boolean result = true;
		if( mConnector != null ) {
			try {
				mConnector.getConnection().commit();
			} catch (SQLException e) {
				throw new IOException(e);
			}
		}
		return result;
	}

	@Override
	public void open() throws IOException {
		mConnector.open();
	}

	@Override
	public void save() throws WriterException {
		try {
			sync();
		} catch (IOException e) {
			throw new WriterException(e);
		}
		
	}

	@Override
	public void saveTo(String location) throws WriterException {
		throw new NotImplementedException();
	}

	@Override
	public void save(IContainer container) throws WriterException {
		throw new NotImplementedException();
	}

	@Override
	public void save(String parentPath, IAttribute attribute) throws WriterException {
		throw new NotImplementedException();
	}

	@Override
	public boolean isOpen() {
		return mConnector.isOpen();
	}

	@Override
	public long getLastModificationDate() {
		throw new NotImplementedException();
	}
	
	public SqlConnector getSqlConnector() {
		return mConnector;
	}
	
	public void setLoginTimeout( int timeout ) {
		DriverManager.setLoginTimeout(timeout);
	}
	
	public int getLogintimeout() {
		return DriverManager.getLoginTimeout();
	}
	
	public SqlCdmaCursor execute_query( String query ) {
		return new SqlCdmaCursor(this, query);
	}
	
}
