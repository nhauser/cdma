//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.engine.sql.navigation;

import java.io.IOException;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.sql.internal.SqlConnector;
import org.cdma.engine.sql.utils.ISqlDataset;
import org.cdma.engine.sql.utils.SqlCdmaCursor;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IGroup;

public final class SqlQueryDataset implements ISqlDataset {
	private SqlConnector mConnector;
	private String mFactory;
	private boolean mNumericalDate;
	private SqlCdmaCursor cursor;
	private IGroup root;

	
	public SqlQueryDataset( String factoryName, String host, String user, String password, String driver, String dbName,
            String dbScheme, boolean rac, String query) {
		mFactory = factoryName;
        mConnector = new SqlConnector(host, user, password, driver, dbName, dbScheme, rac);
		mNumericalDate = false;
		try {
			open();
			executeQuery(query);
		} catch (IOException e) {
			Factory.getLogger().log(Level.WARNING, "Unable to open the dataset", e);
		}
		
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
		if (root == null){
			root = new SqlGroup(this, "", cursor);
		}
		return root;
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
	
	/**
	 * Return the SqlConnector that handles the database
	 */
	public SqlConnector getSqlConnector() {
		return mConnector;
	}
	
	/**
	 * Set the maximum time in seconds a driver can wait when attempting to log into a database.
	 * 
	 * @param timeout in seconds
	 */
	public void setLoginTimeout( int timeout ) {
		DriverManager.setLoginTimeout(timeout);
	}
	
	/**
	 * Get the maximum time in seconds a driver can wait when attempting to log into a database.
	 * 
	 * @return time in seconds
	 */
	public int getLogintimeout() {
		return DriverManager.getLoginTimeout();
	}
	
	/**
	 * Prepare the query for execution and return a SqlCdmaCursor.
	 * 
	 * @param query to be executed
	 * @note the execution is delayed: it will be triggered when a result will b e asked
	 */
	public void executeQuery( String query ) {
		executeQuery( query, new Object[] {} );
	}
	
	/**
	 * Prepare the query for execution and return a SqlCdmaCursor.
	 * 
	 * @param query to be executed
	 * @note the execution is delayed: it will be triggered when a result will b e asked
	 */
	public void executeQuery( String query, Object[] params ) {
		cursor = new SqlCdmaCursor(this, query, params);
		try {
			cursor.next();
		} catch (SQLException e) {
			Factory.getLogger().log(Level.WARNING, "Unable to get execute query: " + query, e);
		}
	}

	/**
	 * Should date be stored as long or should they be stored as string.
	 * @param numerical
	 */
	public void setNumericalDate(boolean numerical) {
		mNumericalDate = numerical;
	}
	
	/**
	 * Return true if date are stored as long, else they are stored as string.
	 */
	public boolean getNumericalDate() {
		return mNumericalDate;
	}
}
