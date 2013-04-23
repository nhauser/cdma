//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.engine.sql.array;

import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.concurrent.Semaphore;

import org.cdma.arrays.DefaultArrayMatrix;
import org.cdma.engine.sql.internal.DataArray;
import org.cdma.engine.sql.internal.ArrayDefaultAppender;
import org.cdma.engine.sql.utils.ISqlArrayAppender;
import org.cdma.exception.InvalidArrayTypeException;

public class SqlArray extends DefaultArrayMatrix {

	private DataArray<?> mDataArray; // Storage that permits to append data
	private Semaphore mLock;         // Lock the array storage for external users
	private int mNbRows;  // Nb of rows expected in the array
	private int mCurRow;  // Current processed row while appending data
	private int mColumn;  // Number of the column this array corresponds to in the SQL result set
	private ISqlArrayAppender mTreatment; // Appender for SqlArray
	
	static public SqlArray instantiate( String factory, ResultSet set, int column, int nbRows, ISqlArrayAppender appender ) throws SQLException {
		SqlArray array = null;
		
		if( appender == null ) {
			appender = new ArrayDefaultAppender();
		}
		
		synchronized( SqlArray.class ) {
			try {
				// Allocate the memory
				DataArray<?> memory = appender.allocate( set, column, nbRows );

				// Instantiate the array with the allocated memory
				array = new SqlArray(factory, memory);
				array.mColumn = column;
				array.mTreatment = appender;
			} catch( InvalidArrayTypeException e ) {
				throw new SQLException(e);
			}
		}
		
		return array;
	}
	
	static public SqlArray instantiate( String factory, ResultSet set, int column ) throws SQLException {
		return SqlArray.instantiate( factory, set, column, -1, null );
	}
	
	
	private SqlArray(String factory, DataArray<?> data ) throws InvalidArrayTypeException {
		super( factory, data.getValue() );
		mDataArray = data;
		mCurRow    = 0;
		mNbRows    = -1;
		mColumn    = -1;
		mLock = new Semaphore(1);
	}
	
	
	public void appendData(ResultSet resultSet) throws SQLException {
		if( mDataArray != null && mNbRows != 1 ) {
			ResultSetMetaData meta = resultSet.getMetaData();
			int type = meta.getColumnType(mColumn);
			mTreatment.append( mDataArray, resultSet, mColumn, mCurRow++, type);
		}
	}
	
	public void appendData(ResultSet resultSet, int type) throws SQLException {
		if( mDataArray != null && mNbRows != 1 ) {
			mTreatment.append( mDataArray, resultSet, mColumn, mCurRow++, type);
		}
	}
	
	@Override
	protected Object loadData() {
		Object result = null;
		lock();
		result = super.loadData();
		unlock();
		return result;
	}
	
	@Override
	protected Object getData() {
		Object result = null;
		lock();
		result = super.getData();
		unlock();
		return result;
	}
	
	@Override
	public void lock() {
		try {
			mLock.acquire();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void unlock() {
		mLock.release();
	}

	/**
	 * Without considering lock() it will return the first cell of the array when it has been loaded
	 */
	public Object getSample() {
		return mDataArray.getSample();
	}
}
