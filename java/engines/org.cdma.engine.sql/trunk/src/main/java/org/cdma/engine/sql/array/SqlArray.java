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
import java.sql.SQLException;

import org.cdma.arrays.DefaultArrayMatrix;
import org.cdma.engine.sql.internal.DataArray;
import org.cdma.exception.InvalidArrayTypeException;

public class SqlArray extends DefaultArrayMatrix {


	private DataArray<?>  mDataArray;
	private int     mNbRows;
	private int     mCurRow;
	private int     mColumn;
	
	static public SqlArray instantiate( String factory, ResultSet set, int column, int nbRows ) throws SQLException {
		SqlArray array = null;
		synchronized( SqlArray.class ) {
			try {
				// Allocate the memory
				DataArray<?> memory = DataArray.allocate( set, column, nbRows );

				// Instantiate the array with the allocated memory
				array = new SqlArray(factory, memory);
				array.mColumn = column;
			} catch( InvalidArrayTypeException e ) {
				throw new SQLException(e);
			}
		}
		
		return array;
	}
	
	static public SqlArray instantiate( String factory, ResultSet set, int column ) throws SQLException {
		return SqlArray.instantiate( factory, set, column, -1 );
	}
	
	
	public SqlArray(String factory, DataArray<?> data ) throws InvalidArrayTypeException {
		super( factory, data.getValue() );
		mDataArray = data;
		mCurRow    = 0;
		mNbRows    = -1;
		mColumn    = -1;
	}
	
	public void appendData(ResultSet resultSet) throws SQLException {
		if( mDataArray != null && mNbRows != 1 ) {
			mDataArray.append( resultSet, mColumn, mCurRow++);
		}
	}
}
