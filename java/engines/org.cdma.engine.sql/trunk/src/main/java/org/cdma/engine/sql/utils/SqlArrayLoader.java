//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.engine.sql.utils;

import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.sql.array.SqlArray;
import org.cdma.utilities.performance.PostTreatment;

public class SqlArrayLoader implements PostTreatment {
	private SqlCdmaCursor cursor;
	private SqlArray[] arrays;
	
	public SqlArrayLoader( SqlCdmaCursor cursor ) throws SQLException {
		this.cursor = cursor;
		
		// Get information from SQL cursor
		ResultSet set = cursor.getResultSet();
		ResultSetMetaData meta = set.getMetaData();
		int count = meta.getColumnCount();
		int nbRow = cursor.getNumberOfResults();
		String factory = cursor.getDataset().getFactoryName();
		
		// Get the SQL array post treatment
		ISqlArrayAppender appender = cursor.getAppender();
		
		// Prepare the internal array
		arrays = new SqlArray[count];
		for( int col = 1; col <= count; col++ ) {
			arrays[col - 1] = SqlArray.instantiate(factory, set, col, nbRow, appender);
			arrays[col - 1].appendData( set );
			arrays[col - 1].lock();
		}
	}

	@Override
	public String getName() {
		return "CDMA SQL array loading";
	}
	
	@Override
	public void process() {
		if( cursor != null && arrays != null ) {
			// Load data from SQL cursor
			ResultSet set = null;
			try {
				set = cursor.getResultSet();
				//ISqlArrayAppender treatment = cursor.getTreatment();
				ResultSetMetaData meta = set.getMetaData();
				int count = meta.getColumnCount();
				int[] types = new int[count];
				for( int i = 0; i < types.length; i++ ) {
					types[i] = meta.getColumnType(i + 1);
				}
				
				// Aggregate results from the result set
				SqlArray array; 
				while( cursor.next() ) {
					for( int col = 0; col < count; col++ ) {
						array = arrays[col];
						array.appendData( set, types[col] );
					}
				}
				set.close();
			} catch (SQLException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to load data from SQL cursor!", e);
			}
			for( SqlArray array : arrays ) {
				array.unlock();
			}
		}
	}



	public SqlArray[] getArrays() {
		return arrays;
	}

}
