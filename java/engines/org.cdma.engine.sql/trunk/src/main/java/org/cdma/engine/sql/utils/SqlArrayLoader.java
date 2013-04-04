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
	
	public SqlArrayLoader( SqlArray[] arrays, SqlCdmaCursor cursor ) {
		this.arrays = arrays;
		this.cursor = cursor;
		
		
		if( this.arrays != null && this.cursor != null ) {
			for( SqlArray array : this.arrays ) {
				array.lock();
			}
		}
		
	}
	
	@Override
	public void process() {
		if( cursor != null && arrays != null ) {
			// Load data from SQL cursor
			try {
				ResultSet set = cursor.getResultSet();
				ResultSetMetaData meta = set.getMetaData();
				int count = meta.getColumnCount();
				
				// Aggregate results from the result set
				SqlArray array; 
				while( cursor.next() ) {
					for( int col = 1; col <= count; col++ ) {
						array = arrays[col - 1];
						array.appendData( set );
					}
				}
			} catch (SQLException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to load data from SQL cursor!", e);
			}
			
			for( SqlArray array : arrays ) {
				array.unlock();
			}
		}
	}

}
