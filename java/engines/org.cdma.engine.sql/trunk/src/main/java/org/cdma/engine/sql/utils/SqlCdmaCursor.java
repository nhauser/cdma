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

import java.io.IOException;
import java.lang.Thread.State;
import java.lang.ref.SoftReference;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.sql.array.SqlArray;
import org.cdma.engine.sql.internal.DataArray;
import org.cdma.engine.sql.internal.SqlConnector;
import org.cdma.engine.sql.navigation.SqlDataItem;
import org.cdma.engine.sql.navigation.SqlDataset;
import org.cdma.engine.sql.navigation.SqlGroup;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.utilities.performance.PostTreatmentManager;

public class SqlCdmaCursor {
	private String mQuery;
	private SqlDataset mDataset;
	private SoftReference<ResultSet> mResult;
	private int mCurRow;
	private PreparedStatement mStatQuery;
	private PreparedStatement mStatCount;
	private boolean mClose;
	private Object[] mParams;
	private int mNbRows;
	private boolean mInitialized;
	private ISqlArrayAppender mTreatment;
	
	public SqlCdmaCursor( SqlDataset dataset, String query ) {
		this( dataset, query, new Object[] {} );
	}
	
	public SqlCdmaCursor( SqlDataset dataset, String query, Object[] params ) {
		mDataset = dataset;
		mCurRow  = 0;
		mQuery   = query;
		mResult  = new SoftReference<ResultSet>(null);
		mClose   = false;
		mParams  = params;
		mNbRows  = -1;
		mInitialized = false;
	}

	public SqlGroup getGroup() throws SQLException {
		SqlGroup result = null;
		
		if( ! mClose ) {
			// Get the result set from the query
			ResultSet sql_set = getResultSet();
			
			// Create the corresponding group
			if( sql_set != null ) {
				result = new SqlGroup(mDataset, "Group_" + mCurRow, sql_set);
			}
		}
		return result;
	}
	
	public List<SqlDataItem> getDataItemList() {
		List<SqlDataItem> result = new ArrayList<SqlDataItem>();
		if( mCurRow == 0 && ! mClose ) {
			try {
				next();
			} catch (SQLException e) {
				Factory.getLogger().log(Level.WARNING, "Unable to initialize group's data items children", e);
			}
		}
		
		if( mNbRows >= 0 && mCurRow == 1 ) {
			try {
				ResultSet set = getResultSet();
				ResultSetMetaData meta = set.getMetaData();
				int count = meta.getColumnCount();
				
				// prepare columns names
				String[]      names = new String[count];
				SqlDataItem[] items = new SqlDataItem[count];
				
				// Create a SQL array loader
				SqlArrayLoader loader = new SqlArrayLoader(this);
				SqlArray[] arrays     = loader.getArrays();				
				
				// Get name for each items 
				for( int col = 1; col <= count; col++ ) {
					// Get the name of the column
					names[col - 1] = meta.getColumnName(col);
				}
				
				// Fill the array asynchronously
				Thread thread = PostTreatmentManager.launchParallelTreatment(loader);
				
				// Create items for each arrays
				for( int col = 1; col <= count; col++ ) {
					// Create the data item
					try {
						items[col - 1] = new SqlDataItem(mDataset.getFactoryName(), (SqlGroup) mDataset.getRootGroup(), names[ col -1 ]);
						result.add( items[col - 1] );
						items[col - 1].setCachedData(arrays[col - 1], false);
					} catch (InvalidArrayTypeException e) {
						Factory.getLogger().log( Level.SEVERE, "Unable to initialize data for the data item: " + names[col - 1], e);
					}
				}
			}
			catch( SQLException e ) {
				Factory.getLogger().log(Level.WARNING, "Unable to initialize group's children", e);
			}
		}
		return result;
	}
	
	public boolean next() throws SQLException {
		boolean result = false;
		
		if( ! mClose ) {
			// Get the result set from the query
			ResultSet sql_set = getResultSet();
	
			// Forward the cursor
			if( sql_set != null ) {
				result = sql_set.next();
				if( result ) {
					mCurRow++;
				}
				else {
					// Close both statement and result set
					close();
				}
			}
		}
		return result;
	}
	
	protected ResultSet getResultSet() throws SQLException {
		// Get the soft ref and check it is still available
		ResultSet sql_set = mResult.get();
		if( sql_set == null || sql_set.isClosed() ) {
			// Get the result set of the query
			sql_set = executeQuery( );

			// Set the cursor to the right position
			if( sql_set != null ) {
				initResultSet(sql_set);
			}
		}
		
		return sql_set;
	}
	
	public void close() throws SQLException {
		if( mStatQuery != null && ! mStatQuery.isClosed() ) {
			mStatQuery.close();
		}
		if( mStatCount != null && ! mStatCount.isClosed() ) {
			mStatCount.close();
		}
		mClose = true;
	}
	
	private ResultSet executeQuery() throws SQLException {
		ResultSet result = null;
		
		if( ! mInitialized ) {
			prepareStatement();
		}
		
		// Get the SQL connection
		SqlConnector sql_connector = mDataset.getSqlConnector();
		if (sql_connector != null) {
			// Count number of result
			if( mNbRows < 0 ) {
				setParams(mStatCount);
				ResultSet tmp = mStatCount.executeQuery();
				if( tmp.next() ) {
					mNbRows = tmp.getInt(1);
				}
			}
			
			// Execute the query
			setParams(mStatQuery);
			result = mStatQuery.executeQuery();
		}
		
		return result;
	}
	
	private void prepareStatement() throws SQLException {
		
		// Get the SQL connection
		SqlConnector sql_connector = mDataset.getSqlConnector();
		if (sql_connector != null) {
			try {
				Connection connection = sql_connector.getConnection();
				// Check statements are still valid
				if( mStatQuery == null || mStatQuery.isClosed() ) {
					// Create the query statement
					mStatQuery = connection.prepareStatement(mQuery);
					mStatQuery.setFetchSize(1000);
				}
				if( mStatCount == null || mStatCount.isClosed() ) {
					// Create the count statement
					mStatCount = connection.prepareStatement( "SELECT COUNT(*) FROM (" + mQuery + ")" );
					mStatCount.setFetchSize(1000);
				}
				
			} catch (IOException e) {
				mNbRows = -1;
				Factory.getLogger().log(Level.SEVERE, e.getMessage(), e);
				close();
			}
		}
				
		
		mInitialized = true;
	}
	
	private void initResultSet(ResultSet sql_set) throws SQLException {
		// Set the cursor on the right position
		if( sql_set.getType() == ResultSet.TYPE_FORWARD_ONLY ) {
			for( int i = 1; i < mCurRow; i++ ) {
				sql_set.next();
			}
		}
		else {
			sql_set.absolute(mCurRow);
		}

		mResult = new SoftReference<ResultSet>( sql_set );
	}
	
	private void setParams( PreparedStatement statement ) {
		if( mParams != null && mParams.length > 0 ) {
			Object param = null;
			for( int i = 0; i < mParams.length; i++ ) {
				param = mParams[i];
				try {
					statement.setObject( i + 1, param );
				} catch (SQLException e) {
					Factory.getLogger().log(Level.SEVERE, "Unable to prepare query!", e);
				}
			}
		}
	}
	
	@Override
	protected void finalize() throws Throwable {
		close();
		super.finalize();
	}
	
	public int getNumberOfResults() {
		return mNbRows;
	}

	/**
	 * Permit to set a ISqlArrayAppender implementation that will be executed on each appended data
	 * @param iSqlArrayAppender
	 */
	public void setAppender( ISqlArrayAppender iSqlArrayAppender ) {
		mTreatment = iSqlArrayAppender;
	}
	
	/**
	 * Permit to get the currently used ISqlArrayAppender implementation that will be executed 
	 * on each appended data
	 */
	public ISqlArrayAppender getAppender() {
		return mTreatment;
	}
	
	/**
	 * Returns the currently used SQL dataset
	 */
	protected SqlDataset getDataset() {
		return mDataset;
	}
}
