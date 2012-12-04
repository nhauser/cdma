package org.cdma.engine.sql.navigation;

import java.io.IOException;
import java.lang.ref.SoftReference;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.engine.sql.array.SqlArray;
import org.cdma.engine.sql.internal.SqlConnector;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IDataItem;

public class SqlCdmaCursor {
	private String mQuery;
	private SqlDataset mDataset;
	private SoftReference<ResultSet> mResult;
	private int mCurRow;
	private Statement mStatement;
	private boolean mClose;
	private String[] mParams;
	private int mNbRows;

	public SqlCdmaCursor( SqlDataset dataset, String query, String[] params ) {
		mDataset = dataset;
		mCurRow  = 0;
		mQuery   = query;
		mResult  = new SoftReference<ResultSet>(null);
		mClose   = false;
		mParams  = params;
		mNbRows  = -1; 
	}
	
	public SqlCdmaCursor( SqlDataset dataset, String query) {
		this( dataset, query, null );
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
				Factory.getLogger().log(Level.WARNING, "Unable to initialize group's children", e);
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
				for( int col = 1; col <= count; col++ ) {
					names[col - 1] = meta.getColumnName(col);

					// Prepare the internal array
					SqlArray array = new SqlArray(mDataset.getFactoryName(), set, col, mNbRows);
					
					// Create the data item
					try {
						items[col - 1] = new SqlDataItem(mDataset.getFactoryName(), (SqlGroup) mDataset.getRootGroup(), names[ col -1 ]);
						result.add( items[col - 1] );
						items[col - 1].setCachedData(array, false);
					} catch (InvalidArrayTypeException e) {
						Factory.getLogger().log( Level.SEVERE, "Unable to initialize data for the data item", e);
					}
				}

				// Agregate results from the resultset
				SqlArray array; 
				while( next() ) {
					for( int col = 1; col <= count; col++ ) {
						array = (SqlArray) items[col - 1].getData();
						array.appendData( getResultSet() );
					}
				}
			}
			catch( SQLException e ) {
				Factory.getLogger().log(Level.WARNING, "Unable to initialize group's children", e);
			} catch (IOException e) {
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
		if( ! mStatement.isClosed() ) {
			mStatement.close();
		}
		mClose = true;
	}
	
	private ResultSet executeQuery() throws SQLException {
		ResultSet result = null;
		
		// Get the SQL connection
		SqlConnector sql_connector = mDataset.getSqlConnector();
		if (sql_connector != null) {
			try {
				Connection connection = sql_connector.getConnection();
				
				// Check statement is still valid
				if( mStatement == null || mStatement.isClosed() ) {
					if( mParams == null ) {
						mStatement = connection.createStatement();
					}
					else {
						mStatement = connection.prepareStatement(mQuery);
						int i = 1;
						for( String param : mParams ) {
							((PreparedStatement) mStatement).setString(i++, param);
						}
					}
				}
				
				// Count number of result
				if( mNbRows < 0 ) {
					ResultSet tmp = mStatement.executeQuery( "SELECT COUNT(*) FROM (" + mQuery + ")" );
					if( tmp.next() ) {
						mNbRows = tmp.getInt(1);
					}
				}
				
				// Execute the query
				result = mStatement.executeQuery(mQuery);
			} catch (IOException e) {
				mNbRows = -1;
				Factory.getLogger().log(Level.SEVERE, e.getMessage(), e);
				close();
			}
		}
		
		return result;
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
	
	@Override
	protected void finalize() throws Throwable {
		close();
		super.finalize();
	}
}
