package org.cdma.engine.sql.utils;

import java.sql.ResultSet;
import java.sql.SQLException;

import org.cdma.engine.sql.internal.DataArray;

public interface ISqlArrayAppender {
	/**
	 * Instantiate a DataArray object according the type of the ResultSet at the
	 * given column index.
	 * 
	 * @param set
	 *            SQL resultset to be analyzed for memory allocation type
	 * @param column
	 *            number to consider for the memory allocation
	 * @param nbRows
	 *            number of rows to allocate
	 * @throws SQLException
	 */
	DataArray<?> allocate(ResultSet set, int column, int nbRows) throws SQLException;
	
	/**
	 * Instantiate a DataArray object according the type of the ResultSet at the
	 * given column index.
	 * 
	 * @param set
	 *            SQL resultset to be analyzed for memory allocation type
	 * @param column
	 *            number to consider for the memory allocation
	 * @param nbRows
	 *            number of rows to allocate
	 * @throws SQLException
	 */
	void append(DataArray<?> array, ResultSet set, int column, int row, int type) throws SQLException;
}
