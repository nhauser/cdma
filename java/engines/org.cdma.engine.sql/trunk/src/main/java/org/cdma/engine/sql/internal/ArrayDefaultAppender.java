package org.cdma.engine.sql.internal;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.sql.Blob;
import java.sql.Clob;
import java.sql.Date;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Time;
import java.sql.Timestamp;
import java.sql.Types;

import org.cdma.engine.sql.utils.ISqlArrayAppender;

public class ArrayDefaultAppender implements ISqlArrayAppender {

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
	public DataArray<?> allocate(ResultSet set, int column, int nbRows) throws SQLException {
		DataArray<?> array;
		int rows = nbRows;
		if (nbRows <= 0) {
			rows = 1;
		}

		ResultSetMetaData meta = set.getMetaData();
		int nb_column = meta.getColumnCount();
		if (column <= nb_column) {
			int type = meta.getColumnType(column);
			switch (type) {
				case Types.ARRAY:
					array = DataArray.allocate(set.getArray(column).getArray(), rows);
					break;
				case Types.DATALINK:
				case Types.CHAR:
				case Types.VARCHAR:
				case Types.LONGVARCHAR:
				case Types.NCHAR:
				case Types.LONGNVARCHAR:
				case Types.NVARCHAR: {
					String value = set.getString(column);
					if( value == null ) {
						value = new String();
					}
					array = DataArray.allocate(value, rows);
					break;
				}
				case Types.BINARY:
				case Types.VARBINARY:
				case Types.LONGVARBINARY: {
					byte[] value = set.getBytes(column);
					if( value == null ) {
						value = new byte[] {};
					}
					array = DataArray.allocate(value, rows);
					break;
				}
				case Types.BIT:
				case Types.BOOLEAN:
					array = DataArray.allocate(set.getBoolean(column), rows);
					break;
				case Types.TINYINT:
				case Types.SMALLINT:
					array = DataArray.allocate(set.getShort(column), rows);
					break;
				case Types.INTEGER:
					array = DataArray.allocate(set.getInt(column), rows);
					break;
				case Types.BIGINT:
					array = DataArray.allocate(set.getLong(column), rows);
					break;
				case Types.DOUBLE:
					array = DataArray.allocate(set.getDouble(column), rows);
					break;
				case Types.FLOAT:
				case Types.REAL:
					array = DataArray.allocate(set.getFloat(column), rows);
					break;
				case Types.DECIMAL:
				case Types.NUMERIC:
					BigDecimal decimal = set.getBigDecimal(column);
					if (decimal != null) {
						if (decimal.scale() == 0) {
							array = DataArray.allocate(decimal.intValue(), rows);
						} else {
							array = DataArray.allocate(decimal.doubleValue(), rows);
						}
					} else {
						array = DataArray.allocate( (double) 0, rows);
					}
					break;
				case Types.DATE: {
					Date time = set.getDate(column);
					if( time == null ) {
						time = new Date(0);
					}
					array = DataArray.allocate( time.getTime(), rows);
					break;
				}
				case Types.TIME: {
					Time time = set.getTime(column);
					if( time == null ) {
						time = new Time(0);
					}
					array = DataArray.allocate( time.getTime(), rows);
					break;
				}
				case Types.TIMESTAMP: {
					Timestamp time = set.getTimestamp(column);
					if( time == null ) {
						time = new Timestamp(0);
					}
					array = DataArray.allocate( time.getTime(), rows);
					break;
				}
				case Types.BLOB: {
					Blob blob = set.getBlob(column);
					array = DataArray.allocate( blob.getBinaryStream(), rows);
					break;
				}
				case Types.NCLOB:
				case Types.CLOB: {
					array = DataArray.allocate( new String(), rows);
					break;
				}
				case Types.DISTINCT:
				case Types.JAVA_OBJECT:
				case Types.NULL:
				case Types.OTHER:
				case Types.REF:
				case Types.ROWID:
				case Types.SQLXML:
				case Types.STRUCT:
				default: {
					Object value = set.getObject(column);
					if( value == null ) {
						value = new Object();
					}
					array = DataArray.allocate(value, rows);
					break;
				}
				
			}
		} else {
			throw new SQLException("Unable to init array: out of range column index!");
		}
		return array;
	}
	
	/**
	 * Append data to a DataArray object according the type of the ResultSet at the
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
	public void append(DataArray<?> array, ResultSet set, int column, int row, int type) throws SQLException {
		switch (type) {
			case Types.ARRAY:
				array.setData(set.getArray(column).getArray(), row);
				break;
			case Types.DATALINK:
			case Types.CHAR:
			case Types.VARCHAR:
			case Types.LONGVARCHAR:
			case Types.NCHAR:
			case Types.LONGNVARCHAR:
			case Types.NVARCHAR:
				array.setData(set.getString(column), row);
				break;
			case Types.BINARY:
			case Types.VARBINARY:
			case Types.LONGVARBINARY:
				array.setData(set.getBytes(column), row);
				break;
			case Types.BIT:
			case Types.BOOLEAN:
				array.setData(set.getBoolean(column), row);
				break;
			case Types.TINYINT:
			case Types.SMALLINT:
				array.setData(set.getShort(column), row);
				break;
			case Types.INTEGER:
				array.setData(set.getInt(column), row);
				break;
			case Types.BIGINT:
				array.setData(set.getLong(column), row);
				break;
			case Types.DOUBLE:
				array.setData(set.getDouble(column), row);
				break;
			case Types.FLOAT:
			case Types.REAL:
				array.setData(set.getFloat(column), row);
				break;
			case Types.DECIMAL:
			case Types.NUMERIC:
				if ( array.isIntegerNumber()) {
					array.setData( set.getInt(column), row );
				} else {
					array.setData( set.getDouble(column), row);
				}
				break;
			case Types.DATE: {
				Date time = set.getDate(column);
				array.setData( time.getTime(), row);
				break;
			}
			case Types.TIME: {
				Time time = set.getTime(column);
				array.setData( time.getTime(), row);
				break;
			}
			case Types.TIMESTAMP: {
				Timestamp time = set.getTimestamp(column);
				array.setData( time.getTime(), row);
				break;
			}
			case Types.BLOB: {
				Blob blob = set.getBlob(column);
				if( blob != null ) {
					// Allocate a byte buffer
					int length = new Long( blob.length() ).intValue();
					ByteBuffer buffer = ByteBuffer.allocate(length);
					
					try {
						blob.getBinaryStream().read(buffer.array());
					} catch (IOException e) {
						throw new SQLException("Unable to read clob from database!", e);
					}
					// Get the blob's values
					buffer.position(0);
					array.setData( buffer.array(), row );
				}
				break;
			}
			case Types.NCLOB:
			case Types.CLOB: {
				// Get the clob
				Clob clob = set.getClob(column);
				
				if( clob != null ) {
					// Allocate a char buffer
					int length = new Long( clob.length() ).intValue();
					CharBuffer buffer = CharBuffer.allocate( length );
					try {
						clob.getCharacterStream().read(buffer);
					} catch (IOException e) {
						throw new SQLException("Unable to read clob from database!", e);
					}
					// Get the clob's values
					buffer.position(0);
					String values = buffer.toString();	
					array.setData( values, row );
				}
				break;
			}
			case Types.DISTINCT:
			case Types.JAVA_OBJECT:
			case Types.NULL:
			case Types.OTHER:
			case Types.REF:
			case Types.ROWID:
			case Types.SQLXML:
			case Types.STRUCT:
				array.setData(set.getObject(column), row);
				break;
			default: {
				Object value = set.getObject(column);
				array.setData(value, row);
				break;
			}
		}
		array.unlock();
	}
}
