//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
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


/**
 * Parameterized class that aims to ease the memory allocation for a SQL result
 * set. It allows to have a type erasure pattern and to deal with primitive
 * types.
 * 
 * @author rodriguez
 */
public class DataArray<T> {
	private T mValue;       // Data storage
	private Class<T> mType; // Data type
	private boolean mIsIntegerNumber;

	@SuppressWarnings("unchecked")
	public DataArray(T value) {
		mValue = value;
		mType = (Class<T>) value.getClass();

		// Try to determine if the type can be associated to an integer 
		Class<?> clazz = mType;
		while( clazz.isArray() ) {
			clazz = clazz.getComponentType();
		}
		mIsIntegerNumber = (clazz.equals(Integer.TYPE) || clazz.equals(Long.TYPE) || clazz.equals(Short.TYPE));
	}
	
	

	/**
	 * Getter on the memory storage
	 * 
	 * @return T object (array of primitive or an object instance)
	 */
	public T getValue() {
		return mValue;
	}

	/**
	 * The element type of the underlying memory storage
	 * 
	 * @return Class<T>
	 */
	public Class<T> getType() {
		return mType;
	}

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
	static public DataArray<?> allocate(ResultSet set, int column, int nbRows)
			throws SQLException {
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
					array = DataArray.allocate(blob.getBinaryStream(), rows);
					break;
				}
				case Types.NCLOB:
				case Types.CLOB: {
					array = DataArray.allocate(new String(), rows);
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
			throw new SQLException(
					"Unable to init array: out of range column index!");
		}
		return array;
	}
	
	public void append(ResultSet set, int column, int row) throws SQLException {
		ResultSetMetaData meta = set.getMetaData();
		int type = meta.getColumnType(column);
		append(set, column, row, type);
	}

	public void append(ResultSet set, int column, int row, int type) throws SQLException {
		switch (type) {
			case Types.ARRAY:
				setData(set.getArray(column).getArray(), row);
				break;
			case Types.DATALINK:
			case Types.CHAR:
			case Types.VARCHAR:
			case Types.LONGVARCHAR:
			case Types.NCHAR:
			case Types.LONGNVARCHAR:
			case Types.NVARCHAR:
				setData(set.getString(column), row);
				break;
			case Types.BINARY:
			case Types.VARBINARY:
			case Types.LONGVARBINARY:
				setData(set.getBytes(column), row);
				break;
			case Types.BIT:
			case Types.BOOLEAN:
				setData(set.getBoolean(column), row);
				break;
			case Types.TINYINT:
			case Types.SMALLINT:
				setData(set.getShort(column), row);
				break;
			case Types.INTEGER:
				setData(set.getInt(column), row);
				break;
			case Types.BIGINT:
				setData(set.getLong(column), row);
				break;
			case Types.DOUBLE:
				setData(set.getDouble(column), row);
				break;
			case Types.FLOAT:
			case Types.REAL:
				setData(set.getFloat(column), row);
				break;
			case Types.DECIMAL:
			case Types.NUMERIC:
				if ( isIntegerNumber()) {
					setData( set.getInt(column), row );
				} else {
					setData( set.getDouble(column), row);
				}
				break;
			case Types.DATE: {
				Date time = set.getDate(column);
				setData( time.getTime(), row);
				break;
			}
			case Types.TIME: {
				Time time = set.getTime(column);
				setData( time.getTime(), row);
				break;
			}
			case Types.TIMESTAMP: {
				Timestamp time = set.getTimestamp(column);
				setData( time.getTime(), row);
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
					setData( buffer.array(), row );
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
					setData( values, row );
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
				setData(set.getObject(column), row);
				break;
			default: {
				Object value = set.getObject(column);
				setData(value, row);
				break;
			}
		}
	}

	// ---------------------------------------------------------
	// Private static methods
	// ---------------------------------------------------------
	@SuppressWarnings({ "unchecked" })
	static private <T> DataArray<?> allocate(T data, int nbRows) {
		DataArray<?> array;
		if( data == null ) {
			if (nbRows > 0) {
				Object[] storage = new Object[nbRows];
				array = new DataArray<Object[]>(storage);
			} else {
				array = new DataArray<Object>(data);
			}
		}
		else {
			if (nbRows > 0) {
				T[] storage = (T[]) java.lang.reflect.Array.newInstance(data.getClass(), nbRows);
				array = new DataArray<T[]>(storage);
			} else {
				array = new DataArray<T>(data);
			}
		}
		return array;
	}

	static private <T> DataArray<?> allocate(int data, int nbRows) {
		DataArray<int[]> array;
		if (nbRows > 0) {
			array = new DataArray<int[]>(new int[nbRows]);
		} else {
			array = new DataArray<int[]>(new int[1]);
		}
		return array;
	}

	static private <T> DataArray<?> allocate(long data, int nbRows) {
		DataArray<long[]> array;
		if (nbRows > 0) {
			array = new DataArray<long[]>(new long[nbRows]);
		} else {
			array = new DataArray<long[]>(new long[1]);
		}
		return array;
	}

	static private <T> DataArray<?> allocate(double data, int nbRows) {
		DataArray<double[]> array;
		if (nbRows > 0) {
			array = new DataArray<double[]>(new double[nbRows]);
		} else {
			array = new DataArray<double[]>(new double[1]);
		}
		return array;
	}

	static private <T> DataArray<?> allocate(short data, int nbRows) {
		DataArray<short[]> array;
		if (nbRows > 0) {
			array = new DataArray<short[]>(new short[nbRows]);
		} else {
			array = new DataArray<short[]>(new short[1]);
		}
		return array;
	}

	static private <T> DataArray<?> allocate(float data, int nbRows) {
		DataArray<float[]> array;
		if (nbRows > 0) {
			array = new DataArray<float[]>(new float[nbRows]);
		} else {
			array = new DataArray<float[]>(new float[1]);
		}
		return array;
	}

	static private <T> DataArray<?> allocate(boolean data, int nbRows) {
		DataArray<boolean[]> array;
		if (nbRows > 0) {
			array = new DataArray<boolean[]>(new boolean[nbRows]);
		} else {
			array = new DataArray<boolean[]>(new boolean[1]);
		}
		return array;
	}

	// ---------------------------------------------------------
	// Private methods
	// ---------------------------------------------------------
	@SuppressWarnings("unchecked")
	private <U> void setData(U data, int row) {
		if (mValue != null) {
			((U[]) mValue)[row] = data;
		}
	}

	private void setData(int data, int row) {
		if (mValue != null) {
			((int[]) mValue)[row] = data;
		}
	}

	private void setData(long data, int row) {
		if (mValue != null) {
			((long[]) mValue)[row] = data;
		}
	}

	private void setData(double data, int row) {
		if (mValue != null) {
			((double[]) mValue)[row] = data;
		}
	}

	private void setData(short data, int row) {
		if (mValue != null) {
			((short[]) mValue)[row] = data;
		}
	}

	private void setData(float data, int row) {
		if (mValue != null) {
			((float[]) mValue)[row] = data;
		}
	}

	private void setData(boolean data, int row) {
		if (mValue != null) {
			((boolean[]) mValue)[row] = data;
		}
	}

	private boolean isIntegerNumber() {
		return mIsIntegerNumber;
	}
}
