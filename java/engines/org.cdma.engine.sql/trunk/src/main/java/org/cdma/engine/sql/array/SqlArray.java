package org.cdma.engine.sql.array;

import java.math.BigDecimal;
import java.sql.Blob;
import java.sql.Clob;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Types;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.sql.utils.SqlArrayMath;
import org.cdma.engine.sql.utils.SqlArrayUtils;
import org.cdma.exception.BackupException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.ISliceIterator;
import org.cdma.math.IArrayMath;
import org.cdma.utilities.memory.ArrayTools;
import org.cdma.utilities.memory.DefaultIndex;
import org.cdma.utilities.performance.Benchmarker;
import org.cdma.utils.IArrayUtils;

public class SqlArray implements IArray {


	public class DataArray<T> {
        private T mValue;
        private Class<T> mType;

        @SuppressWarnings("unchecked")
        protected DataArray(T tValue) {
            mValue = tValue;
            mType = (Class<T>) tValue.getClass();
        }

        public T getValue() {
            return mValue;
        }

        public Class<T> getType() {
            return mType;
        }
    }

	private IIndex   mIndexData;
	private DataArray<?>  mData;
	private Class<?> mClazz;
	private String   mFactory;
	private int     mNbRows;
	private int     mCurRow;
	private int     mColumn;
	
	public SqlArray(String factory, ResultSet set, int column ) {
		this( factory, set, column, -1);
	}

	public SqlArray(String factory, ResultSet set, int column, int nbRows ) {
		mFactory   = factory;
		mClazz     = null;
		mIndexData = null;
		mData      = null;
		mCurRow    = 0;
		mNbRows    = nbRows;
		mColumn    = column;
		try {
			appendData(set);
			initElementType();
			initIndex();
		} catch (SQLException e) {
			Factory.getLogger().log(Level.SEVERE, "Unable to initialize array!", e);
		}
	}

	private void initIndex() {
		int[] shape = ArrayTools.detectShape( mData.getValue() );
		mIndexData = new DefaultIndex(mFactory, shape);
	}
	
    /*
	@SuppressWarnings({ "unchecked", "rawtypes" })
	private void initData(ResultSet set) throws SQLException {
		ResultSetMetaData meta = set.getMetaData();
		int nb_column = meta.getColumnCount();
		if (mColumn <= nb_column) {
			int type = meta.getColumnType(mColumn);
			switch (type) {
				case Types.ARRAY:
					setData(set.getArray(mColumn).getArray());
					initIndex();
					break;
				case Types.DATALINK:
				case Types.CHAR:
				case Types.VARCHAR:
				case Types.LONGVARCHAR:
				case Types.NCHAR:
				case Types.LONGNVARCHAR:
				case Types.NVARCHAR:
					setData(new String[] { set.getString(mColumn) });
					break;
				case Types.BINARY:
				case Types.VARBINARY:
				case Types.LONGVARBINARY:
					setData(set.getBytes(mColumn));
					break;
				case Types.BIT:
				case Types.BOOLEAN:
					setData(new boolean[] { set.getBoolean(mColumn) });
					break;
				case Types.TINYINT:
				case Types.SMALLINT:
					setData(new short[] { set.getShort(mColumn) });
					break;
				case Types.INTEGER:
					setData(new int[] { set.getInt(mColumn) });
					break;
				case Types.BIGINT:
					setData(new long[] { set.getLong(mColumn) });
					break;
				case Types.DOUBLE:
				case Types.FLOAT:
					setData(new double[] { set.getDouble(mColumn) });
					break;
				case Types.REAL:
					setData(new float[] { set.getFloat(mColumn) });
					break;
				case Types.DECIMAL:
				case Types.NUMERIC:
					BigDecimal decimal = set.getBigDecimal(mColumn);
					
					if (decimal != null ) {
						if( decimal.scale() == 0 ) {
							setData(new int[] { decimal.intValue() });
						}
						else {
							setData(new double[] { decimal.doubleValue() });
						}
					}
					else {
						mData = null;
					}
					break;
				case Types.DATE:
					setData(new Date[]{ set.getDate(mColumn) });
					break;
				case Types.TIME:
					setData(new Time[]{ set.getTime(mColumn) });
					break;
				case Types.TIMESTAMP:
					setData(new Timestamp[]{ set.getTimestamp(mColumn) });
					break;
				case Types.BLOB: {
					Blob blob = set.getBlob(mColumn);
					setData(new InputStream[]{ blob.getBinaryStream() });
					break;
				}
				case Types.NCLOB:
				case Types.CLOB: {
					Clob clob = set.getClob(mColumn);
					setData(new Reader[] { clob.getCharacterStream() });
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
					setData(new Object[] { set.getObject(mColumn) });
					break;
				default: {
					Object value = set.getObject(mColumn);
					Object array = java.lang.reflect.Array.newInstance( value.getClass(), 1);
					java.lang.reflect.Array.set( array, 0, value );
					setData(array);
					break;
				}
			}
		} else {
			throw new SQLException(
					"Unable to init array: out of range column index!");
		}
	}
	*/
	private void initElementType() {
		if( mData != null ) {
			Class<?> clazz = mData.getType();
			if( clazz.isArray() ) {
				clazz = clazz.getComponentType();
			}
			mClazz = clazz;
		}
	}
	
	@Override
	public String getFactoryName() {
		return mFactory;
	}

	@Override
	public IArray copy() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArray copy(boolean data) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArrayUtils getArrayUtils() {
		return new SqlArrayUtils(this);			
	}

	@Override
	public IArrayMath getArrayMath() {
		return new SqlArrayMath(this);
	}

	@Override
	public boolean getBoolean(IIndex ima) {
		boolean result;
		if( Boolean.TYPE.equals( mClazz ) ) {
			int[] pos = ima.getCurrentCounter();
			Object array = getMostVaryingRaw( ima, (boolean[]) mData.getValue() );
			result = ((boolean[]) array)[pos.length - 1];
		}
		else {
			result = (Boolean) getObject(ima);
		}
		return result;
	}

	@Override
	public byte getByte(IIndex ima) {
		byte result;
		if( Byte.TYPE.equals( mClazz ) ) {
			int[] pos = ima.getCurrentCounter();
			Object array = getMostVaryingRaw( ima, (byte[]) mData.getValue() );
			result = ((byte[]) array)[pos.length - 1];
		}
		else {
			result = (Byte) getObject(ima);
		}
		return result;
	}

	@Override
	public char getChar(IIndex ima) {
		char result;
		if( Character.TYPE.equals( mClazz ) ) {
			int[] pos = ima.getCurrentCounter();
			Object array = getMostVaryingRaw( ima, (char[]) mData.getValue() );
			result = ((char[]) array)[pos.length - 1];
		}
		else {
			result = (Character) getObject(ima);
		}
		return result;
	}

	@Override
	public double getDouble(IIndex ima) {
		double result;
		if( Double.TYPE.equals( mClazz ) ) {
			int[] pos = ima.getCurrentCounter();
			Object array = getMostVaryingRaw( ima, (double[]) mData.getValue() );
			result = ((double[]) array)[pos.length - 1];
		}
		else {
			Number tmp = (Number) getObject(ima);
			result = tmp.doubleValue();
		}
		return result;
	}

	@Override
	public Class<?> getElementType() {
		if( mClazz == null ) {
			initElementType();
		}
		return mClazz;
	}

	@Override
	public float getFloat(IIndex ima) {
		float result;
		if( Double.TYPE.equals( mClazz ) ) {
			int[] pos = ima.getCurrentCounter();
			Object array = getMostVaryingRaw( ima, (float[]) mData.getValue() );
			result = ((float[]) array)[pos.length - 1];
		}
		else {
			Number tmp = (Number) getObject(ima);
			result = tmp.floatValue();
		}
		return result;
	}

	@Override
	public IIndex getIndex() {
		return mIndexData;
	}

	@Override
	public int getInt(IIndex ima) {
		int result;
		if( Integer.TYPE.equals( mClazz ) ) {
			int[] pos = ima.getCurrentCounter();
			Object array = getMostVaryingRaw( ima, (int[]) mData.getValue() );
			result = ((int[]) array)[pos.length - 1];
		}
		else {
			Number tmp = (Number) getObject(ima);
			result = tmp.intValue();
		}
		return result;
	}

	@Override
	public IArrayIterator getIterator() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public long getLong(IIndex ima) {
		long result;
		if( Long.TYPE.equals( mClazz ) ) {
			int[] pos = ima.getCurrentCounter();
			Object array = getMostVaryingRaw( ima, (long[]) mData.getValue() );
			result = ((long[]) array)[pos.length - 1];
		}
		else {
			Number tmp = (Number) getObject(ima);
			result = tmp.longValue();
		}
		return result;
	}

	@Override
	public Object getObject(IIndex index) {
		Object result = null;
		
		if( index.getRank() == mIndexData.getRank() ) {
			result = mData.getValue();
			int[] counter = index.getCurrentCounter();
			for( int position : counter ) {
				result = java.lang.reflect.Array.get( result, position );
			}
		}
		
		return result;
		
		/*
		int lPos = (int) index.currentElement();
		Object result = java.lang.reflect.Array.get(mData.getValue(), lPos);
		return result;
		*/
	}

	@Override
	public int getRank() {
		return mIndexData.getRank();
	}

	@Override
	public IArrayIterator getRegionIterator(int[] reference, int[] range) throws InvalidRangeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int[] getShape() {
		return mIndexData.getShape();
	}

	@Override
	public short getShort(IIndex ima) {
		short result;
		if( Short.TYPE.equals( mClazz ) ) {
			int[] pos = ima.getCurrentCounter();
			Object array = getMostVaryingRaw( ima, (short[]) mData.getValue() );
			result = ((short[]) array)[pos.length - 1];
		}
		else {
			Number tmp = (Number) getObject(ima);
			result = tmp.shortValue();
		}
		return result;
	}

	@Override
	public long getSize() {
		return mIndexData.getSize();
	}

	@Override
	public Object getStorage() {
		return mData.getValue();
	}

	@Override
	public void setBoolean(IIndex ima, boolean value) {
		throw new NotImplementedException();
	}

	@Override
	public void setByte(IIndex ima, byte value) {
		throw new NotImplementedException();
	}

	@Override
	public void setChar(IIndex ima, char value) {
		throw new NotImplementedException();
	}

	@Override
	public void setDouble(IIndex ima, double value) {
		throw new NotImplementedException();
	}

	@Override
	public void setFloat(IIndex ima, float value) {
		throw new NotImplementedException();
	}

	@Override
	public void setInt(IIndex ima, int value) {
		throw new NotImplementedException();
	}

	@Override
	public void setLong(IIndex ima, long value) {
		throw new NotImplementedException();
	}

	@Override
	public void setObject(IIndex ima, Object value) {
		throw new NotImplementedException();
	}

	@Override
	public void setShort(IIndex ima, short value) {
		throw new NotImplementedException();
	}

	@Override
	public String shapeToString() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setIndex(IIndex index) {
		mIndexData = index;
	}

	@Override
	public ISliceIterator getSliceIterator(int rank) throws ShapeNotMatchException, InvalidRangeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void releaseStorage() throws BackupException {
		// TODO Auto-generated method stub

	}

	@Override
	public long getRegisterId() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void lock() {
		// TODO Auto-generated method stub

	}

	@Override
	public void unlock() {
		// TODO Auto-generated method stub
	}

	@Override
	public boolean isDirty() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void setDirty(boolean dirty) {
		// TODO Auto-generated method stub
	}

	@Override
	public IArray setDouble(double value) {
		throw new NotImplementedException();
	}
	
	@SuppressWarnings({ "unchecked" })
    private <T> void setData( T data ) {
		if( mData == null ) {
	    	DataArray<?> tmp;
	    	if( mNbRows > 0 ) {
				T[] array = (T[]) java.lang.reflect.Array.newInstance(data.getClass(), mNbRows);
	    		array[0] = data;
	    		tmp = new DataArray<T[]>( array );
	    	}
	    	else {
	    		tmp = new DataArray<T>( data );
	    	}
	    	mData = tmp;
		}
		else {
			if( mNbRows > 0 && mCurRow < mNbRows ) {
				((DataArray<T[]>) mData).getValue()[mCurRow] = data;
			}
		}
    }
	
	@SuppressWarnings("unchecked")
	private void setData(int data) {
		if( mData == null ) {
			DataArray<int[]> array;
	    	if( mNbRows > 0 ) {
	    		array = new DataArray<int[]>( new int[mNbRows] );
	    	}
	    	else {
	    		array = new DataArray<int[]>( new int[1] );
	    	}
	    	mData = array;
	    	array.getValue()[mCurRow] = data;
		}
		else {
			if( mNbRows > 0 && mCurRow < mNbRows ) {
				((DataArray<int[]>) mData).getValue()[mCurRow] = data;
			}
		}
	}
	
	@SuppressWarnings("unchecked")
	private void setData(long data) {
		if( mData == null ) {
			DataArray<long[]> array;
	    	if( mNbRows > 0 ) {
	    		array = new DataArray<long[]>( new long[mNbRows] );
	    	}
	    	else {
	    		array = new DataArray<long[]>( new long[1] );
	    	}
	    	mData = array;
	    	array.getValue()[mCurRow] = data;
		}
		else {
			if( mNbRows > 0 && mCurRow < mNbRows ) {
				((DataArray<long[]>) mData).getValue()[mCurRow] = data;
			}
		}
	}
	
	@SuppressWarnings("unchecked")
	private void setData(double data) {
		if( mData == null ) {
			DataArray<double[]> array;
	    	if( mNbRows > 0 ) {
	    		array = new DataArray<double[]>( new double[mNbRows] );
	    	}
	    	else {
	    		array = new DataArray<double[]>( new double[1] );
	    	}
	    	mData = array;
	    	array.getValue()[mCurRow] = data;
		}
		else {
			if( mNbRows > 0 && mCurRow < mNbRows ) {
				((DataArray<double[]>) mData).getValue()[mCurRow] = data;
			}
		}
	}
	
	@SuppressWarnings("unchecked")
	private void setData(short data) {
		if( mData == null ) {
			DataArray<short[]> array;
	    	if( mNbRows > 0 ) {
	    		array = new DataArray<short[]>( new short[mNbRows] );
	    	}
	    	else {
	    		array = new DataArray<short[]>( new short[1] );
	    	}
	    	mData = array;
	    	array.getValue()[mCurRow] = data;
		}
		else {
			if( mNbRows > 0 && mCurRow < mNbRows ) {
				((DataArray<short[]>) mData).getValue()[mCurRow] = data;
			}
		}
	}
	
	@SuppressWarnings("unchecked")
	private void setData(float data) {
		if( mData == null ) {
			DataArray<float[]> array;
	    	if( mNbRows > 0 ) {
	    		array = new DataArray<float[]>( new float[mNbRows] );
	    	}
	    	else {
	    		array = new DataArray<float[]>( new float[1] );
	    	}
	    	mData = array;
	    	array.getValue()[mCurRow] = data;
		}
		else {
			if( mNbRows > 0 && mCurRow < mNbRows ) {
				((DataArray<float[]>) mData).getValue()[mCurRow] = data;
			}
		}
	}
	
	@SuppressWarnings("unchecked")
	private void setData(boolean data) {
		if( mData == null ) {
			DataArray<boolean[]> array;
	    	if( mNbRows > 0 ) {
	    		array = new DataArray<boolean[]>( new boolean[mNbRows] );
	    	}
	    	else {
	    		array = new DataArray<boolean[]>( new boolean[1] );
	    	}
	    	mData = array;
	    	array.getValue()[mCurRow] = data;
		}
		else {
			if( mNbRows > 0 && mCurRow < mNbRows ) {
				((DataArray<boolean[]>) mData).getValue()[mCurRow] = data;
			}
		}
	}
	
	public void appendData(ResultSet set) throws SQLException {
		Benchmarker.start("appendData");
		ResultSetMetaData meta = set.getMetaData();
		int nb_column = meta.getColumnCount();
		if (mColumn <= nb_column) {
			int type = meta.getColumnType(mColumn);
			switch (type) {
				case Types.ARRAY:
					/*
					Object data;
					if( mNbRows > 0 ) {
						data = new Object[mNbRows];
						((Object[]) data)[0] = set.getArray(column).getArray();
					}
					else {
						data = set.getArray(column).getArray();
					}
					mData = new DataArray( data );
					*/
					setData(set.getArray(mColumn).getArray());
					initIndex();
					break;
				case Types.DATALINK:
				case Types.CHAR:
				case Types.VARCHAR:
				case Types.LONGVARCHAR:
				case Types.NCHAR:
				case Types.LONGNVARCHAR:
				case Types.NVARCHAR:
					//mData = new DataArray( new String[] { set.getString(column) } );
					setData(set.getString(mColumn));
					break;
				case Types.BINARY:
				case Types.VARBINARY:
				case Types.LONGVARBINARY:
					//mData = new DataArray( set.getBytes(column) );
					setData(set.getBytes(mColumn));
					break;
				case Types.BIT:
				case Types.BOOLEAN:
					//mData = new DataArray( new boolean[] { set.getBoolean(column) } );
					setData(set.getBoolean(mColumn));
					break;
				case Types.TINYINT:
				case Types.SMALLINT:
					//mData = new DataArray( new short[] { set.getShort(column) } );
					setData(set.getShort(mColumn));
					break;
				case Types.INTEGER:
					//mData = new DataArray( new int[] { set.getInt(column) } );
					setData(set.getInt(mColumn));
					break;
				case Types.BIGINT:
					//mData = new DataArray( new long[] { set.getLong(column) } );
					setData(set.getLong(mColumn));
					break;
				case Types.DOUBLE:
				case Types.FLOAT:
					//mData = new DataArray( new double[] { set.getDouble(column) } );
					setData(set.getDouble(mColumn));
					break;
				case Types.REAL:
					//mData = new DataArray( new float[] { set.getFloat(column) } );
					setData(set.getFloat(mColumn));
					break;
				case Types.DECIMAL:
				case Types.NUMERIC:
					BigDecimal decimal = set.getBigDecimal(mColumn);
					
					if (decimal != null ) {
						if( decimal.scale() == 0 ) {
							//mData = new DataArray( new int[] { decimal.intValue() } );
							setData(decimal.intValue());
						}
						else {
							//mData = new DataArray( new double[] { decimal.doubleValue() } );
							setData(decimal.doubleValue());
						}
					}
					else {
						mData = null;
					}
					break;
				case Types.DATE:
					//mData = new DataArray( new Date[]{ set.getDate(column) } );
					setData(set.getDate(mColumn));
					break;
				case Types.TIME:
					//mData = new DataArray( new Time[]{ set.getTime(column) } );
					setData(set.getTime(mColumn));
					break;
				case Types.TIMESTAMP:
					//mData = new DataArray( new Timestamp[]{ set.getTimestamp(column) } );
					setData(set.getTimestamp(mColumn));
					break;
				case Types.BLOB: {
					Blob blob = set.getBlob(mColumn);
					//mData = new DataArray( new InputStream[]{ blob.getBinaryStream() } );
					setData(blob.getBinaryStream());
					break;
				}
				case Types.NCLOB:
				case Types.CLOB: {
					Clob clob = set.getClob(mColumn);
					//mData = new DataArray( new Reader[] { clob.getCharacterStream() } );
					setData(clob.getCharacterStream());
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
					//mData = new DataArray( new Object[] { set.getObject(column) } );
					setData(set.getObject(mColumn));
					break;
				default: {
					Object value = set.getObject(mColumn);
					/*Object array = java.lang.reflect.Array.newInstance( value.getClass(), 1);
					java.lang.reflect.Array.set( array, 0, value );
					//mData = new DataArray( array );
					*/setData(value);
					break;
				}
			}
			mCurRow++;
		} else {
			throw new SQLException("Unable to init array: out of range column index!");
		}
		Benchmarker.stop("appendData");
	}

	/**
	 * Return the most varying raw of the given data, positioned at the given coordinates.
	 * @param coordinate position to get slab data from data
	 * @param data of type T to be sliced
	 * @return an Object that is an array of type T
	 */
	private <T> Object getMostVaryingRaw( IIndex index, T data ) {
		int rank = index.getRank();
		int[] shape = java.util.Arrays.copyOf( index.getCurrentCounter(), rank );
		return getMostVaryingRaw( shape, data, 0 );
	}
	
	private <T, C> Object getMostVaryingRaw( int[] coordinates, T data, int depth ) {
		Object result = data;
		
		if( coordinates.length - 1 > depth ) {
			result = getMostVaryingRaw( coordinates, ((C[]) data)[depth], depth++ );
		}
		else {
			result = data;
		}
		return result;
	}
}
