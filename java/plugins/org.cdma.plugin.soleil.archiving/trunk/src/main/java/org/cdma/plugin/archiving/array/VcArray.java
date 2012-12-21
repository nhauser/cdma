package org.cdma.plugin.archiving.array;

import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.exception.BackupException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.ISliceIterator;
import org.cdma.math.IArrayMath;
import org.cdma.plugin.archiving.VcFactory;
import org.cdma.plugin.xml.util.XmlArrayMath;
import org.cdma.utilities.memory.DefaultIndex;
import org.cdma.utils.IArrayUtils;

public class VcArray implements IArray {

	public static final int TIME_INDEX = 0;
	public static final int VALUE_INDEX = 1;

	private Object mStorage;
	private DefaultIndex mView;
	private boolean mIsRawArray;

	public VcArray(Object storage, int[] shape) {
		mView = new DefaultIndex(VcFactory.NAME, shape);
		mStorage = storage;
		if (shape.length == 1) {
			mIsRawArray = true;
		} else {
			mIsRawArray = (storage.getClass().isArray() && !(java.lang.reflect.Array
					.get(storage, 0).getClass().isArray()));
		}
	}

	protected VcArray(Object storage, final IIndex index) {
		mStorage = storage;
		mView = new DefaultIndex( getFactoryName(), index );
	}

	@Override
	public String getFactoryName() {
		return VcFactory.NAME;
	}

	@Override
	public IArray copy() {
		return new VcArray(mStorage, mView);
	}

	@Override
	public IArray copy(boolean data) {
		VcArray result = (VcArray) copy();
		if (data) {
			result.mStorage = copyJavaArray(mStorage);
		}
		return result;
	}

	@Override
	public IArrayUtils getArrayUtils() {
		return new VcArrayUtils(this);
	}

	@Override
	public IArrayMath getArrayMath() {
		return new XmlArrayMath(this);
	} 

	public boolean isScalar() {
		return !(mStorage.getClass().isArray());
	}

	@Override
	public boolean getBoolean(IIndex ima) {
		boolean result;
		IIndex idx;
		try {
			idx = ima.clone();
			Object oData = mStorage;

			// If it's a scalar value then we return it
			if (!oData.getClass().isArray()) {
				result = (Boolean) oData;
			}
			// else it's a single raw array, then we compute indexes to have the
			// corresponding cell number
			else {
				int lPos = (int) idx.currentElement();
				result = ((boolean[]) oData)[lPos];
			}
		} catch (CloneNotSupportedException e) {
			result = false;
		} catch (ClassCastException e) {
			result = false;
		}
		return result;
	}

	@Override
	public byte getByte(IIndex ima) {
		byte result;
		IIndex idx;
		try {
			idx = ima.clone();
			Object oData = mStorage;

			// If it's a scalar value then we return it
			if (!oData.getClass().isArray()) {
				result = (Byte) oData;
			}
			// else it's a single raw array, then we compute indexes to have the
			// corresponding cell number
			else {
				int lPos = (int) idx.currentElement();
				result = ((byte[]) oData)[lPos];
			}
		} catch (CloneNotSupportedException e) {
			result = -1;
		} catch (ClassCastException e) {
			result = -1;
		}
		return result;
	}

	@Override
	public char getChar(IIndex ima) {
		char result;
		IIndex idx;
		try {
			idx = ima.clone();
			Object oData = mStorage;

			// If it's a scalar value then we return it
			if (!oData.getClass().isArray()) {
				result = (Character) oData;
			}
			// else it's a single raw array, then we compute indexes to have the
			// corresponding cell number
			else {
				int lPos = (int) idx.currentElement();
				result = ((char[]) oData)[lPos];
			}
		} catch (CloneNotSupportedException e) {
			result = 0;
		} catch (ClassCastException e) {
			result = 0;
		}
		return result;
	}

	@Override
	public double getDouble(IIndex ima) {
		double result;
		IIndex idx;
		try {
			idx = ima.clone();
			Object oData = mStorage;

			// If it's a scalar value then we return it
			if (!oData.getClass().isArray()) {
				result = (Double) oData;
			}
			// else it's a single raw array, then we compute indexes to have the
			// corresponding cell number
			else {
				int lPos = (int) idx.currentElement();
				result = ((double[]) oData)[lPos];
			}
		} catch (CloneNotSupportedException e) {
			result = 0;
		} catch (ClassCastException e) {
			result = 0;
		}
		return result;
	}

	@Override
	public float getFloat(IIndex ima) {
		float result;
		IIndex idx;
		try {
			idx = ima.clone();
			Object oData = mStorage;

			// If it's a scalar value then we return it
			if (!oData.getClass().isArray()) {
				result = (Float) oData;
			}
			// else it's a single raw array, then we compute indexes to have the
			// corresponding cell number
			else {
				int lPos = (int) idx.currentElement();
				result = ((float[]) oData)[lPos];
			}
		} catch (CloneNotSupportedException e) {
			result = 0;
		} catch (ClassCastException e) {
			result = 0;
		}
		return result;
	}

	@Override
	public int getInt(IIndex ima) {
		int result;
		IIndex idx;
		try {
			idx = ima.clone();
			Object oData = mStorage;

			// If it's a scalar value then we return it
			if (!oData.getClass().isArray()) {
				result = (Integer) oData;
			}
			// else it's a single raw array, then we compute indexes to have the
			// corresponding cell number
			else {
				int lPos = (int) idx.currentElement();
				result = ((int[]) oData)[lPos];
			}
		} catch (CloneNotSupportedException e) {
			result = 0;
		} catch (ClassCastException e) {
			result = 0;
		}
		return result;
	}

	@Override
	public long getLong(IIndex ima) {
		long result;
		IIndex idx;
		try {
			idx = ima.clone();
			Object oData = mStorage;

			// If it's a scalar value then we return it
			if (!oData.getClass().isArray()) {
				result = (Long) oData;
			}
			// else it's a single raw array, then we compute indexes to have the
			// corresponding cell number
			else {
				int lPos = (int) idx.currentElement();
				result = ((long[]) oData)[lPos];
			}
		} catch (CloneNotSupportedException e) {
			result = 0;
		} catch (ClassCastException e) {
			result = 0;
		}
		return result;
	}

	@Override
	public Object getObject(IIndex ima) {
		Object result = null;
		Object oData = mStorage;

		// If it's a scalar value then we return it
		if (!oData.getClass().isArray()) {
			result = oData;
		}
		// else it's a single raw array, then we compute indexes to have the
		// corresponding cell number
		else {
			if( mIsRawArray ) {
				int lPos = (int) ima.currentElement();
				result = java.lang.reflect.Array.get(oData, lPos);
			}
			else {
				// The whole storage array of the data is contained in this array.
				// If this array is a sub-part of a bigger one, we should consider
				// the starting position of this
				int[] counter = ima.getCurrentCounter();
				int[] current = mView.getProjectionOrigin();
				if( counter.length < current.length ) {
					int delta = current.length - counter.length;
					for( int i = 0; i < counter.length; i++ ) {
						current[delta + i] += counter[i];
					}
				}
				else {
					current = counter;
				}
				
				for( int position : current ) {
					oData = java.lang.reflect.Array.get(oData, position);
				}
				result = oData;
			}
		}
		return result;
	}

	@Override
	public short getShort(IIndex ima) {
		short result;
		IIndex idx;
		try {
			idx = ima.clone();
			Object oData = mStorage;

			// If it's a scalar value then we return it
			if (!oData.getClass().isArray()) {
				result = (Short) oData;
			}
			// else it's a single raw array, then we compute indexes to have the
			// corresponding cell number
			else {
				int lPos = (int) idx.currentElement();
				result = ((short[]) oData)[lPos];
			}
		} catch (CloneNotSupportedException e) {
			result = 0;
		} catch (ClassCastException e) {
			result = 0;
		}
		return result;
	}

	@Override
	public Class<?> getElementType() {

		Class<?> result = null;
		Object oData = mStorage;
		if (oData != null) {
			if (oData.getClass().isArray()) {
				result = oData.getClass().getComponentType();
				while (result.isArray()) {
					result = result.getComponentType();
				}
			} else {
				result = oData.getClass();
			}
		}
		return result;
	}

	@Override
	public IIndex getIndex() {
		return mView;
	}

	@Override
	public IArrayIterator getIterator() {
		return new VcArrayIterator(this);
	}

	@Override
	public int getRank() {
		return mView.getRank();
	}

	@Override
	public IArrayIterator getRegionIterator(int[] reference, int[] range)
			throws InvalidRangeException {
		DefaultIndex index = new DefaultIndex(VcFactory.NAME, mView.getShape(),
				reference, range);
		return new VcArrayIterator(this, index);
	}

	@Override
	public int[] getShape() {
		return mView.getShape();
	}

	@Override
	public long getSize() {
		return mView.getSize();
	}

	@Override
	public Object getStorage() {
		return mStorage;
	}

	/**
	 * Set the given object into the targeted cell by given index (eventually
	 * using auto-boxing). It's the central data access method that all other
	 * methods rely on.
	 * 
	 * @param index
	 *            targeting a cell
	 * @param value
	 *            new value to set in the array
	 * @throws InvalidRangeException
	 *             if one of the index is bigger than the corresponding
	 *             dimension shape
	 */
	private void set(IIndex index, Object value) {
		// If array has string class: then it's a scalar string
		Object oData = mStorage;
		if (oData.getClass().equals(String.class)
				&& value.getClass().equals(String.class)) {
			mStorage = value;
		}
		// If array isn't an array we set the scalar value
		else if (!oData.getClass().isArray()) {
			mStorage = value;
		}
		// If it's a single raw array, then we compute indexes to have the
		// corresponding cell number
		else if (mIsRawArray) {
			int lPos = translateIndex(index);
			java.lang.reflect.Array.set(oData, lPos, value);
		}
		// Else it's a multidimensional array, so we will take slices from each
		// dimension until we can reach requested cell
		else {
			int[] indexes = null;
			if (index instanceof DefaultIndex) {
				indexes = ((DefaultIndex) index).getCurrentPos();
			} else {
				indexes = index.getCurrentCounter();
			}
			Object oCurObj = oData;
			for (int i = 0; i < indexes.length - 1; i++) {
				oCurObj = java.lang.reflect.Array.get(oCurObj, indexes[i]);
			}
			java.lang.reflect.Array.set(oCurObj, indexes[indexes.length - 1],
					value);
		}
	}

	/**
	 * Translate the given IIndex into the corresponding cell index. This method
	 * is used to access a multidimensional array's cell when the memory storage
	 * is a single raw array.
	 * 
	 * @param index
	 *            sibling a cell in a multidimensional array
	 * @return the cell number in a single raw array (that carry the same
	 *         logical shape)
	 */
	private int translateIndex(IIndex index) {
		int[] indexes = index.getCurrentCounter();
		int[] shape = mView.getShape();

		int lPos = 0, lStartRaw;
		for (int k = 1; k < shape.length; k++) {

			lStartRaw = 1;
			for (int j = 0; j < k; j++) {
				lStartRaw *= shape[j];
			}
			lStartRaw *= indexes[k - 1];
			lPos += lStartRaw;
		}
		lPos += indexes[indexes.length - 1];
		return lPos;
	}

	@Override
	public void setBoolean(IIndex ima, boolean value) {
		set(ima, value);
	}

	@Override
	public void setByte(IIndex ima, byte value) {
		set(ima, value);
	}

	@Override
	public void setChar(IIndex ima, char value) {
		set(ima, value);
	}

	@Override
	public void setDouble(IIndex ima, double value) {
		set(ima, value);
	}

	@Override
	public void setFloat(IIndex ima, float value) {
		set(ima, value);
	}

	@Override
	public void setInt(IIndex ima, int value) {
		set(ima, value);
	}

	@Override
	public void setLong(IIndex ima, long value) {
		set(ima, value);
	}

	@Override
	public void setObject(IIndex ima, Object value) {
		set(ima, value);
	}

	@Override
	public void setShort(IIndex ima, short value) {
		set(ima, value);
	}

	@Override
	public IArray setDouble(double value) {
		Object oData = mStorage;
		if (mIsRawArray) {
			java.util.Arrays.fill((double[]) oData, value);
		} else {
			setDouble(oData, value);
		}
		return this;
	}

	/**
	 * Recursive method that sets all values of the given array (whatever it's
	 * form is) to the same given double value
	 * 
	 * @param array
	 *            object array to fill
	 * @param value
	 *            double value to be set in the array
	 * @return the array filled properly
	 * @note ensure the given array is a double[](...[]) or a Double[](...[])
	 */
	private Object setDouble(Object array, double value) {
		if (array.getClass().isArray()) {
			int iLength = java.lang.reflect.Array.getLength(array);
			for (int j = 0; j < iLength; j++) {
				Object o = java.lang.reflect.Array.get(array, j);
				if (o.getClass().isArray()) {
					setDouble(o, value);
				} else {
					java.util.Arrays.fill((double[]) array, value);
					return array;
				}
			}
		} else {
			java.lang.reflect.Array.set(array, 0, value);
		}

		return array;
	}

	@Override
	public String shapeToString() {
		int[] shape = getShape();
		StringBuilder sb = new StringBuilder();
		if (shape.length != 0) {
			sb.append('(');
			for (int i = 0; i < shape.length; i++) {
				int s = shape[i];
				if (i > 0) {
					sb.append(",");
				}
				sb.append(s);
			}
			sb.append(')');
		}
		return sb.toString();
	}

	@Override
	public void setIndex(IIndex index) {
		if (index != null) {
			if( index instanceof DefaultIndex ) {
				try {
					mView = (DefaultIndex) index.clone();
				} catch (CloneNotSupportedException e) {
					Factory.getLogger().log(Level.SEVERE, "Unable to set index!", e);
				}
			}
			else {
				mView = new DefaultIndex(getFactoryName(), index);
			}
		}
	}

	@Override
	public ISliceIterator getSliceIterator(int rank) throws ShapeNotMatchException, InvalidRangeException {
		return new VcSliceIterator(this, rank);
	}

	// //////////////
	// Static Methods
	// //////////////

	static public Object copyJavaArray(Object array) {
		Object result = array;
		if (result == null) {
			return null;
		} else {
			// Determine rank of array (by parsing data array class name)
			String sClassName = array.getClass().getName();
			int iRank = 0;
			int iIndex = 0;
			char cChar;
			while (iIndex < sClassName.length()) {
				cChar = sClassName.charAt(iIndex);
				iIndex++;
				if (cChar == '[') {
					iRank++;
				}
			}

			// Set dimension rank
			int[] shape = new int[iRank];

			// Fill dimension size array
			for (int i = 0; i < iRank; i++) {
				shape[i] = java.lang.reflect.Array.getLength(result);
				result = java.lang.reflect.Array.get(result, 0);
			}

			// Define a convenient array (shape and type)
			result = java.lang.reflect.Array.newInstance(array.getClass()
					.getComponentType(), shape);
			result = copyJavaArray(array, result);
		}
		return result;
	}

	static public Object copyJavaArray(Object source, Object target) {
		Object item = java.lang.reflect.Array.get(source, 0);
		int length = java.lang.reflect.Array.getLength(source);

		if (item.getClass().isArray()) {
			Object tmpSrc;
			Object tmpTar;
			for (int i = 0; i < length; i++) {
				tmpSrc = java.lang.reflect.Array.get(source, i);
				tmpTar = java.lang.reflect.Array.get(target, i);
				copyJavaArray(tmpSrc, tmpTar);
			}
		} else {
			System.arraycopy(source, 0, target, 0, length);
		}

		return target;
	}

	// //////////////
	// Unused Methods
	// //////////////

	@Override
	public long getRegisterId() {
		return 0;
	}

	@Override
	public void lock() {
	}

	@Override
	public void unlock() {
	}

	@Override
	public void releaseStorage() throws BackupException {
	}

	@Override
	public boolean isDirty() {
		return false;
	}

	@Override
	public void setDirty(boolean dirty) {
	}

}
