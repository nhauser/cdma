package org.cdma.plugin.xml.array;

import org.cdma.exception.BackupException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.ISliceIterator;
import org.cdma.math.IArrayMath;
import org.cdma.plugin.xml.util.XmlArrayMath;
import org.cdma.plugin.xml.util.XmlArrayUtils;
import org.cdma.utils.IArrayUtils;

public class XmlArray implements IArray {

	private String mFactoryName;
	private String mValue;
	private IIndex mIndex;

	public XmlArray(String factory, String value) {
		mValue = value;
	}

	@Override
	public String getFactoryName() {
		return mFactoryName;
	}

	@Override
	public IArray copy() {
		return new XmlArray(mFactoryName, mValue);
	}

	@Override
	public IArray copy(boolean data) {
		return copy();
	}

	@Override
	public IArrayUtils getArrayUtils() {
		return new XmlArrayUtils(this);
	}

	@Override
	public IArrayMath getArrayMath() {
		return new XmlArrayMath(this);
	}

	@Override
	public boolean getBoolean(IIndex ima) {
		boolean result = false;
		try {
			result = Boolean.parseBoolean(mValue);
		} catch (NumberFormatException e) {
			// nothing to do
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	@Override
	public byte getByte(IIndex ima) {
		byte result = 0;
		try {
			result = Byte.parseByte(mValue);
		} catch (NumberFormatException e) {
			// nothing to do
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	@Override
	public char getChar(IIndex ima) {
		char result = 0;
		try {
			if (mValue.length() == 1) {
				result = mValue.charAt(0);
			}
		} catch (IndexOutOfBoundsException e) {
			// nothing to be done
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	@Override
	public double getDouble(IIndex ima) {
		double result = 0.0;
		try {
			result = Double.parseDouble(mValue);
		} catch (NumberFormatException e) {
			// nothing to do
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	@Override
	public Class<?> getElementType() {
		return String.class;
	}

	@Override
	public float getFloat(IIndex ima) {
		float result = 0;
		try {
			result = Float.parseFloat(mValue);
		} catch (NumberFormatException e) {
			// nothing to do
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	@Override
	public IIndex getIndex() {
		return mIndex;
	}

	@Override
	public int getInt(IIndex ima) {
		int result = 0;
		try {
			result = Integer.parseInt(mValue);
		} catch (NumberFormatException e) {
			// nothing to do
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	@Override
	public IArrayIterator getIterator() {
		return null;
	}

	@Override
	public long getLong(IIndex ima) {
		long result = 0;
		try {
			result = Long.parseLong(mValue);
		} catch (NumberFormatException e) {
			// nothing to do
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	@Override
	public Object getObject(IIndex ima) {
		return mValue;
	}

	@Override
	public int getRank() {
		return 1;
	}

	@Override
	public IArrayIterator getRegionIterator(int[] reference, int[] range)
			throws InvalidRangeException {
		return null;
	}

	@Override
	public int[] getShape() {
		return new int[] { 1 };
	}

	@Override
	public short getShort(IIndex ima) {
		short result = 0;
		try {
			result = Short.parseShort(mValue);
		} catch (NumberFormatException e) {
			// nothing to do
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	@Override
	public long getSize() {
		return 1;
	}

	@Override
	public Object getStorage() {
		return mValue;
	}

	@Override
	public void setBoolean(IIndex ima, boolean value) {
		mValue = Boolean.toString(value);
	}

	@Override
	public void setByte(IIndex ima, byte value) {
		mValue = Byte.toString(value);
	}

	@Override
	public void setChar(IIndex ima, char value) {
		mValue = Character.toString(value);
	}

	@Override
	public void setDouble(IIndex ima, double value) {
		mValue = Double.toString(value);
	}

	@Override
	public void setFloat(IIndex ima, float value) {
		mValue = Float.toString(value);
	}

	@Override
	public void setInt(IIndex ima, int value) {
		mValue = Integer.toString(value);
	}

	@Override
	public void setLong(IIndex ima, long value) {
		mValue = Long.toString(value);
	}

	@Override
	public void setObject(IIndex ima, Object value) {
		if (value instanceof String) {
			mValue = (String) value;
		}
	}

	@Override
	public void setShort(IIndex ima, short value) {
		mValue = Short.toString(value);
	}

	@Override
	public String shapeToString() {
		return null;
	}

	@Override
	public void setIndex(IIndex index) {
		mIndex = index;
	}

	@Override
	public ISliceIterator getSliceIterator(int rank)
			throws ShapeNotMatchException, InvalidRangeException {
		return null;
	}

	@Override
	public void releaseStorage() throws BackupException {
	}

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
	public boolean isDirty() {
		return false;
	}

	@Override
	public void setDirty(boolean dirty) {
	}

	@Override
	public IArray setDouble(double value) {
		mValue = Double.toString(value);
		return this;
	}

}
