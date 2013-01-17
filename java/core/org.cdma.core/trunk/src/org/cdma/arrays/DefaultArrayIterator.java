// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr)
// ****************************************************************************
package org.cdma.arrays;

import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.IIndex;
import org.cdma.utils.ArrayTools;

public class DefaultArrayIterator implements IArrayIterator {
	/// Members
	private IArray mArray;     // Array that supports the storage and a view on it
	private IIndex mIndex;     // Iterator's view of the array (as described in mArray)
	private boolean mAccess;  // Indicates that this can access the storage memory or not
	private boolean mStarted; // Has the next method been called at least one time

	public DefaultArrayIterator(IArray array) {
		this( array, array.getIndex(), true );
	}

	public DefaultArrayIterator(IArray array, IIndex index) {
		this(array, index, true);
	}

	public DefaultArrayIterator(IArray array, IIndex index, boolean accessData) {
		mArray = array.copy(false);
		try {
			mIndex = index.clone();
		} catch (CloneNotSupportedException e) {
			mIndex = index;
		}
		mAccess  = accessData;
		mStarted = false;
		
		// set the current starting point
		int[] count = index.getCurrentCounter();
		if( index.getRank() > 0 ) {
			count[count.length - 1]--;
			mIndex.set(count);
		}
		mArray.setIndex(mIndex);
	}

	@Override
	public boolean getBooleanNext() {
		incrementIndex();
		return mArray.getBoolean(mIndex);
	}

	@Override
	public byte getByteNext() {
		incrementIndex();
		return mArray.getByte(mIndex);
	}

	@Override
	public char getCharNext() {
		incrementIndex();
		return mArray.getChar(mIndex);
	}

	@Override
	public int[] getCounter() {
		return mIndex.getCurrentCounter();
	}

	@Override
	public double getDoubleNext() {
		incrementIndex();
		return mArray.getDouble(mIndex);
	}

	@Override
	public float getFloatNext() {
		incrementIndex();
		return mArray.getFloat(mIndex);
	}

	@Override
	public int getIntNext() {
		incrementIndex();
		return mArray.getInt(mIndex);
	}

	@Override
	public long getLongNext() {
		incrementIndex();
		return mArray.getLong(mIndex);
	}

	@Override
	public Object getObjectNext() {
		return next();
	}

	@Override
	public short getShortNext() {
		incrementIndex();
		return mArray.getShort(mIndex);
	}

	@Override
	public boolean hasNext() {
		boolean result;
		if( mStarted ) {
			long index = mIndex.currentElement();
			long last = mIndex.lastElement();
			result = (index < last && index >= -1);
		}
		else {
			result = true;
		}
		return result;
	}

	@Override
	public Object next() {
		Object result;
		incrementIndex();
		if (mAccess) {
			long currentPos = mIndex.currentElement();
			if (currentPos <= mIndex.lastElement() && currentPos != -1) {
				result = mArray.getObject(mIndex);
			} else {
				result = null;
			}
		}
		else {
			result = null;
		}
		return result;
	}

	@Override
	public void setBoolean(boolean val) {
		setObject(val);
	}

	@Override
	public void setByte(byte val) {
		setObject(val);
	}

	@Override
	public void setChar(char val) {
		setObject(val);
	}

	@Override
	public void setDouble(double val) {
		setObject(val);
	}

	@Override
	public void setFloat(float val) {
		setObject(val);
	}

	@Override
	public void setInt(int val) {
		setObject(val);
	}

	@Override
	public void setLong(long val) {
		setObject(val);
	}

	@Override
	public void setObject(Object val) {
		mArray.setObject(mIndex, val);
	}

	@Override
	public void setShort(short val) {
		setObject(val);
	}

	@Override
	public String getFactoryName() {
		return mArray.getFactoryName();
	}

	/// protected method
	protected void incrementIndex() {
		mStarted = true;
		int[] current = mIndex.getCurrentCounter();
		int[] shape = mIndex.getShape();
		ArrayTools.incrementCounter(current, shape);
		mIndex.set(current);
	}
}