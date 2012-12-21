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

import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.ISliceIterator;
import org.cdma.math.IArrayMath;
import org.cdma.utils.ArrayTools;
import org.cdma.utils.IArrayUtils;

/**
 * This helper class aims to provide a simple way to instantiate IArray implementation
 * which have a standard way of managing their backing array storage. 
 * 
 * Those implementations can manage:
 *  - java inline arrays (a single raw of data that can represent
 * multidimensional array i.e an image for example).
 *  - java multi-dimensional arrays which the shape can be fully described by a 
 *  simple int[]. It means no raged array, only 'squared' matrices.
 *  
 * @author rodriguez
 *
 */

public abstract class DefaultArray implements IArray {
	/**
	 * Instantiate a default IArray implementation having the given storage inline
	 * storage and the given shape
	 * 
	 * @param factory: name of the plug-in
	 * @param storage: inline storage of memory 
	 * @param shape: how this inline storage should be interpreted (1D, 2D,..., ND)
	 * @return a default array implementation
	 * @throws InvalidArrayTypeException
	 */
	static public IArray instantiateDefaultArray(String factory, Object storage, int[] shape ) throws InvalidArrayTypeException {
    	IArray result = null;
		int[] storageShape = ArrayTools.detectShape(storage);
		if( storageShape.length == 1 ) {
			result = new DefaultArrayInline(factory, storage, shape);
		}
		else {
			result = new DefaultArrayMatrix(factory, storage);
		}
		return result;
	}
	
	/**
	 * Instantiate a default IArray implementation having the given multi-dimensional
	 * storage. The shape will be determined according the storage.
	 * 
	 * @param factory: name of the plug-in
	 * @param storage: storage of memory 
	 * @return a default array implementation
	 * @throws InvalidArrayTypeException
	 */
    static public IArray instantiateDefaultArray(String factory, Object storage ) throws InvalidArrayTypeException {
    	IArray result = null;
    	int[] shape = ArrayTools.detectShape(storage);
		switch( shape.length ) {
		case 0:
			Object array = java.lang.reflect.Array.newInstance(storage.getClass(), 1);
			java.lang.reflect.Array.set(array, 0, storage);
			result = new DefaultArrayInline(factory, array, shape);
			break;
		case 1:
			result = new DefaultArrayInline(factory, storage, shape);
			break;
		default:
			result = new DefaultArrayMatrix(factory, storage);
			break;
		}
		return result;
	}
    
    private DefaultIndex mIndex;        // IIndex corresponding to this IArray (dimension sizes defining the viewable part of the array)
    private boolean     mIsDirty;      // Is the array synchronized with the handled file
    private String       mFactory;      // Name of the instantiating factory 
    private Class<?>     mClazz;        // Type of the array's element
    private boolean     mLock;
    
    protected DefaultArray( String factory, Object inlineArray, int[] shape ) throws InvalidArrayTypeException {
    	if( inlineArray == null ) {
    		throw new InvalidArrayTypeException("Null backing storage is not permitted!");
    	}

    	mLock    = false;
    	mClazz   = inlineArray.getClass().getComponentType();
		mIndex   = new DefaultIndex(mFactory, shape.clone());
		mIsDirty = false;
		mFactory = factory;
    }
    
    protected DefaultArray( String factory, Object array ) throws InvalidArrayTypeException {
    	if( array == null ) {
    		throw new InvalidArrayTypeException("Null backing storage is not permitted!");
    	}

    	// Check the given array is a real java array
    	Class<?> clazz = array.getClass();
    	while( clazz != null && clazz.isArray() ) {
    		clazz = clazz.getComponentType();
    	}
    	mClazz   = clazz;
		mIndex   = new DefaultIndex(mFactory, ArrayTools.detectShape( array ) );
		mFactory = factory;
		mLock    = false;
		mIsDirty = false;
    }
    
    protected DefaultArray( DefaultArray array ) {
        mIndex   = (DefaultIndex) array.mIndex.clone();
        mIsDirty = array.mIsDirty;
        mFactory = array.mFactory; 
        mClazz   = array.mClazz;
        mLock    = array.mLock;
    }
    
    
    
    @Override
    public IArray copy() {
        return copy(true);
    }
    
    @Override
    public void lock() {
        mLock = true;
    }
    
    @Override
    public void unlock() {
    	mLock = false;
    }
    
    public boolean isLocked() {
    	return mLock;
    }
    
    @Override
    public void setIndex(IIndex index) {
    	if( index instanceof DefaultIndex ) {
    		mIndex = (DefaultIndex) index;
    	}
    	else {
    		mIndex = new DefaultIndex(index);
    	}
    }

    @Override
    public String getFactoryName() {
        return mFactory;
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

    // / IArray data manipulation
    @Override
    public DefaultIndex getIndex() {
        return mIndex;
    }

	@Override
	public IArrayIterator getIterator() {
		return new DefaultArrayIterator(this);
	}

    @Override
    public int getRank() {
        return mIndex.getRank();
    }

    @Override
    public IArrayIterator getRegionIterator(int[] reference, int[] range) throws InvalidRangeException {
    	IArrayIterator iterator = null;
		IIndex index = mIndex.clone();
    	index.setShape(range);
    	index.setOrigin(reference);
    	iterator = new DefaultArrayIterator(this, index);

		return iterator;
    }

    @Override
    public long getRegisterId() {
        throw new NotImplementedException();
    }

    @Override
    public long getSize() {
        return mIndex.getSize();
    }

    @Override
    public ISliceIterator getSliceIterator(int rank) throws ShapeNotMatchException, InvalidRangeException {
        return new DefaultSliceIterator(this, rank);
    }
    
    @Override
    public IArrayUtils getArrayUtils() {
    	return new DefaultArrayUtils(this);
    }
    
    
    @Override
    public IArrayMath getArrayMath() {
        throw new NotImplementedException();
    }

    @Override
    public int[] getShape() {
        return mIndex.getShape();
    }

    @Override
    public Class<?> getElementType() {
        return mClazz;
    }
    
    @Override
    public void setDirty(boolean dirty) {
    	mIsDirty = dirty;
    }
    
    @Override
    public boolean isDirty() {
        return mIsDirty;
    }

    /**
     * Override this method in case of specific need (use of SoftReference for instance).
     * It is called each time the memory is accessed when (the array isn't locked)
     * 
     * @return the backing storage of the array
     */
    protected abstract Object loadData();

    /**
     * Override this method in case of specific need (use of SoftReference for instance).
     * It is called each time the memory is accessed when (the array isn't locked)
     * 
     * @return the backing storage of the array
     */
    protected abstract Object getData();

    /**
     * Override this method in case of specific need (use of SoftReference for instance).
     * It is called each time the memory is accessed when (the array isn't locked)
     * 
     * @return the backing storage of the array
     */
    protected abstract Object copyTo1DJavaArray();

    /**
     * Override this method in case of specific need (use of SoftReference for instance).
     * It is called each time the memory is accessed when (the array isn't locked)
     * 
     * @return the backing storage of the array
     */
    protected abstract Object copyToNDJavaArray();
}
