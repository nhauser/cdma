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

import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.exception.BackupException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.IRange;
import org.cdma.interfaces.ISliceIterator;

public class DefaultArrayMatrix extends  DefaultArray {
    private Object mData; // Memory storage: an array of Object or primitive
	
    /**
     * Constructor
     * 
     * @param factory plug-in factory's name
     * @param storage carrying the data (multi-dimensional java array)
     * @throws InvalidArrayTypeException
     * @note raged arrays are not supported 
     */
	public DefaultArrayMatrix( String factory, Object storage ) throws InvalidArrayTypeException {
		super( factory, storage );
		
		// Check the storage's validity
		Class<?> clazz = storage.getClass();
		if( ! clazz.isArray() ) {
    		throw new InvalidArrayTypeException("Only java arrays are permitted for the storage!");
    	}
		
		mData = storage;
	}
	
    protected DefaultArrayMatrix( DefaultArrayMatrix array ) throws InvalidArrayTypeException {
    	super( array );
    	mData = array.mData;
    }

	@Override
	public boolean getBoolean(IIndex index) {
		boolean result;
		if( Boolean.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (boolean[]) mData );
			result = ((boolean[]) array)[ pos[ pos.length - 1 ] ];
		}
		else {
			result = (Boolean) getObject(index);
		}
		return result;
	}

	@Override
	public byte getByte(IIndex index) {
		byte result;
		if( Byte.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (byte[]) mData );
			result = ((byte[]) array)[ pos[ pos.length - 1 ] ];
		}
		else {
			result = (Byte) getObject(index);
		}
		return result;
	}

	@Override
	public char getChar(IIndex index) {
		char result;
		if( Character.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (char[]) mData );
			result = ((char[]) array)[ pos[ pos.length - 1 ] ];
		}
		else {
			result = (Character) getObject(index);
		}
		return result;
	}

	@Override
	public double getDouble(IIndex index) {
		double result;
		if( Double.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (double[]) mData );
			result = ((double[]) array)[ pos[ pos.length - 1 ] ];
		}
		else {
			Number tmp = (Number) getObject(index);
			result = tmp.doubleValue();
		}
		return result;
	}

	@Override
	public float getFloat(IIndex index) {
		float result;
		if( Double.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (float[]) mData );
			result = ((float[]) array)[ pos[ pos.length - 1 ] ];
		}
		else {
			Number tmp = (Number) getObject(index);
			result = tmp.floatValue();
		}
		return result;
	}

	@Override
	public int getInt(IIndex index) {
		int result;
		if( Integer.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, mData );
			result = ((int[]) array)[ pos[ pos.length - 1 ] ];
		}
		else {
			Number tmp = (Number) getObject(index);
			result = tmp.intValue();
		}
		return result;
	}

	@Override
	public long getLong(IIndex index) {
		long result;
		if( Long.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (long[]) mData );
			result = ((long[]) array)[ pos[ pos.length - 1 ] ];
		}
		else {
			Number tmp = (Number) getObject(index);
			result = tmp.longValue();
		}
		return result;
	}

	@Override
	public Object getObject(IIndex index) {
		Object result = null;

		if( index.getRank() == index.getRank() ) {
			result = mData;
			int[] counter = index.getCurrentCounter();
			for( int position : counter ) {
				result = java.lang.reflect.Array.get( result, position );
			}
		}
		
		return result;
	}

	@Override
	public short getShort(IIndex index) {
		short result;
		if( Short.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (short[]) mData );
			result = ((short[]) array)[ pos[ pos.length - 1 ] ];
		}
		else {
			Number tmp = (Number) getObject(index);
			result = tmp.shortValue();
		}
		return result;
	}

	@Override
	public void setBoolean(IIndex index, boolean value) {
		if( Boolean.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (boolean[]) mData );
			((boolean[]) array)[ pos[ pos.length - 1 ] ] = value;
		}
		else {
			setObject(index, value);
		}
	}

	@Override
	public void setByte(IIndex index, byte value) {
		if( Byte.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (byte[]) mData );
			((byte[]) array)[ pos[ pos.length - 1 ] ] = value;
		}
		else {
			setObject(index, value);
		}
	}

	@Override
	public void setChar(IIndex index, char value) {
		if( Character.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (char[]) mData );
			((char[]) array)[ pos[ pos.length - 1 ] ] = value;
		}
		else {
			setObject(index, value);
		}
	}

	@Override
	public void setDouble(IIndex index, double value) {
		if( Double.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (double[]) mData );
			((double[]) array)[ pos[ pos.length - 1 ] ] = value;
		}
		else {
			setObject(index, value);
		}
	}

	@Override
	public void setFloat(IIndex index, float value) {
		if( Float.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (float[]) mData );
			((float[]) array)[ pos[ pos.length - 1 ] ] = value;
		}
		else {
			setObject(index, value);
		}
	}

	@Override
	public void setInt(IIndex index, int value) {
		if( Integer.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (int[]) mData );
			((int[]) array)[ pos[ pos.length - 1 ] ] = value;
		}
		else {
			setObject(index, value);
		}
	}

	@Override
	public void setLong(IIndex index, long value) {
		if( Long.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (long[]) mData );
			((long[]) array)[ pos[ pos.length - 1 ] ] = value;
		}
		else {
			setObject(index, value);
		}
	}

	@Override
	public void setObject(IIndex index, Object value) {
		int[] pos = projectCoordinates(index);
		Object array = getMostVaryingRaw( pos, (boolean[]) mData );
		java.lang.reflect.Array.set(array, pos[ pos.length - 1 ], value);
	}

	@Override
	public void setShort(IIndex index, short value) {
		if( Short.TYPE.equals( getElementType() ) ) {
			int[] pos = projectCoordinates(index);
			Object array = getMostVaryingRaw( pos, (short[]) mData );
			((short[]) array)[ pos[ pos.length - 1 ] ] = value;
		}
		else {
			setObject(index, value);
		}
	}

	@Override
	public IArray setDouble(double value) {
		// Use an iterator to run through all slices of the array
        ISliceIterator iter;
        Object slab;
        int[] position;
		try {
			iter = getSliceIterator(1);
			while( iter.hasNext() ) {
	        	// increment the slice iterator
	        	iter.next();
	        	
	        	// get the position
	        	position = iter.getSlicePosition();
	        	
	        	// select the right slab
	        	slab = getData();
	        	for( int index = 0; index < position.length - 1; index++ ) {
	        		slab = java.lang.reflect.Array.get(slab, position[index]);
	        	}
	        	
	        	// Copy data
	        	java.util.Arrays.fill((double[]) slab, value);
	        	
	        }
		} catch (ShapeNotMatchException e) {
			Factory.getLogger().log(Level.WARNING, "Unable to copy data!", e);
		} catch (InvalidRangeException e) {
			Factory.getLogger().log(Level.WARNING, "Unable to copy data!", e);
		}
		return this;
	}
	
	@Override
	public IArray copy(boolean data) {
        DefaultArrayMatrix result = null;
		try {
			result = new DefaultArrayMatrix(this);
			if (data) {
				result.mData = this.copyToNDJavaArray();
	        }
		} catch (InvalidArrayTypeException e) {
			Factory.getLogger().log(Level.SEVERE, "Unable to copy the array!", e);
		}

        return result;
	}

	@Override
	public Object getStorage() {
		return mData;
	}

	@Override
	public void releaseStorage() throws BackupException {
		Factory.getLogger().log( Level.WARNING, "Unable to release storage", new NotImplementedException() );
	}
	
    // ---------------------------------------------------------
    /// Protected methods
    // ---------------------------------------------------------
    /**
     * Override this method in case of specific need (use of SoftReference for instance).
     * It is called each time the memory is accessed when (the array isn't locked)
     * 
     * @return the backing storage of the array
     */
    protected Object loadData() {
        return mData;
    }
    
    protected Object getData() {
    	Object result;
    	if( isLocked() ) {
    		result = mData;
    	}
    	else {
    		result = loadData();
    	}
    	return result;
    }
	
	@Override
	protected Object copyTo1DJavaArray() {
		int[] shape = getIndex().getProjectionShape();
		int[] position;
		int[] origin = getIndex().getProjectionOrigin();
		int slabLength = shape[shape.length - 1];
        Object slabSrc;
        Long length = new Long(getSize());
        
		// Instantiate a new convenient array for storage
        Class<?> type = getElementType();
        Object array = java.lang.reflect.Array.newInstance( type, length.intValue() );
        
        // Use an iterator to run through all slices of the array
        ISliceIterator iter;
        int start = 0;
		try {
			iter = getSliceIterator(1);
			while( iter.hasNext() ) {
	        	// increment the slice iterator
	        	iter.next();
	        	
	        	// get the slice position
	        	position = iter.getSlicePosition();
	        	
	        	// select the right slab in memory storage
	        	slabSrc = getData();
	        	for( int index = 0; index < position.length - 1; index++ ) {
	        		slabSrc = java.lang.reflect.Array.get(slabSrc, position[index] + origin[index]);
	        	}
	        	
	        	// Copy data
	        	System.arraycopy(slabSrc, origin[ origin.length - 1], array, start, slabLength);
	        	start += slabLength;
	        }
		} catch (ShapeNotMatchException e) {
			Factory.getLogger().log(Level.WARNING, "Unable to copy data!", e);
		} catch (InvalidRangeException e) {
			Factory.getLogger().log(Level.WARNING, "Unable to copy data!", e);
		}
		return array;
    }

    @Override
    protected Object copyToNDJavaArray() {
		int[] shape = getIndex().getProjectionShape();
		int[] visibleShape = getShape();
		int[] position;
		int[] origin = getIndex().getProjectionOrigin();
		boolean[] reduced = new boolean[shape.length];
		
		// Seek reduced dimensions
		int i = 0;
		int j = 0;
		for( int len : shape ) {
			if( len == visibleShape[i] ) {
				reduced[j] = false;
				i++;
			}
			else {
				reduced[j] = true;
			}
			j++;
		}
		
		int slabLength = shape[shape.length - 1];
        Object slabSrc, slabTgt;
        
		// Instantiate a new convenient array for storage
        Class<?> type = getElementType();
        Object array = java.lang.reflect.Array.newInstance(type, visibleShape);
        
        // Use an iterator to run through all slices of the array
        ISliceIterator iter;
        
		try {
			iter = getSliceIterator(1);
			while( iter.hasNext() ) {
	        	// increment the slice iterator
	        	iter.next();
	        	
	        	// get the position
	        	position = iter.getSlicePosition();
	        	
	        	// select the right slab
	        	slabSrc = getData();
	        	slabTgt = array;
	        	for( int index = 0; index < shape.length - 1; index++ ) {
	        		if( reduced[index] ) {
	        			slabSrc = java.lang.reflect.Array.get(slabSrc, origin[index]);
	        		}
	        		else {
	        			slabSrc = java.lang.reflect.Array.get(slabSrc, position[index] + origin[index]);
		        		slabTgt = java.lang.reflect.Array.get(slabTgt, position[index]);	
	        		}
	        	}
	        	
	        	// Copy data
	        	System.arraycopy(slabSrc, origin[ origin.length - 1], slabTgt, 0, slabLength);
	        	
	        }
		} catch (ShapeNotMatchException e) {
			Factory.getLogger().log(Level.WARNING, "Unable to copy data!", e);
		} catch (InvalidRangeException e) {
			Factory.getLogger().log(Level.WARNING, "Unable to copy data!", e);
		}
		return array;
    }

	// ---------------------------------------------------------
    /// Private methods
    // ---------------------------------------------------------
	/**
	 * Return the most varying raw of the given data, positioned at the given coordinates.
	 * @param coordinate position to get slab data from data
	 * @param data of type T to be sliced
	 * @return an Object that is an array of type T
	 */
	private <T> Object getMostVaryingRaw( int[] position, T data ) {
		return getMostVaryingRaw( position, data, 0 );
	}
	
	private <T, C> Object getMostVaryingRaw( int[] coordinates, T data, int depth ) {
		Object result = data;
		
		if( coordinates.length - 1 > depth ) {
			result = getMostVaryingRaw( coordinates, ((C[]) data)[coordinates[depth]], depth + 1);
		}
		
		return result;
	}
	
	/**
	 * Calculate the given index current position projected into this array view.
	 * 
	 * @param coordinate position to get slab data from data
	 * @param data of type T to be sliced
	 * @return an Object that is an array of type T
	 */
	private int[] projectCoordinates( IIndex index ) {
		DefaultIndex srcIndex = getIndex();
		int[] result = getIndex().getProjectionOrigin();
		int[] position = index.getCurrentCounter();
		int dimStart = 0, dimPos = 0;
		for(DefaultRange range : srcIndex.getRangeList() ) {
			if( ! range.reduced() ) {
				result[dimStart] += position[dimPos];
				dimPos++;
			}
			dimStart++;
		}
		
		return result;
	}
}
