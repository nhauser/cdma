/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.utilities.conversion;

import org.cdma.utilities.conversion.ArrayConverters.StringArrayConverter;

/**
 * Convert a String array into a primitive array of the same shape.
 */

public class StringArrayToPrimitiveArray {
	private Class<?> mClazz;
	private Object mSource;
	private Object mTarget;
	private int[] mShape;
	private StringArrayConverter mConverter;
	
	/**
	 * Constructor
	 * @param array of string to be converted
	 * @param shape of the desired output array
	 * @param clazz the type of element expected in the output array
	 */
	public StringArrayToPrimitiveArray(Object[] array, int[] shape, Class<?> clazz) {
		this( array, shape, ArrayConverters.detectConverter(clazz) );
	}
	
	/**
	 * Constructor
	 * @param array of string to be converted
	 * @param shape of the desired output array
	 * @param converted that will do the array conversion
	 */
	public StringArrayToPrimitiveArray(Object[] array, int[] shape, StringArrayConverter converter) {
		mSource = array;
		mClazz  = converter.primitiveType();
		mShape  = shape;
		mTarget = null;
		mConverter = converter;
	}
	
	public Object convert() {
		if( mTarget == null && mSource != null && mShape != null && mClazz != null ) {
			// Create an array of the given shape and form
			mTarget = java.lang.reflect.Array.newInstance( mClazz, mShape );
			
			// Fill the array with the String converted into primitive values
			convert( mSource, mTarget );
		}

		return mTarget;
	}

	/** 
	 * Seek the last dim of the array and apply the conversion method on it
	 * @param srcArray input array not modified
	 * @param tgtArray output array that is modified
	 */
	private void convert( Object srcArray, Object tgtArray ) {
        Object srcBuf;
        Object tgtBuf;
        
        // Get the length of the source array
        int length = java.lang.reflect.Array.getLength(srcArray);
        
        // For each dimension from source and target
        for (int index = 0; index < length; index++) {
        	// iterate over source array cell
            srcBuf = java.lang.reflect.Array.get(srcArray, index);
            
            // if source array cell are still arrays
            if (srcBuf.getClass().isArray() ) {
            	// Get the corresponding cell int the target array
            	tgtBuf = java.lang.reflect.Array.get(tgtArray, index);

            	// Recursively seek the most varying dimension
            	convert(srcBuf, tgtBuf);
            }
            // Convert the whole dimension
            else {
            	mConverter.convert( (String[]) srcArray, tgtArray);
            	return;
            }
        }
	}
	
}
