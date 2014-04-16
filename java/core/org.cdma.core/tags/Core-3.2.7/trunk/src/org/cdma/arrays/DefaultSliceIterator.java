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
import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.ISliceIterator;

public class DefaultSliceIterator implements ISliceIterator {
    /// Members
    private DefaultArrayIterator mIterator;      // iterator of the whole_array
    private IArray               mArray;         // array of the original shape containing all slices
    private int[]               mDimension;     // shape of the slice
    private IArray               mSlice;         // array that will be returned when asking getArrayNext

    /// Constructor
    /**
     * Create a new NxsSliceIterator that permits to iterate over
     * slices that are the last dim dimensions of this array.
     * 
     * @param array source of the slice
     * @param dim returned dimensions
     */
    /*
    public DefaultSliceIterator(final DefaultArray array, final int dim) throws InvalidRangeException {
        // If ranks are equal, make sure at least one iteration is performed.
        // We cannot use 'reshape' (which would be ideal) as that creates a
        // new copy of the array storage and so is unusable
        mArray = array;
        
        DefaultIndex arrayIndex = array.getIndex();
        
		int[] shape = arrayIndex.getProjectionShape().clone();
		int[] origin = arrayIndex.getProjectionOrigin().clone();
        
        int[] rangeList = shape.clone();
        //long[] stride = arrayIndex.getStride().clone();
        int rank = mArray.getRank();

        // shape to make a 2D array from multiple-dim array
        mDimension = new int[dim];
        System.arraycopy(shape, shape.length - dim, mDimension, 0, dim);
        for (int i = 0; i < dim; i++) {
            rangeList[rank - i - 1] = 1;
        }
        // Create an iterator over the higher dimensions. We leave in the
        // final dimensions so that we can use the getCurrentCounter method
        // to create an origin.

			IIndex index = arrayIndex.clone();
			
	        // As we get a reference on array's IIndex we directly modify it
	        index.setOrigin(origin);
	        //index.setStride(stride);
	        index.setShape(rangeList);
	        index.set(new int[index.getRank()]);
	        mIterator = new DefaultArrayIterator(mArray, index, false);
    }
    */
    public DefaultSliceIterator(final IArray array, final int dim) throws InvalidRangeException {
        // If ranks are equal, make sure at least one iteration is performed.
        // We cannot use 'reshape' (which would be ideal) as that creates a
        // new copy of the array storage and so is unusable
        mArray = array;
        
        IIndex arrayIndex = array.getIndex();
        
		int[] shape = arrayIndex.getShape().clone();
		int[] origin = arrayIndex.getOrigin().clone();
        
        int[] rangeList = shape.clone();
        long[] stride = arrayIndex.getStride().clone();
        int rank = mArray.getRank();

        // shape to make a 2D array from multiple-dim array
        mDimension = new int[dim];
        System.arraycopy(shape, shape.length - dim, mDimension, 0, dim);
        for (int i = 0; i < dim; i++) {
            rangeList[rank - i - 1] = 1;
        }
        // Create an iterator over the higher dimensions. We leave in the
        // final dimensions so that we can use the getCurrentCounter method
        // to create an origin.

		try {
			IIndex index = arrayIndex.clone();
			
	        // As we get a reference on array's IIndex we directly modify it
	        index.setOrigin(origin);
	        index.setStride(stride);
	        index.setShape(rangeList);
	        index.set(new int[index.getRank()]);
	        mIterator = new DefaultArrayIterator(index);

		} catch (CloneNotSupportedException e) {
			Factory.getLogger().log(Level.SEVERE, "Unable to initialize slice iterator!", e);
		}

    }

    /// Public methods
    @Override
    public IArray getArrayNext() throws InvalidRangeException {
        next();
        if( mSlice == null ) {
            createSlice();
        }
        else {
            updateSlice();
        }
        return mSlice.copy(false);
    }

    @Override
    public int[] getSliceShape() throws InvalidRangeException {
        return mDimension.clone();
    }

    public int[] getSlicePosition() {
        return mIterator.getCounter();
    }

    public int[] getSlicePositionProjection() {
        return mIterator.getCounterProjection();
    }
    
    @Override
    public boolean hasNext() {
        return mIterator.hasNext();
    }

    @Override
    public void next() {
        mIterator.next();
    }

    @Override
    public String getFactoryName() {
        return mArray.getFactoryName();
    }
    
    // ---------------------------------------------------------
    /// Private methods
    // ---------------------------------------------------------
    private void createSlice() throws InvalidRangeException {
        mSlice = mArray.copy(false);
        updateSlice();
    }
    
    private void updateSlice() throws InvalidRangeException {
        int i = 0;
        int[] iShape = mArray.getShape();
        int[] iOrigin = mArray.getIndex().getOrigin();
        int[] iCurPos = mIterator.getCounter();
        for (int pos : iCurPos) {
            if (pos >= iShape[i]) {
                mSlice = null;
                return;
            }
            iOrigin[i] += pos;
            i++;
        }
        
        java.util.Arrays.fill(iShape, 1);
        System.arraycopy(mDimension, 0, iShape, iShape.length - mDimension.length, mDimension.length);
        IIndex index;
        try {
            index = mArray.getIndex().clone();
            index.setShape(iShape);
            index.setOrigin(iOrigin);
            index.reduce();
            mSlice.setIndex(index);
        } catch (CloneNotSupportedException e) {
            mSlice = null;
        }
    }
}
