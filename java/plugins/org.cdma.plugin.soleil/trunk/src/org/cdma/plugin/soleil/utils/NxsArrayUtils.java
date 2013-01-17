//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.soleil.utils;

import java.util.List;

import org.cdma.arrays.DefaultIndex;
import org.cdma.engine.nexus.array.NexusIndex;
import org.cdma.engine.nexus.utils.NexusArrayUtils;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IRange;
import org.cdma.interfaces.ISliceIterator;
import org.cdma.plugin.soleil.array.NxsArray;
import org.cdma.plugin.soleil.array.NxsIndex;
import org.cdma.utils.IArrayUtils;

public final class NxsArrayUtils implements IArrayUtils {
    private IArrayUtils mUtils;


    public NxsArrayUtils( NxsArray array) {
        mUtils = new NexusArrayUtils(array);
    }

    @Override
    public Object copyTo1DJavaArray() {
        // Instantiate a new convenient array for storage
        int length   = ((Long) getArray().getSize()).intValue();
        Class<?> type = getArray().getElementType();
        Object array = java.lang.reflect.Array.newInstance(type, length);

        NexusIndex storageIndex = ((NxsIndex) getArray().getIndex()).getIndexStorage();
        DefaultIndex matrixIndex = ((NxsIndex) getArray().getIndex()).getIndexMatrix();
        
        // If the storing array is a stack of DataItem
        Long size = matrixIndex.getSize();
        Long nbMatrixCells  = size == 0 ? 1 : size;
        Long nbStorageCells = storageIndex.getSize();
        
        Long startPos = storageIndex.elementOffset( new int[storageIndex.getRank()] );
        Object fullArray = getArray().getStorage();
        Object partArray = null;
        int[] posMatrix = new int[1];
        for( int cell = 0; cell < nbMatrixCells; cell++ ) {
        	posMatrix[0] = cell;
            partArray = java.lang.reflect.Array.get(fullArray, cell );
            System.arraycopy(partArray, startPos.intValue(), array, cell * nbStorageCells.intValue(), nbStorageCells.intValue());
        }

        return array;
    }


    @Override
    public Object copyToNDJavaArray() {
        return copyMatrixItemsToMultiDim();
    }

    // --------------------------------------------------
    // tools methods
    // --------------------------------------------------
    public static Object copyJavaArray(Object array) {
        Object result = array;
        if( result == null ) {
            return null;
        }

        // Determine rank of array (by parsing data array class name)
        String sClassName = array.getClass().getName();
        int iRank  = 0;
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
        int[] shape    = new int[iRank];

        // Fill dimension size array
        for ( int i = 0; i < iRank; i++) {
            shape[i] = java.lang.reflect.Array.getLength(result);
            result = java.lang.reflect.Array.get(result,0);
        }

        // Define a convenient array (shape and type)
        result = java.lang.reflect.Array.newInstance(array.getClass().getComponentType(), shape);

        return copyJavaArray(array, result);
    }

    public static Object copyJavaArray(Object source, Object target) {
        Object item = java.lang.reflect.Array.get(source, 0);
        int length = java.lang.reflect.Array.getLength(source);

        if( item.getClass().isArray() ) {
            Object tmpSrc;
            Object tmpTar;
            for( int i = 0; i < length; i++ ) {
                tmpSrc = java.lang.reflect.Array.get(source, i);
                tmpTar = java.lang.reflect.Array.get(target, i);
                copyJavaArray( tmpSrc, tmpTar);
            }
        }
        else {
            System.arraycopy(source, 0, target, 0, length);
        }

        return target;
    }


    @Override
    public IArray getArray() {
        return mUtils.getArray();
    }

    @Override
    public void copyTo(IArray newArray) throws ShapeNotMatchException {
        mUtils.copyTo(newArray);
    }

    @Override
    public Object get1DJavaArray(Class<?> wantType) {
        return mUtils.get1DJavaArray(wantType);
    }

    @Override
    public void checkShape(IArray newArray) throws ShapeNotMatchException {
        mUtils.checkShape(newArray);
    }

    @Override
    public IArrayUtils concatenate(IArray array) throws ShapeNotMatchException {
        return mUtils.concatenate(array);
    }

    @Override
    public IArrayUtils reduce() {
        return mUtils.reduce();
    }

    @Override
    public IArrayUtils reduce(int dim) {
        return mUtils.reduce(dim);
    }

    @Override
    public IArrayUtils reduceTo(int rank) {
        return mUtils.reduceTo(rank);
    }

    @Override
    public IArrayUtils reshape(int[] shape) throws ShapeNotMatchException {
        return mUtils.reshape(shape);
    }

    @Override
    public IArrayUtils section(int[] origin, int[] shape)
            throws InvalidRangeException {
        return mUtils.section(origin, shape);
    }

    @Override
    public IArrayUtils section(int[] origin, int[] shape, long[] stride)
            throws InvalidRangeException {
        return mUtils.section(origin, shape, stride);
    }

    @Override
    public IArrayUtils sectionNoReduce(int[] origin, int[] shape, long[] stride)
            throws InvalidRangeException {
        return mUtils.sectionNoReduce(origin, shape, stride);
    }

    @Override
    public IArrayUtils sectionNoReduce(List<IRange> ranges)
            throws InvalidRangeException {
        return mUtils.sectionNoReduce(ranges);
    }

    @Override
    public IArrayUtils slice(int dim, int value) {
        return mUtils.slice(dim, value);
    }

    @Override
    public IArrayUtils transpose(int dim1, int dim2) {
        return mUtils.transpose(dim1, dim2);
    }

    @Override
    public boolean isConformable(IArray array) {
        return mUtils.isConformable(array);
    }

    @Override
    public IArrayUtils eltAnd(IArray booleanMap) throws ShapeNotMatchException {
        return mUtils.eltAnd(booleanMap);
    }

    @Override
    public IArrayUtils integrateDimension(int dimension, boolean isVariance)
            throws ShapeNotMatchException {
        return mUtils.integrateDimension(dimension, isVariance);
    }

    @Override
    public IArrayUtils enclosedIntegrateDimension(int dimension,
            boolean isVariance) throws ShapeNotMatchException {
        return mUtils.enclosedIntegrateDimension(dimension, isVariance);
    }

    @Override
    public IArrayUtils flip(int dim) {
        return mUtils.flip(dim);
    }

    @Override
    public IArrayUtils permute(int[] dims) {
        return mUtils.permute(dims);
    }


    // --------------------------------------------------
    // private methods
    // --------------------------------------------------
    /**
     * Copy the backing storage of this NxsArray into multidimensional 
     * corresponding Java primitive array
     */
    private Object copyMatrixItemsToMultiDim() {
    	// Create an array corresponding to the shape
        NxsArray array = (NxsArray) getArray();
        int[] shape   = array.getShape();
        Object result  = java.lang.reflect.Array.newInstance(array.getElementType(), shape);
        
        // Get the array's storage
        Object values = array.getStorage();
        
        int[] current;
        int   length;
        Long  startCell;
        
        try {
            
            ISliceIterator iter = array.getSliceIterator(1);
            NxsIndex startIdx   = (NxsIndex) array.getIndex();
            NexusIndex storage  = startIdx.getIndexStorage();
            DefaultIndex items  = startIdx.getIndexMatrix();
            startIdx.setOrigin(new int[startIdx.getRank()]);
            
            length = startIdx.getShape()[ startIdx.getRank() - 1 ];
            
            storage.set( new int[storage.getRank()] );

            // Turning buffers
            Object slab = null;
            Object dataset = null;
            
            // Copy each slice
            int last = -1;
            int cell = (int) (items.currentElement() - items.firstElement());
            while( iter.hasNext() ) {
                iter.next();
                slab = result;

                // Getting the right slab in the multidimensional resulting array
                current = iter.getSlicePosition();
                startIdx.set(current);
                for( int pos = 0;  pos < current.length - 1; pos++ ) {
                    slab = java.lang.reflect.Array.get(slab, current[pos]);
                }
                
                cell = (int) (items.currentElement() - items.firstElement());
                if( last != cell ) {
                	dataset = java.lang.reflect.Array.get(values, cell );
                	last = cell;
                }
                startCell = storage.currentElement();
                System.arraycopy(dataset, startCell.intValue(), slab, 0, length);
            }
        } catch (ShapeNotMatchException e) {
        } catch (InvalidRangeException e) {
        }
        return result;
    }
}
