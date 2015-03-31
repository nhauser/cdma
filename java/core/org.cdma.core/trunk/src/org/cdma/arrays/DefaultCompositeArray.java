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
//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.arrays;


import org.cdma.Factory;
import org.cdma.exception.BackupException;
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

public final class DefaultCompositeArray implements IArray {
    private final String factoryName;
    private Object mData; // It's an array of values
    private DefaultCompositeIndex mIndex; // IIndex corresponding to mArray shape
    private final IArray[] mArrays; // IArray of IArray

    public DefaultCompositeArray(final String factName, final IArray[] arrays) {
        factoryName = factName;
        mArrays   = arrays.clone();
        mData     = null;
        initDimSize();

        // Define the same viewable part for all sub-IArray
        DefaultIndex index = mIndex.getIndexStorage();
        for( IArray array : mArrays ) {
            array.setIndex(index.clone());
        }
    }

    public DefaultCompositeArray(final DefaultCompositeArray array) {
        mIndex = (DefaultCompositeIndex) array.mIndex.clone();
        mData = array.mData;

        IIndex index = mIndex.getIndexStorage();
        mArrays      = new IArray[array.mArrays.length];
        for( int i = 0; i < array.mArrays.length; i++ ) {
            mArrays[i] = array.mArrays[i].copy(false);
            mArrays[i].setIndex(index);

        }
        factoryName = array.factoryName;
    }


    public DefaultCompositeArray(final String factoryName, final Object oArray, final int[] iShape)
            throws InvalidArrayTypeException {
        this(factoryName, new IArray[] { new DefaultArrayInline(factoryName, oArray, iShape) });
    }

    @Override
    public Object getObject(final IIndex index) {
        return this.get(index);
    }

    @Override
    public IArray copy() {
        return copy(true);
    }

    // @Override
    // public IArray copy(boolean data) {
    // DefaultArrayInline result = null;
    // try {
    // result = new DefaultArrayInline(this.factoryName, mArrays[0].getClass(), getShape());
    //
    //
    // if( data ) {
    // result.setData(ArrayTools.copyJavaArray(mData));
    // }
    // }
    // catch (InvalidArrayTypeException e) {
    // // TODO Auto-generated catch block
    // e.printStackTrace();
    // }
    // return result;
    // }

    @Override
    public IArray copy(final boolean data) {
        DefaultCompositeArray result = new DefaultCompositeArray(this);

        if (data) {
            result.mData = ArrayTools.copyJavaArray(mData);
        }

        return result;
    }

    @Override
    public IArrayMath getArrayMath() {
        return new DefaultCompositeArrayMath(this, Factory.getFactory(getFactoryName()));
        // return mArrays[0].getArrayMath();
    }

    @Override
    public IArrayUtils getArrayUtils() {
        return new DefaultCompositeArrayUtils(this);
    }

    @Override
    public boolean getBoolean(final IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[nodeIndex].getBoolean(itemIdx);
    }

    @Override
    public byte getByte(final IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[nodeIndex].getByte(itemIdx);
    }

    @Override
    public char getChar(final IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[nodeIndex].getChar(itemIdx);
    }

    @Override
    public double getDouble(final IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[nodeIndex].getDouble(itemIdx);
    }

    @Override
    public float getFloat(final IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[nodeIndex].getFloat(itemIdx);
    }

    @Override
    public int getInt(final IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[nodeIndex].getInt(itemIdx);
    }

    @Override
    public long getLong(final IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[nodeIndex].getLong(itemIdx);
    }

    @Override
    public short getShort(final IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[nodeIndex].getShort(itemIdx);
    }

    @Override
    public Class<?> getElementType() {
        Class<?> result = null;
        if( mArrays != null )
        {
            result = mArrays[0].getElementType();
        }
        return result;
    }

    @Override
    public IIndex getIndex() {
        return mIndex.clone();
    }

    @Override
    public IArrayIterator getIterator() {
        DefaultCompositeIndex index = (DefaultCompositeIndex) mIndex.clone();
        index.set( new int[index.getRank()] );
        return new DefaultArrayIterator(this, index );
    }

    @Override
    public int getRank() {
        return mIndex.getRank();
    }

    @Override
    public IArrayIterator getRegionIterator(final int[] reference, final int[] range)
            throws InvalidRangeException {
        int[] shape = mIndex.getShape();
        IIndex index = new DefaultCompositeIndex(this.factoryName, shape, reference, range);
        return new DefaultArrayIterator(this, index);
    }

    @Override
    public int[] getShape() {
        int[] result = mIndex.getShape();
        return result;
    }

    @Override
    public long getSize() {
        // DefaultCompositeIndex idx = (DefaultCompositeIndex) getIndex();
        // TODO Check HDF5 regression on this:
        // long result = mArrays.length * idx.getIndexStorage().getSize();
        long result = mIndex.getSize();
        return result;
    }

    @Override
    public Object getStorage() {
        Object result = mData;
        IIndex index = mIndex.getIndexMatrix().clone();
        int rank = index.getRank();

        if ( mData == null && mArrays != null ) {
            Object[] array;
            int offset;

            // Case of several NexusArray in the visible part of the matrix
            if (rank != 0) {
                int[] pos = new int[1];
                int size = (int) index.getSize();
                array = new Object[size];
                for (int i = 0; i < size; i++) {
                    pos[0] = i;
                    index.set(pos);
                    offset = (int) index.currentElement();
                    if (offset < 0) {
                        array = null;
                        break;
                    }
                    array[i] = mArrays[offset].getStorage();
                }
            }
            // Case of one NexusArray in the visible part of the matrix
            else {
                offset = (int) index.currentElement();
                array = new Object[1];
                array[0] = mArrays[offset].getStorage();
            }
            result = array;
        }
        return result;
    }

    @Override
    public void setBoolean(final IIndex ima, final boolean value) {
        set(ima, value);
    }

    @Override
    public void setByte(final IIndex ima, final byte value) {
        set(ima, value);
    }

    @Override
    public void setChar(final IIndex ima, final char value) {
        set(ima, value);
    }

    @Override
    public void setDouble(final IIndex ima, final double value) {
        set(ima, value);
    }

    @Override
    public void setFloat(final IIndex ima, final float value) {
        set(ima, value);
    }

    @Override
    public void setInt(final IIndex ima, final int value) {
        set(ima, value);
    }

    @Override
    public void setLong(final IIndex ima, final long value) {
        set(ima, value);
    }

    @Override
    public void setObject(final IIndex ima, final Object value) {
        set(ima, value);
    }

    @Override
    public void setShort(final IIndex ima, final short value) {
        set(ima, value);
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
    public void setIndex(final IIndex index) {
        if (index instanceof DefaultCompositeIndex) {
            mIndex = (DefaultCompositeIndex) index;
        }
        else {
            mIndex = new DefaultCompositeIndex(this.factoryName, mIndex.getIndexMatrix().getRank(),
                    index.getShape(), index.getOrigin(), index.getShape());
            mIndex.set(index.getCurrentCounter());
        }

        for( IArray array : mArrays ) {
            array.setIndex(mIndex.getIndexStorage());
        }
    }

    @Override
    public ISliceIterator getSliceIterator(final int rank)
            throws ShapeNotMatchException, InvalidRangeException {
        return new DefaultSliceIterator(this, rank);
    }

    @Override
    public void releaseStorage() throws BackupException {
        throw new NotImplementedException();
    }

    @Override
    public long getRegisterId() {
        throw new NotImplementedException();
    }

    @Override
    public void lock() {
        throw new NotImplementedException();
    }

    @Override
    public void unlock() {
        throw new NotImplementedException();
    }

    @Override
    public boolean isDirty() {
        throw new NotImplementedException();
    }

    @Override
    public IArray setDouble(final double value) {
        throw new NotImplementedException();
    }

    @Override
    public String getFactoryName() {
        return this.factoryName;
    }

    @Override
    public String toString() {
        return mIndex.toString();
    }

    @Override
    public void setDirty(final boolean dirty) {
        throw new NotImplementedException();
    }

    /**
     * @return this array sub-parts if any
     */
    public IArray[] getArrayParts() {
        return mArrays;
    }

    // ---------------------------------------------------------
    /// Private methods
    // ---------------------------------------------------------
    /**
     * InitDimSize
     * Initialize member dimension sizes 'm_iDimS' according to defined member data 'm_oData'
     */
    private void initDimSize()
    {
        // Check data existence
        if( mArrays != null ) {
            // Set dimension rank
            int matrixRank = mArrays.length > 1 ? 1 : 0;
            int[] shape    = new int[ matrixRank + mArrays[0].getRank() ];


            // Fill dimension size array
            if( matrixRank > 0 ) {
                shape[0] = mArrays.length;
            }

            int i = 0;
            for( int size : mArrays[0].getShape() ) {
                shape[i + matrixRank] = size;
                i++;
            }

            mIndex = new DefaultCompositeIndex(factoryName, matrixRank, shape,
                    new int[shape.length], shape);
        }
    }

    /**
     * Get the object targeted by given index and return it (eventually using auto-boxing). It's the
     * central data access method that all other methods rely on.
     *
     * @param index targeting a cell
     * @return the content of cell designed by the index
     * @throws InvalidRangeException if one of the index is bigger than the corresponding dimension
     *             shape
     */
    private Object get(final IIndex index) {
        Object result = null;
        IndexNode ind = getIndexNode(index);
        IIndex itemIdx = ind.getIndex();

        int nodeIndex = ind.getNode();
        IArray slab = mArrays[nodeIndex];
        if( slab != null ) {
            result = slab.getObject(itemIdx);
        }
        return result;
    }

    private IndexNode getIndexNode(final IIndex index) {
        int nodeIndex;
        IIndex itemIdx;
        if( mArrays.length > 1 ) {
            DefaultCompositeIndex idx;
            if (!(index instanceof DefaultCompositeIndex)) {
                int rank = mIndex.getIndexMatrix().getRank();
                idx = new DefaultCompositeIndex(factoryName, rank, mIndex.getShape(),
                        index.getOrigin(), index.getShape());
                idx.set(index.getCurrentCounter());
            }
            else {
                idx = (DefaultCompositeIndex) index;
            }
            nodeIndex = (int) idx.currentElementMatrix();
            itemIdx = mIndex.getIndexStorage().clone();
            itemIdx.set(idx.getIndexStorage().getCurrentCounter());
        }
        else {
            nodeIndex = 0;
            itemIdx = mIndex.getIndexStorage().clone();
            itemIdx.set(index.getCurrentCounter());
        }
        return new IndexNode(itemIdx, nodeIndex);
    }

    private void set(final IIndex index, final Object value) {
        DefaultCompositeIndex idx = null;
        if (!(index instanceof DefaultCompositeIndex)) {
            idx = new DefaultCompositeIndex(factoryName, mIndex.getIndexMatrix().getRank(), index);
        }
        else {
            idx = (DefaultCompositeIndex) index;
        }

        DefaultIndex itemIdx = idx.getIndexStorage();
        long nodeIndex = idx.currentElementMatrix();
        IArray slab = mArrays[(int) nodeIndex];
        if( slab != null ) {
            slab.setObject(itemIdx, value);
        }
    }

    private class IndexNode {
        private final IIndex mIndex;
        private final int mNode;

        public IndexNode( final IIndex index, final int node ) {
            mIndex = index;
            mNode  = node;
        }

        public IIndex getIndex() {
            return mIndex;
        }

        public int getNode() {
            return mNode;
        }
    }

}
