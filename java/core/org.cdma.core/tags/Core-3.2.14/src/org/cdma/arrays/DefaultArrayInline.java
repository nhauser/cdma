/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
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

import java.lang.ref.WeakReference;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.exception.BackupException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.ISliceIterator;

public class DefaultArrayInline extends DefaultArray {
    private Object mData; // Memory storage (always a java inline array)

    // Constructors
    protected DefaultArrayInline(String factoryName, Class<?> clazz, int[] iShape) throws InvalidArrayTypeException {
        this(factoryName, clazz, null, iShape.clone());
    }

    protected DefaultArrayInline(String factoryName, Object inlineArray, int[] iShape) throws InvalidArrayTypeException {
        this(factoryName, inlineArray.getClass().getComponentType(), inlineArray, iShape.clone());
    }

    protected DefaultArrayInline(String factoryName, WeakReference<Object> inlineArrayReference, int[] iShape)
            throws InvalidArrayTypeException {
        this(factoryName, inlineArrayReference.get().getClass().getComponentType(), inlineArrayReference, iShape
                .clone());
    }

    protected DefaultArrayInline(DefaultArrayInline array) throws InvalidArrayTypeException {
        super(array);
        mData = array.mData;
    }

    protected DefaultArrayInline(String factoryName, Class<?> clazz, Object inlineArray, int[] iShape)
            throws InvalidArrayTypeException {
        super(factoryName, clazz, iShape.clone());

        if (clazz == null || clazz.isArray()) {
            throw new InvalidArrayTypeException("Only inline array is permitted!");
        }

        mData = inlineArray;
    }

    // ---------------------------------------------------------
    // / public methods
    // ---------------------------------------------------------
    // / IArray underlying data access
    @Override
    public Object getStorage() {
        if (!isLocked()) {
            mData = loadData();
        }
        return getData();
    }

    @Override
    public void releaseStorage() throws BackupException {
        Factory.getLogger().log(Level.WARNING, "Unable to release storage", new NotImplementedException());
    }

    // IArray data getters and setters
    @Override
    public boolean getBoolean(IIndex index) {
        boolean result;
        IIndex idx = getIndex().clone();
        idx.set(index.getCurrentCounter());
        Object oData = getStorage();

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
        return result;
    }

    @Override
    public byte getByte(IIndex index) {
        byte result;
        IIndex idx;
        idx = getIndex().clone();
        idx.set(index.getCurrentCounter());
        Object oData = getStorage();

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
        return result;
    }

    @Override
    public char getChar(IIndex index) {
        char result;
        IIndex idx;
        idx = getIndex().clone();
        idx.set(index.getCurrentCounter());
        Object oData = getStorage();

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
        return result;
    }

    @Override
    public double getDouble(IIndex index) {
        double result;
        IIndex idx = getIndex().clone();
        idx.set(index.getCurrentCounter());
        Object oData = getStorage();
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
        return result;
    }

    @Override
    public float getFloat(IIndex index) {
        float result = 0;
        IIndex idx = getIndex().clone();
        idx.set(index.getCurrentCounter());
        Object oData = getStorage();

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
        return result;
    }

    @Override
    public int getInt(IIndex index) {
        int result = 0;
        IIndex idx = getIndex().clone();
        idx.set(index.getCurrentCounter());
        Object oData = getStorage();

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
        return result;
    }

    @Override
    public long getLong(IIndex index) {
        long result = 0;
        IIndex idx = getIndex().clone();
        idx.set(index.getCurrentCounter());
        Object oData = getStorage();

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
        return result;
    }

    @Override
    public Object getObject(IIndex index) {
        Object result = new Object();
        IIndex idx = getIndex().clone();
        idx.set(index.getCurrentCounter());
        Object oData = getStorage();

        // If it's a scalar value then we return it
        if (!oData.getClass().isArray()) {
            result = oData;
        }
        // else it's a single raw array, then we compute indexes to have the
        // corresponding cell number
        else {
            int lPos = (int) idx.currentElement();
            result = java.lang.reflect.Array.get(oData, lPos);
        }
        return result;
    }

    @Override
    public short getShort(IIndex index) {
        short result = 0;
        IIndex idx = getIndex().clone();
        idx.set(index.getCurrentCounter());
        Object oData = getStorage();

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
        return result;
    }

    @Override
    public void setBoolean(IIndex index, boolean value) {
        Object data = getStorage();
        IIndex idx = getIndex().clone();
        int pos = (int) idx.currentElement();
        if (getElementType().equals(Boolean.TYPE)) {
            ((boolean[]) data)[pos] = value;
        } else {
            java.lang.reflect.Array.set(data, pos, value);
        }
        setDirty(true);
    }

    @Override
    public void setByte(IIndex index, byte value) {
        Object data = getStorage();
        IIndex idx = getIndex().clone();
        int pos = (int) idx.currentElement();
        if (getElementType().equals(Boolean.TYPE)) {
            ((byte[]) data)[pos] = value;
        } else {
            java.lang.reflect.Array.set(data, pos, value);
        }
        setDirty(true);
    }

    @Override
    public void setChar(IIndex index, char value) {
        Object data = getStorage();
        IIndex idx = getIndex().clone();
        int pos = (int) idx.currentElement();
        if (getElementType().equals(Boolean.TYPE)) {
            ((char[]) data)[pos] = value;
        } else {
            java.lang.reflect.Array.set(data, pos, value);
        }
        setDirty(true);
    }

    @Override
    public void setDouble(IIndex index, double value) {
        Object data = getStorage();
        IIndex idx = getIndex().clone();
        int pos = (int) idx.currentElement();
        if (getElementType().equals(Boolean.TYPE)) {
            ((double[]) data)[pos] = value;
        } else {
            java.lang.reflect.Array.set(data, pos, value);
        }
        setDirty(true);
    }

    @Override
    public void setFloat(IIndex index, float value) {
        Object data = getStorage();
        IIndex idx = getIndex().clone();
        int pos = (int) idx.currentElement();
        if (getElementType().equals(Boolean.TYPE)) {
            ((float[]) data)[pos] = value;
        } else {
            java.lang.reflect.Array.set(data, pos, value);
        }
        setDirty(true);
    }

    @Override
    public void setInt(IIndex index, int value) {
        Object data = getStorage();
        IIndex idx = getIndex().clone();
        int pos = (int) idx.currentElement();
        if (getElementType().equals(Boolean.TYPE)) {
            ((int[]) data)[pos] = value;
        } else {
            java.lang.reflect.Array.set(data, pos, value);
        }
        setDirty(true);
    }

    @Override
    public void setLong(IIndex index, long value) {
        Object data = getStorage();
        IIndex idx = getIndex().clone();
        int pos = (int) idx.currentElement();
        if (getElementType().equals(Boolean.TYPE)) {
            ((long[]) data)[pos] = value;
        } else {
            java.lang.reflect.Array.set(data, pos, value);
        }
        setDirty(true);
    }

    @Override
    public void setObject(IIndex index, Object value) {
        Object data = getStorage();
        IIndex idx = getIndex().clone();
        int pos = (int) idx.currentElement();
        if (getElementType().equals(Boolean.TYPE)) {
            ((Object[]) data)[pos] = value;
        } else {
            java.lang.reflect.Array.set(data, pos, value);
        }
        setDirty(true);
    }

    @Override
    public void setShort(IIndex index, short value) {
        Object data = getStorage();
        IIndex idx = getIndex().clone();
        int pos = (int) idx.currentElement();
        if (getElementType().equals(Boolean.TYPE)) {
            ((short[]) data)[pos] = value;
        } else {
            java.lang.reflect.Array.set(data, pos, value);
        }
        setDirty(true);
    }

    @Override
    public IArray setDouble(double value) {
        IIndex index = getIndex();
        Object data = getStorage();
        if (getElementType().equals(Double.TYPE)) {
            java.util.Arrays.fill((double[]) data, value);
        } else {
            for (int i = 0; i < index.getSize(); i++) {
                java.lang.reflect.Array.set(data, i, value);
            }
        }
        setDirty(true);
        return this;
    }

    @Override
    public IArray copy(boolean data) {
        IArray result = null;
        IFactory factory = Factory.getFactory(getFactoryName());
        if (factory != null) {
            if (!data) {
                result = factory.createArray(getElementType(), getShape(), mData);
            } else {
                Object store = this.copyTo1DJavaArray();
                result = factory.createArray(getElementType(), getShape(), store);
            }
        }

        return result;
    }

    // ---------------------------------------------------------
    // / Protected methods
    // ---------------------------------------------------------
    protected IArray sectionNoReduce(int[] origin, int[] shape, long[] stride) throws ShapeNotMatchException {
        DefaultArray array = (DefaultArray) copy(false);
        IIndex index = getIndex();
        index.setShape(shape);
        index.setStride(stride);
        index.setOrigin(origin);
        array.setIndex(index);
        return array;
    }

    @Override
    protected Object copyTo1DJavaArray() {
        // Instantiate a new convenient array for storage
        int length = ((Long) this.getSize()).intValue();
        Class<?> type = getElementType();
        Object array = java.lang.reflect.Array.newInstance(type, length);

        // Calculate the starting point of the visible part of the array's storage
        IIndex index = getIndex().clone();
        int[] position = new int[index.getRank()];
        index.set(position);
        int start = ((Long) index.currentElement()).intValue();

        // Copy the visible part of the array
        System.arraycopy(getStorage(), start, array, 0, length);

        return array;
    }

    @Override
    protected Object copyToNDJavaArray() {
        int[] shape = getShape();
        int[] current;
        int length;
        int startCell;
        Object slab;

        // Create an empty array of the right shape
        Object result = java.lang.reflect.Array.newInstance(getElementType(), shape);

        ISliceIterator iter;
        try {
            iter = getSliceIterator(1);
            length = shape[shape.length - 1];

            DefaultIndex startIdx = (DefaultIndex) getIndex().clone();
            Object values = this.getStorage();
            while (iter.hasNext()) {
                // Increment the slice iterator
                iter.next();
                slab = result;

                // Get the right slab in the multi-dimensional resulting array
                current = iter.getSlicePosition();
                startIdx.set(current);
                for (int pos = 0; pos < current.length - 1; pos++) {
                    slab = java.lang.reflect.Array.get(slab, current[pos]);
                }

                // Get the starting offset
                startCell = startIdx.currentProjectionElement();

                // Copy the array
                System.arraycopy(values, startCell, slab, 0, length);
            }
        } catch (ShapeNotMatchException e) {
            result = null;
        } catch (InvalidRangeException e) {
            result = null;
        }
        return result;
    }

    /**
     * Override this method in case of specific need (use of SoftReference for instance).
     * It is called each time the memory is accessed when (the array isn't locked).
     * 
     * The storage of this array, will be replaced by the object returned by this method.
     * 
     * @return the backing storage (as it is) of the array
     */
    @Override
    protected Object loadData() {
        return mData;
    }

    /**
     * Replace the backing storage by the given one.
     * 
     * @param data to set in place of the current one
     * @note no index calculation or type detection is done, the given data should
     *       have the same properties as the current one.
     */
    protected void setData(Object data) {
        mData = data;
    }

    /**
     * Returns a direct access on the underlying data.
     * 
     * @return the backing storage of the array (as it is)
     * @note override this method to prepare the data
     */
    @Override
    protected Object getData() {
        return mData;
    }
}
