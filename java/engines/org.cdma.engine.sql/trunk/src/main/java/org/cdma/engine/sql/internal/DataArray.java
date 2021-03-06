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
 *        St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.engine.sql.internal;

import java.util.Arrays;
import java.util.concurrent.Semaphore;

/**
 * Parameterized class that aims to ease the memory allocation for a SQL result
 * set. It allows to have a type erasure pattern and to deal with primitive
 * types.
 * 
 * @author rodriguez
 */
public class DataArray<T> {
    private final T mValue; // Data storage
    private final Class<T> mType; // Data type
    private final boolean mIsIntegerNumber;
    private int mPos;
    private final Semaphore mLock;

    @SuppressWarnings("unchecked")
    private DataArray(T value) {
        mValue = value;
        mType = (Class<T>) value.getClass();
        mPos = -1;
        // Try to determine if the type can be associated to an integer
        Class<?> clazz = mType;
        while (clazz.isArray()) {
            clazz = clazz.getComponentType();
        }
        mIsIntegerNumber = (clazz.equals(Integer.TYPE) || clazz.equals(Long.TYPE) || clazz.equals(Short.TYPE));
        mLock = new Semaphore(1);
    }

    /**
     * Getter on the memory storage
     * 
     * @return T object (array of primitive or an object instance)
     */
    public T getValue() {
        return mValue;
    }

    /**
     * The element type of the underlying memory storage
     * 
     * @return Class<T>
     */
    public Class<T> getType() {
        return mType;
    }

    public int getLastAppendPosition() {
        return mPos;
    }

    /**
     * Without it will return the first defined cell of the array when it has been loaded
     */
    public Object getSample() {
        Object result = null;
        boolean available = true;
        lock();
        if (available) {
            if (mValue.getClass().isArray()) {
                result = java.lang.reflect.Array.get(mValue, mPos);
            } else {
                result = mValue;
            }
        }
        unlock();
        return result;
    }

    // ---------------------------------------------------------
    // Private static methods
    // ---------------------------------------------------------
    @SuppressWarnings({ "unchecked" })
    static public <T> DataArray<?> allocate(T data, int nbRows) {
        DataArray<?> array;
        if (data == null) {
            if (nbRows > 0) {
                Object[] storage = new Object[nbRows];
                array = new DataArray<Object[]>(storage);
                array.lock();
            } else {
                array = new DataArray<Object>(data);
            }
        } else {
            if (nbRows > 0) {
                T[] storage = (T[]) java.lang.reflect.Array.newInstance(data.getClass(), nbRows);
                Arrays.fill(storage, data);
                array = new DataArray<T[]>(storage);
                array.lock();
            } else {
                array = new DataArray<T>(data);
            }
        }
        return array;
    }

    static public <T> DataArray<?> allocate(int data, int nbRows) {
        DataArray<int[]> array;
        if (nbRows > 0) {
            array = new DataArray<int[]>(new int[nbRows]);
            array.lock();
        } else {
            array = new DataArray<int[]>(new int[1]);
        }
        return array;
    }

    static public <T> DataArray<?> allocate(long data, int nbRows) {
        DataArray<long[]> array;
        if (nbRows > 0) {
            array = new DataArray<long[]>(new long[nbRows]);
            array.lock();
        } else {
            array = new DataArray<long[]>(new long[1]);
        }
        return array;
    }

    static public <T> DataArray<?> allocate(double data, int nbRows) {
        DataArray<double[]> array;
        if (nbRows > 0) {
            array = new DataArray<double[]>(new double[nbRows]);
            array.lock();
        } else {
            array = new DataArray<double[]>(new double[1]);
        }
        return array;
    }

    static public <T> DataArray<?> allocate(short data, int nbRows) {
        DataArray<short[]> array;
        if (nbRows > 0) {
            array = new DataArray<short[]>(new short[nbRows]);
            array.lock();
        } else {
            array = new DataArray<short[]>(new short[1]);
        }
        return array;
    }

    static public <T> DataArray<?> allocate(float data, int nbRows) {
        DataArray<float[]> array;
        if (nbRows > 0) {
            array = new DataArray<float[]>(new float[nbRows]);
            array.lock();
        } else {
            array = new DataArray<float[]>(new float[1]);
        }
        return array;
    }

    static public <T> DataArray<?> allocate(boolean data, int nbRows) {
        DataArray<boolean[]> array;
        if (nbRows > 0) {
            array = new DataArray<boolean[]>(new boolean[nbRows]);
            array.lock();
        } else {
            array = new DataArray<boolean[]>(new boolean[1]);
        }
        return array;
    }

    // ---------------------------------------------------------
    // Private methods
    // ---------------------------------------------------------
    @SuppressWarnings("unchecked")
    public <U> void setData(U data, int row) {
        try {
            if (mValue != null) {
                ((U[]) mValue)[row] = data;
                mPos = row;
                unlock();
            }
        } catch (ArrayStoreException e) {
            e.printStackTrace();
        }
    }

    public void setData(int data, int row) {
        if (mValue != null) {
            ((int[]) mValue)[row] = data;
            mPos = row;
            unlock();
        }
    }

    public void setData(long data, int row) {
        if (mValue != null) {
            ((long[]) mValue)[row] = data;
            mPos = row;
            unlock();
        }
    }

    public void setData(double data, int row) {
        if (mValue != null) {
            ((double[]) mValue)[row] = data;
            mPos = row;
            unlock();
        }
    }

    public void setData(short data, int row) {
        if (mValue != null) {
            ((short[]) mValue)[row] = data;
            mPos = row;
            unlock();
        }
    }

    public void setData(float data, int row) {
        if (mValue != null) {
            ((float[]) mValue)[row] = data;
            mPos = row;
            unlock();
        }
    }

    public void setData(boolean data, int row) {
        if (mValue != null) {
            ((boolean[]) mValue)[row] = data;
            mPos = row;
            unlock();
        }
    }

    protected boolean isIntegerNumber() {
        return mIsIntegerNumber;
    }

    protected void lock() {
        try {
            mLock.acquire();
        } catch (InterruptedException e) {
        }
    }

    protected void unlock() {
        mLock.release();
    }
}
