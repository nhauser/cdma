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
// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr) - API evolution
// ****************************************************************************
package org.cdma.interfaces;

import org.cdma.exception.BackupException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.internal.IModelObject;
import org.cdma.math.IArrayMath;
import org.cdma.utils.IArrayUtils;

/**
 * @brief The IArray interface manages multiple types of data.
 */

/// @cond pluginAPI

/**
 * @note When developing a plug-in consider using the data format engine's implementation. 
 * You should <b>redefine this interface implementation</b>, only in case of <b>very specific needs.</b>
 * <p>
 */

/// @endcond pluginAPI

/**
 * A IArray can be a matrix or a single scalar it carries the underlying data
 * and provide a standardized way to access it.<br/>
 * An array has:
 *    - a <b>type info</b> which gives the type of its elements,<br/>
 *    - a <b>shape</b> which describes the number of elements in each dimension<br/>
 *    - a <b>size</b> that is the total number of elements<br/>
 *    - a <b>rank</b> that is the number of dimension<br/>
 * A <b>scalar</b> IArray has a rank = 0, an IArray may have arbitrary rank.
 * <p>
 * The data storage is done by the plug-in as stride index
 * calculations. However a 1D array of object should always be a possible storage
 * for those data. This makes our IArray rectangular, i.e. no "ragged arrays"
 * where different elements can have different lengths as multidimensional 
 * arrays, which are arrays of arrays.
 * <p>
 * For efficiency, each IArray should use primitive Java type (boolean, byte, 
 * char, int, float, ...) for the underlying storage.
 * <p>
 * The stride index calculations allows <b>logical views</b> (IIndex) of the 
 * underlying data. To be efficiently implemented, eg subset, transpose, slice,
 * etc. Those views use the same data storage as the original array they are 
 * derived from. The index stride calculations are equally efficient for any
 * chain of logical views.
 * <p>
 * The type, shape and backing storage of an IArray are immutable. The data
 * itself is read or written using a IIndex or a IArrayIterator, which stores
 * any needed state information for efficient traversal. This makes use of
 * Arrays thread-safe (as long as you don't share the IIndex or IArrayIterator)
 * except for the possibility of non-atomic read/write on long/doubles. If this
 * is the case, you should probably synchronize your calls.
 *
 * @see org.cdma.interfaces.IIndex
 * 
 * @author nxi
 */

public interface IArray extends IModelObject {

    /**
     * Create a copy of this Array, copying the data so that physical order is
     * the same as logical order.
     * 
     * @return the new IArray
     */
    IArray copy();

    /**
     * Create a copy of this IArray. Whether to copy the data so that physical order is
     * the same as logical order, it will share it. So both arrays reference the same backing storage.
     * 
     * @param data if true the backing storage will be copied too, else it will be shared 
     * @return the new IArray
     */
    IArray copy(boolean data);

    /**
     * Get an IArrayUtils that permits shape manipulations on arrays
     * 
     * @return new IArrayUtils object
     */
    IArrayUtils getArrayUtils();

    /**
     * Get an IArrayMath that permits math calculations on arrays
     * 
     * @return new IArrayMath object
     */
    IArrayMath getArrayMath();

    /**
     * Get the array element at the current element offset of index, as a boolean.
     * 
     * @param index IIndex with current element set
     * @return value at <code>index</code> cast to boolean if necessary.
     */
    boolean getBoolean(IIndex index);

    /**
     * Get the array element at the current element offset of index, as a byte.
     * 
     * @param index IIndex with current element set
     * @return value at <code>index</code> cast to float if necessary.
     */
    byte getByte(IIndex index);

    /**
     * Get the array element at the current element offset of index, as a char.
     * 
     * @param index IIndex with current element set
     * @return value at <code>index</code> cast to char if necessary.
     */
    char getChar(IIndex index);

    /**
     * Get the array element at the current element offset of index, as a double.
     * 
     * @param index IIndex with current element set
     * @return value at <code>index</code> cast to double if necessary.
     */
    double getDouble(IIndex index);

    /**
     * Get the element class type of this IArray.
     * 
     * @return Class object
     */
    Class<?> getElementType();

    /**
     * Get the array element at the current element offset of index, as a float.
     * 
     * @param index IIndex with current element set
     * @return value at <code>index</code> cast to float if necessary.
     */
    float getFloat(IIndex index);

    /**
     * Get an IIndex object used for indexed access of this IArray.
     * 
     * @return the IIndex object currently used by the IArray 
     * @see IIndex
     */
    IIndex getIndex();

    /**
     * Get the array element at the current element offset of index, as a int.
     * 
     * @param index IIndex with current element set
     * @return value at <code>index</code> cast to int if necessary.
     */
    int getInt(IIndex index);

    /**
     * Get Iterator to traverse the IArray.
     * 
     * @return ArrayIterator
     */
    IArrayIterator getIterator();

    /**
     * Get the array element at the current element offset of index, as a long.
     * 
     * @param index IIndex with current element set
     * @return value at <code>index</code> cast to long if necessary.
     */
    long getLong(IIndex index);

    /**
     * Get the array element at index as an Object. The returned value is
     * wrapped in an object, eg Double for double
     * 
     * @param index element Index
     * @return Object value at <code>index</code>
     */
    Object getObject(IIndex index);

    /**
     * Get the number of dimensions of the array.
     * 
     * @return number of dimensions of the array
     */
    int getRank();

    /**
     * Get an iterator over a region of the IArray. The region is
     * described by the reference and range parameters.
     * 
     * @param reference integer array of starting position for each dimension 
     * @param range integer array of length for each dimension
     * @return IArrayIterator
     * @throws InvalidRangeException
     */
    IArrayIterator getRegionIterator(int[] reference, int[] range) throws InvalidRangeException;

    /**
     * Get the shape: length of array in each dimension.
     * 
     * @return array whose length is the rank of this IArray and whose elements
     *         represent the length of each of its indices.
     */
    int[] getShape();

    /**
     * Get the array element at the current element offset of index, as a short.
     * 
     * @param index IIndex with current element set
     * @return value at <code>index</code> cast to short if necessary.
     */
    short getShort(IIndex index);

    /**
     * Get the total number of elements in the array.
     * 
     * @return total number of elements in the array
     */
    long getSize();

    /**
     * Get the underlying primitive array storage. Exposed for efficiency, use at
     * your own risk.
     * 
     * @return Object that is plug-in and format dependent
     */
    Object getStorage();

    /**
     * Set the array element at the current element offset of index.
     * 
     * @param index IIndex with current element set
     * @param value the new value; cast to underlying data type if necessary.
     */
    void setBoolean(IIndex index, boolean value);

    /**
     * Set the array element at the current element offset of index.
     * 
     * @param index IIndex with current element set
     * @param value the new value; cast to underlying data type if necessary.
     */
    void setByte(IIndex index, byte value);

    /**
     * Set the array element at the current element offset of index.
     * 
     * @param index IIndex with current element set
     * @param value the new value; cast to underlying data type if necessary.
     */
    void setChar(IIndex index, char value);

    /**
     * Set the array element at the current element offset of index.
     * 
     * @param index IIndex with current element set
     * @param value the new value; cast to underlying data type if necessary.
     */
    void setDouble(IIndex index, double value);

    /**
     * Set the array element at the current element offset of index.
     * 
     * @param index IIndex with current element set
     * @param value the new value; cast to underlying data type if necessary.
     */
    void setFloat(IIndex index, float value);

    /**
     * Set the array element at the current element offset of index.
     * 
     * @param index IIndex with current element set
     * @param value the new value; cast to underlying data type if necessary.
     */
    void setInt(IIndex index, int value);

    /**
     * Set the array element at the current element offset of index.
     * 
     * @param index IIndex with current element set
     * @param value the new value; cast to underlying data type if necessary.
     */
    void setLong(IIndex index, long value);

    /**
     * Set the array element at index to the specified value. the value must be
     * passed wrapped in the appropriate Object (eg Double for double)
     * 
     * @param index IIndex with current element set
     * @param value the new value.
     */
    void setObject(IIndex index, Object value);

    /**
     * Set the array element at the current element offset of index.
     * 
     * @param index IIndex with current element set
     * @param value the new value; cast to underlying data type if necessary.
     */
    void setShort(IIndex index, short value);

    /**
     * Convert the shape information to String type.
     * 
     * @return String type 
     */
    String shapeToString();

    /**
     * Set the given index as the current one for this array. Defines a viewable
     * part of this array.
     * 
     * @param index of the viewable part
     */
    void setIndex(IIndex index);

    /**
     * Get a slice iterator with certain rank. The rank of the slice must be
     * equal or smaller than the array itself. Otherwise throw
     * ShapeNotMatchException. <br>
     * For example, for an array with the shape of [2x3x4x5]. If the rank of the
     * slice is 1, there will be 2x3x4=24 slices of 5 elements each. If the rank 
     * of slice is 2, there will be 2x3=6 slices. If the rank of the slice is 3, 
     * there will be 2 slices. If the rank of slice is 4, which is not recommended,
     * there will be just 1 slices. If the rank of slice is 0, in which case it is pretty
     * costly, there will be 120 slices of 1 element each.
     * 
     * @param rank an integer value, this will be the rank of the slice
     * @return SliceIterator object
     * @throws ShapeNotMatchException mismatching shape
     * @throws InvalidRangeException invalid range 
     */
    ISliceIterator getSliceIterator(int rank) throws ShapeNotMatchException,
    InvalidRangeException;

    /**
     * Release the back storage of this IArray. It will trigger backup routine,
     * which saves the data into the file system that can be load back when this
     * IArray is accessed next time.
     * 
     * 
     * @throws BackupException failed to put in storage
     */
    void releaseStorage() throws BackupException;

    /**
     * Get the register ID of the array.
     * 
     * @return long value 
     */
    long getRegisterId();

    /**
     * Lock the array from loading data from backup storage. If the data is not
     * backed up, this will not affecting reading out the data.
     */
    void lock();

    /**
     * Release the lock of the array from loading data from backup storage.
     */
    void unlock();

    /**
     * Return true if the array has been changed since last read out from the 
     * backup storage.
     */
    boolean isDirty();

    /**
     * Set the array to indicate changes since last read out from the backup
     * storage.
     */
    void setDirty(boolean dirty);

    /**
     * Set double value to all values of the IArray.
     * 
     * @param value double value
     * @return this
     */
    IArray setDouble(final double value);

}
