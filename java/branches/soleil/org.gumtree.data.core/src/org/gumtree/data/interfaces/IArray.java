/*******************************************************************************
 * Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 *    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 ******************************************************************************/
package org.gumtree.data.interfaces;

import org.gumtree.data.exception.BackupException;
import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.math.IArrayMath;
import org.gumtree.data.utils.IArrayUtils;

/**
 * Array for multiple type. An Array has a <b>classType</b> which gives the
 * Class of its elements, and a <b>shape</b> which describes the number of
 * elements in each index. The <b>rank</b> is the number of indices. A
 * <b>scalar</b> Array has rank = 0. An Array may have arbitrary rank. The Array
 * <b>size</b> is the total number of elements, which must be less than 2^31
 * (about 2x10^9).
 * <p>
 * Actual data storage is done with Java 1D arrays and stride index
 * calculations. This makes our Arrays rectangular, i.e. no "ragged arrays"
 * where different elements can have different lengths as in Java
 * multidimensional arrays, which are arrays of arrays.
 * <p>
 * Each primitive Java type (boolean, byte, char, short, int, long, float,
 * double) has a corresponding concrete implementation, e.g. ArrayBoolean,
 * ArrayDouble. Reference types are all implemented using the ArrayObject class,
 * with the exceptions of the reference types that correspond to the primitive
 * types, eg Double.class is mapped to double.class.
 * <p>
 * For efficiency, each Array type implementation has concrete subclasses for
 * ranks 0-7, eg ArrayDouble.D0 is a double array of rank 0, ArrayDouble.D1 is a
 * double array of rank 1, etc. These type and rank specific classes are
 * convenient to work with when you know the type and rank of the Array. Ranks
 * greater than 7 are handled by the type-specific superclass e.g. ArrayDouble.
 * The Array class itself is used for fully general handling of any type and
 * rank array. Use the Array.factory() methods to create Arrays in a general
 * way.
 * <p>
 * The stride index calculations allow <b>logical views</b> to be efficiently
 * implemented, eg subset, transpose, slice, etc. These views use the same data
 * storage as the original Array they are derived from. The index stride
 * calculations are equally efficient for any chain of logical views.
 * <p>
 * The type, shape and backing storage of an Array are immutable. The data
 * itself is read or written using an Index or an IndexIterator, which stores
 * any needed state information for efficient traversal. This makes use of
 * Arrays thread-safe (as long as you dont share the Index or IndexIterator)
 * except for the possibility of non-atomic read/write on long/doubles. If this
 * is the case, you should probably synchronize your calls. Presumably 64-bit
 * CPUs will make those operations atomic also.
 * 
 * @author nxi
 */
public interface IArray extends IModelObject {

  /**
   * Create a copy of this Array, copying the data so that physical order is
   * the same as logical order.
   * 
   * @return the new Array
   */
  IArray copy();

  /**
   * Create a copy of this Array. Whether to copy the data so that physical order is
   * the same as logical order, it will share it. So both arrays reference the same backing storage.
   * 
   * @param data if true the backing storage will be copied too, else it will be shared 
   * @return the new Array
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
   * Get the array element at the current element of ima, as a boolean.
   * 
   * @param ima Index with current element set
   * @return value at <code>index</code> cast to boolean if necessary.
   */
  boolean getBoolean(IIndex ima);

  /**
   * Get the array element at the current element of ima, as a byte.
   * 
   * @param ima Index with current element set
   * @return value at <code>index</code> cast to float if necessary.
   */
  byte getByte(IIndex ima);

  /**
   * Get the array element at the current element of ima, as a char.
   * 
   * @param ima Index with current element set
   * @return value at <code>index</code> cast to char if necessary.
   */
  char getChar(IIndex ima);

  /**
   * Get the array element at the current element of ima, as a double.
   * 
   * @param ima Index with current element set
   * @return value at <code>index</code> cast to double if necessary.
   */
  double getDouble(IIndex ima);

  /**
   * Get the element class type of this Array.
   * 
   * @return Class object
   */
  Class<?> getElementType();

  /**
   * Get the array element at the current element of ima, as a float.
   * 
   * @param ima Index with current element set
   * @return value at <code>index</code> cast to float if necessary.
   */
  float getFloat(IIndex ima);

  /**
   * Get an Index object used for indexed access of this Array.
   * 
   * @return IIndex object
   * @see IIndex
   */
  IIndex getIndex();

  /**
   * Get the array element at the current element of ima, as a int.
   * 
   * @param ima Index with current element set
   * @return value at <code>index</code> cast to int if necessary.
   */
  int getInt(IIndex ima);

  /**
   * Get Iterator to traverse the Array.
   * 
   * @return ArrayIterator
   */
  IArrayIterator getIterator();

  /**
   * Get the array element at the current element of ima, as a long.
   * 
   * @param ima Index with current element set
   * @return value at <code>index</code> cast to long if necessary.
   */
  long getLong(IIndex ima);

  /**
   * Get the array element at index as an Object. The returned value is
   * wrapped in an object, eg Double for double
   * 
   * @param ima element Index
   * @return Object value at <code>index</code>
   */
  Object getObject(IIndex ima);

  /**
   * Get the number of dimensions of the array.
   * 
   * @return number of dimensions of the array
   */
  int getRank();

  /**
   * Get the iterator that only iterate a region of the Array. The region is
   * described by the reference and range parameters.
   * 
   * @param reference java array of integer
   * @param range java array of integer
   * @return ArrayIterator
   * @throws InvalidRangeException
   */
  IArrayIterator getRegionIterator(int[] reference, int[] range)
      throws InvalidRangeException;

  /**
   * Get the shape: length of array in each dimension.
   * 
   * @return array whose length is the rank of this Array and whose elements
   *         represent the length of each of its indices.
   */
  int[] getShape();

  /**
   * Get the array element at the current element of ima, as a short.
   * 
   * @param ima Index with current element set
   * @return value at <code>index</code> cast to short if necessary.
   */
  short getShort(IIndex ima);

  /**
   * Get the total number of elements in the array.
   * 
   * @return total number of elements in the array
   */
  long getSize();

  /**
   * Get underlying primitive array storage. Exposed for efficiency, use at
   * your own risk.
   * 
   * @return any Object
   */
  Object getStorage();

  /**
   * Set the array element at the current element of ima.
   * 
   * @param ima Index with current element set
   * @param value the new value; cast to underlying data type if necessary.
   */
  void setBoolean(IIndex ima, boolean value);

  /**
   * Set the array element at the current element of ima.
   * 
   * @param ima Index with current element set
   * @param value the new value; cast to underlying data type if necessary.
   */
  void setByte(IIndex ima, byte value);

  /**
   * Set the array element at the current element of ima.
   * 
   * @param ima Index with current element set
   * @param value the new value; cast to underlying data type if necessary.
   */
  void setChar(IIndex ima, char value);

  /**
   * Set the array element at the current element of ima.
   * 
   * @param ima Index with current element set
   * @param value the new value; cast to underlying data type if necessary.
   */
  void setDouble(IIndex ima, double value);

  /**
   * Set the array element at the current element of ima.
   * 
   * @param ima Index with current element set
   * @param value the new value; cast to underlying data type if necessary.
   */
  void setFloat(IIndex ima, float value);

  /**
   * Set the array element at the current element of ima.
   * 
   * @param ima Index with current element set
   * @param value the new value; cast to underlying data type if necessary.
   */
  void setInt(IIndex ima, int value);

  /**
   * Set the array element at the current element of ima.
   * 
   * @param ima Index with current element set
   * @param value the new value; cast to underlying data type if necessary.
   */
  void setLong(IIndex ima, long value);

  /**
   * Set the array element at index to the specified value. the value must be
   * passed wrapped in the appropriate Object (eg Double for double)
   * 
   * @param ima Index with current element set
   * @param value the new value.
   */
  void setObject(IIndex ima, Object value);

  /**
   * Set the array element at the current element of ima.
   * 
   * @param ima Index with current element set
   * @param value the new value; cast to underlying data type if necessary.
   */
  void setShort(IIndex ima, short value);

  /**
   * Convert the shape information to String type.
   * 
   * @return String type 
   */
  String shapeToString();

  /**
   * Set the given index as current one for this array. Defines a viewable
   * part of this array.
   * 
   * @param index of the viewable part
   */
  void setIndex(IIndex index);

  /**
   * Get the slice iterator with certain rank. The rank of the slice must be
   * equal or smaller than the array itself. Otherwise throw
   * ShapeNotMatchException. <br>
   * For example, for an array with the shape of [2x3x4x5]. If the rank of the
   * slice is 1, there will be 2x3x4=24 slices. If the rank of slice is 2,
   * there will be 2x3=6 slices. If the rank of the slice is 3, there will be
   * 2 slices. if the rank of slice is 4, which is not recommended, there will
   * be just 1 slices. If the rank of slice is 0, in which case it is pretty
   * costly, there will be 120 slices.
   * 
   * @param rank an integer value, this will be the rank of the slice
   * @return SliceIterator object
   * @throws ShapeNotMatchException
   *             mismatching shape
   * @throws InvalidRangeException
   *             invalid range 
   */
  ISliceIterator getSliceIterator(int rank) throws ShapeNotMatchException,
      InvalidRangeException;

  /**
   * Release the back storage of this Array. It will trigger backup routine,
   * which saves the data into the file system that can be load back when this
   * Array is accessed next time.
   * 
   * 
   * @throws BackupException
   *             failed to put in storage
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
   * backed up, this will not affecting reading out the data. Created on
   * 05/03/2009
   */
  void lock();

  /**
   * Release the lock of the array from loading data from backup storage.
   * 
   */
  void unlock();

  /**
   * If the array has been changed since last read out from the backup
   * storage.
   * 
   * @return 
   */
  boolean isDirty();

  /**
   * Set the array to indicate changes since last read out from the backup
   * storage.
   * 
   * @return 
   */
  void setDirty(boolean dirty);

  /**
   * Set double value to all values of the Array.
   * 
   * @param value double value
   * @return this
   */
  IArray setDouble(final double value);

}
