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

import java.io.IOException;
import java.util.List;

import org.gumtree.data.exception.DimensionNotSupportedException;
import org.gumtree.data.exception.InvalidArrayTypeException;
import org.gumtree.data.exception.InvalidRangeException;

/**
 * A DataItem is a logical container for data. It has a DataType, a set of 
 * Dimensions that define its array shape, and optionally a set of Attributes.
 * @author nxi
 * 
 */
public interface IDataItem extends IContainer {

	/**
	 * Find an Attribute by name, ignoring the case.
	 * 
	 * @param name
	 *            the name of the attribute
	 * @return the attribute, or null if not found
	 */
	IAttribute findAttributeIgnoreCase(String name);

	/**
	 * Find the index of the named Dimension in this DataItem.
	 * 
	 * @param name
	 *            the name of the dimension
	 * @return the index of the named Dimension, or -1 if not found.
	 */
	int findDimensionIndex(String name);

	/**
	 * Create a new DataItem that is a logical slice of this DataItem, by fixing
	 * the specified dimension at the specified index value. This reduces rank
	 * by 1. No data is read until a read method is called on it.
	 * 
	 * @param dimension
	 *            which dimension to fix
	 * @param value
	 *            at what index value
	 * @return a new DataItem which is a logical slice of this DataItem.
	 * @throws InvalidRangeException
	 *             Created on 13/03/2008
	 */
	IDataItem getASlice(int dimension, int value) throws InvalidRangeException;

	/**
	 * Get its parent Group, or null if its the root group.
	 * 
	 * @return GDM group object
	 */
	@Override
	IGroup getParentGroup();

	/**
	 * Get the root group of the tree that holds the current Group.
	 * 
	 * @return GDM Group Created on 18/06/2008
	 */
	@Override
	IGroup getRootGroup();

	/**
	 * Read all the data for this DataItem and return a memory resident Array.
	 * The Array has the same element type and shape as the DataItem.
	 * <p>
	 * If the DataItem is a member of an array of Structures, this returns only
	 * the variable's data in the first Structure, so that the Array shape is
	 * the same as the DataItem. To read the data in all structures, use
	 * readAllStructures().
	 * 
	 * @return the requested data in a memory-resident Array.
	 * @throws IOException
	 *             I/O exception
	 */
	IArray getData() throws IOException;

	/**
	 * Read a section of the data for this DataItem and return a memory resident
	 * Array. The Array has the same element type as the DataItem. The size of
	 * the Array will be either smaller or equal to the DataItem.
	 * <p>
	 * If the DataItem is a member of an array of Structures, this returns only
	 * the variable's data in the first Structure, so that the Array shape is
	 * the same as the DataItem. To read the data in all structures, use
	 * readAllStructures().
	 * 
	 * @param origin
	 *            array of int
	 * @param shape
	 *            array of int
	 * @return the requested data in a memory-resident Array.
	 * @throws IOException
	 *             I/O exception
	 * @throws InvalidRangeException
	 *             invalid range
	 */
	IArray getData(int[] origin, int[] shape) throws IOException,
			InvalidRangeException;

	/**
	 * Get the description of the DataItem. Default is to use "long_name"
	 * attribute value. If not exist, look for "description", "title", or
	 * "standard_name" attribute value (in that order).
	 * 
	 * @return description, or null if not found.
	 */
	String getDescription();

	/**
	 * Get the ith dimensions (if several are available return a populated corresponding list).
	 * 
	 * @param i
	 *            index of the dimension.
	 * @return requested Dimensions, or null if i is out of bounds.
	 */
	List<IDimension> getDimensions(int i);
	
	/**
	 * Get the list of all dimensions used by this variable. The most slowly varying
	 * (leftmost for Java and C programmers) dimension is first. For scalar
	 * variables, the list is empty.
	 * 
	 * @return List with objects of type ucar.nc2.Dimension
	 */
	List<IDimension> getDimensionList();

	/**
	 * Get the list of Dimension names, space delineated.
	 * 
	 * @return String object
	 */
	String getDimensionsString();

	/**
	 * Get the number of bytes for one element of this DataItem. For DataItems
	 * of primitive type, this is equal to getDataType().getSize(). DataItems of
	 * String type does not know their size, so what they return is undefined.
	 * DataItems of Structure type return the total number of bytes for all the
	 * members of one Structure, plus possibly some extra padding, depending on
	 * the underlying format. DataItems of Sequence type return the number of
	 * bytes of one element.
	 * 
	 * @return total number of bytes for the DataItem
	 */
	int getElementSize();

	/**
	 * display name plus the dimensions.
	 * 
	 * @return String object
	 */
	String getNameAndDimensions();

	/**
	 * display name plus the dimensions.
	 * 
	 * @param buf
	 *            StringBuffer object
	 * @param useFullName
	 *            true or false value
	 * @param showDimLength
	 *            true or false value
	 */
	void getNameAndDimensions(StringBuffer buf, boolean useFullName,
			boolean showDimLength);

	/**
	 * Get shape as an array of Range objects.
	 * 
	 * @return array of Ranges, one for each Dimension.
	 */
	List<IRange> getRangeList();

	/**
	 * Get the number of dimensions of the DataItem.
	 * 
	 * @return integer value
	 */
	int getRank();

	/**
	 * Create a new DataItem that is a logical subsection of this DataItem. No
	 * data is read until a read method is called on it.
	 * 
	 * @param section
	 *            List of type Range, with size equal to getRank(). Each Range
	 *            corresponds to a Dimension, and specifies the section of data
	 *            to read in that Dimension. A Range object may be null, which
	 *            means use the entire dimension.
	 * @return a new DataItem which is a logical section of this DataItem.
	 * @throws InvalidRangeException
	 *             invalid range
	 */
	IDataItem getSection(List<IRange> section) throws InvalidRangeException;

	/**
	 * Get index subsection as an array of Range objects, relative to the
	 * original variable. If this is a section, will reflect the index range
	 * relative to the original variable. If its a slice, it will have a rank
	 * different from this variable. Otherwise it will correspond to this
	 * DataItem's shape, ie match getRanges().
	 * 
	 * @return array of Ranges, one for each Dimension.
	 */
	List<IRange> getSectionRanges();

	/**
	 * Get the shape: length of DataItem in each dimension.
	 * 
	 * @return int array whose length is the rank of this and whose values equal
	 *         the length of that Dimension.
	 */
	int[] getShape();

	/**
	 * Get the total number of elements in the DataItem. If this is an unlimited
	 * DataItem, will return the current number of elements. If this is a
	 * Sequence, will return 0.
	 * 
	 * @return total number of elements in the DataItem.
	 */
	long getSize();

	/**
	 * If total data is less than SizeToCache in bytes, then cache.
	 * 
	 * @return integer value
	 */
	int getSizeToCache();

	/**
	 * Create a new DataItem that is a logical slice of this DataItem, by fixing
	 * the specified dimension at the specified index value. This reduces rank
	 * by 1. No data is read until a read method is called on it.
	 * 
	 * @param dim
	 *            which dimension to fix
	 * @param value
	 *            at what index value
	 * @return a new DataItem which is a logical slice of this DataItem.
	 * @throws InvalidRangeException
	 *             invalid range
	 */
	IDataItem getSlice(int dim, int value) throws InvalidRangeException;

	/**
	 * Get the java class of the DataItem data.
	 * 
	 * @return Class object
	 */
	Class<?> getType();

	/**
	 * Get the Unit String for the DataItem. Default is to use "units" attribute
	 * value
	 * 
	 * @return unit string, or null if not found.
	 */
	String getUnitsString();

	/**
	 * Does this have its data read in and cached?
	 * 
	 * @return true or false
	 */
	boolean hasCachedData();

	/**
	 * Override Object.hashCode() to implement equals.
	 * 
	 * @return integer value
	 */
	int hashCode();

	/**
	 * Invalidate the data cache.
	 */
	void invalidateCache();

	/**
	 * Will this DataItem be cached when read. Set externally, or calculated
	 * based on total size < sizeToCache.
	 * 
	 * @return true is caching
	 */
	boolean isCaching();

	/**
	 * Is this variable is a member of a Structure?
	 * 
	 * @return boolean value
	 */
	boolean isMemberOfStructure();

	/**
	 * Is this variable metadata?. Yes, if needs to be included explicitly in
	 * NcML output.
	 * 
	 * @return true or false
	 */
	boolean isMetadata();

	/**
	 * Whether this is a scalar DataItem (rank == 0).
	 * 
	 * @return true or false
	 */
	boolean isScalar();

	/**
	 * Can this variable's size grow?. This is equivalent to saying at least one
	 * of its dimensions is unlimited.
	 * 
	 * @return boolean true iff this variable can grow
	 */
	boolean isUnlimited();

	/**
	 * Is this DataItem unsigned?. Only meaningful for byte, short, int, long
	 * types.
	 * 
	 * @return true or false
	 */
	boolean isUnsigned();

	/**
	 * Get the value as a byte for a scalar DataItem. May also be
	 * one-dimensional of length 1.
	 * 
	 * @return byte object
	 * @throws IOException
	 *             if there is an IO Error
	 */
	byte readScalarByte() throws IOException;

	/**
	 * Get the value as a double for a scalar DataItem. May also be
	 * one-dimensional of length 1.
	 * 
	 * @return double value
	 * @throws IOException
	 *             if there is an IO Error
	 */
	double readScalarDouble() throws IOException;

	/**
	 * Get the value as a float for a scalar DataItem. May also be
	 * one-dimensional of length 1.
	 * 
	 * @return float value
	 * @throws IOException
	 *             if there is an IO Error
	 */
	float readScalarFloat() throws IOException;

	/**
	 * Get the value as a int for a scalar DataItem. May also be one-dimensional
	 * of length 1.
	 * 
	 * @return integer value
	 * @throws IOException
	 *             if there is an IO Error
	 */
	int readScalarInt() throws IOException;

	/**
	 * Get the value as a long for a scalar DataItem. May also be
	 * one-dimensional of length 1.
	 * 
	 * @return long value
	 * @throws IOException
	 *             if there is an IO Error
	 */
	long readScalarLong() throws IOException;

	/**
	 * Get the value as a short for a scalar DataItem. May also be
	 * one-dimensional of length 1.
	 * 
	 * @return short value
	 * @throws IOException
	 *             if there is an IO Error
	 */
	short readScalarShort() throws IOException;

	/**
	 * Get the value as a String for a scalar DataItem. May also be
	 * one-dimensional of length 1. May also be one-dimensional of type CHAR,
	 * which will be turned into a scalar String.
	 * 
	 * @return String object
	 * @throws IOException
	 *             if there is an IO Error
	 */
	String readScalarString() throws IOException;

	/**
	 * Remove an Attribute : uses the attribute hashCode to find it.
	 * 
	 * @param a
	 *            IAttribute object
	 * @return true if was found and removed
	 */
	boolean removeAttribute(IAttribute a);

	/**
	 * Set the data cache.
	 * 
	 * @param cacheData
	 *            IArray object
	 * @param isMetadata
	 *            : synthesised data, set true if must be saved in NcML output
	 *            (i.e. data not actually in the file).
	 * @throws InvalidArrayTypeException
	 *             invalid type
	 */
	void setCachedData(IArray cacheData, boolean isMetadata)
			throws InvalidArrayTypeException;

	/**
	 * Set whether to cache or not. Implies that the entire array will be
	 * stored, once read. Normally this is set automatically based on size of
	 * data.
	 * 
	 * @param caching
	 *            set if caching.
	 */
	void setCaching(boolean caching);

		/**
	 * Create a DataItem. Also must call setDataType() and setDimensions()
	 * 
	 * @param ncfile
	 *            the containing NetcdfFile.
	 * @param group
	 *            the containing group; if null, use rootGroup
	 * @param parentStructure
	 *            the containing structure; may be null
	 * @param shortName
	 *            variable shortName.
	 */

	/**
	 * Set the data type.
	 * 
	 * @param dataType
	 *            Class object
	 */
	void setDataType(Class<?> dataType);

	/**
	 * Set the dimensions using the dimensions names. The dimension is searched
	 * for recursively in the parent groups.
	 * 
	 * @param dimString
	 *            : whitespace separated list of dimension names, or '*' for
	 *            Dimension.UNKNOWN.
	 */
	void setDimensions(String dimString);

    /**
     * Set the dimension on the specified index.
     * 
     * @param dim IDimension to add to this data item
     * @param ind Index the dimension matches
     */
    void setDimension(IDimension dim, int ind) throws DimensionNotSupportedException;
    
	/**
	 * Set the element size. Usually elementSize is determined by the dataType,
	 * use this only for exceptional cases.
	 * 
	 * @param elementSize
	 *            integer value
	 */
	void setElementSize(int elementSize);

	/**
	 * Set sizeToCache.
	 * 
	 * @param sizeToCache
	 *            integer value
	 */
	void setSizeToCache(int sizeToCache);

	/**
	 * Set the units of the DataItem.
	 * 
	 * @param units
	 *            String object Created on 20/03/2008
	 */
	void setUnitsString(String units);

	/**
	 * String representation of DataItem and its attributes.
	 * 
	 * @return String object
	 */
	String toStringDebug();

	/**
	 * String representation of a DataItem and its attributes.
	 * 
	 * @param indent
	 *            start each line with this much space
	 * @param useFullName
	 *            use full name, else use short name
	 * @param strict
	 *            strictly comply with ncgen syntax
	 * @return CDL representation of the DataItem.
	 */
	String writeCDL(String indent, boolean useFullName, boolean strict);
	
	/**
	 * Clone this data item. Return a new DataItem instance but share the same
	 * Array data storage.
	 * 
	 * @return new DataItem instance
	 */
	@Override
    IDataItem clone();
    
}
