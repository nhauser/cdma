package org.gumtree.data.utils;

import java.util.List;

import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IRange;

public interface IArrayUtils {
    
	IArray getArray();
    
    /**
     * Copy the contents of this array to another array. The two arrays must
     * have the same size.
     * 
     * @param newArray
     *            an existing array
     * @throws ShapeNotMatchException
     *             wrong shape
     */
	void copyTo(final IArray newArray) throws ShapeNotMatchException;

   /**
    * Copy this array to a 1D Java primitive array of type getElementType(),
    * with the physical order of the result the same as logical order.
    * 
    * @return a Java 1D array of type getElementType().
    */
	Object copyTo1DJavaArray();

	/**
	 * @param wantType
	 *            a Class instance
	 * @return generic Object instance
	 */
	Object get1DJavaArray(final Class<?> wantType);
	
   /**
    * Copy this array to a n-Dimensional Java primitive array of type
    * getElementType() and rank getRank(). Makes a copy of the data.
    * 
    * @return a Java ND array of type getElementType().
    */
	Object copyToNDJavaArray();    
    
    /**
     * Check if the shape matches with another Array object.
     * 
     * @param newArray
     *            another Array object
     * @throws ShapeNotMatchException
     *             shape not match
     */
	void checkShape(final IArray newArray) throws ShapeNotMatchException;
    
    /**
     * Concatenate with another array. The array need to be equal of less in
     * rank.
     * 
     * @param array
     *            IArray object
     * @return new IArray
     * @throws ShapeNotMatchException
     *             mismatching shape
     */
	IArrayUtils concatenate(final IArray array) throws ShapeNotMatchException;

    /**
     * Create a new Array using same backing store as this Array, by eliminating
     * any dimensions with length one.
     * 
     * @return the new Array
     */
	IArrayUtils reduce();

    /**
     * Create a new Array using same backing store as this Array, by eliminating
     * the specified dimension.
     * 
     * @param dim
     *            dimension to eliminate: must be of length one, else
     *            IllegalArgumentException
     * @return the new Array
     */
	IArrayUtils reduce(int dim);
    
    /**
     * Reduce the array to at least certain rank. The dimension with only 1 bin
     * will be reduced.
     * 
     * @param rank
     *            in int type
     * @return GDM Array with the same storage Created on 10/11/2008
     */
	IArrayUtils reduceTo(int rank);
    
    /**
     * Create a new Array, with the given shape, that references the same backing store as this Array.
     * 
     * @param shape
     *            the new shape
     * @return the new Array
     */
	IArrayUtils reshape(int[] shape) throws ShapeNotMatchException;

    
    /**
     * Create a new Array as a subsection of this Array, with rank reduction. No
     * data is moved, so the new Array references the same backing store as the
     * original.
     * <p>
     * 
     * @param origin
     *            int array specifying the starting index. Must be same rank as
     *            original Array.
     * @param shape
     *            int array specifying the extents in each dimension. This
     *            becomes the shape of the returned Array. Must be same rank as
     *            original Array. If shape[dim] == 1, then the rank of the
     *            resulting Array is reduced at that dimension.
     * @return IArray object
     * @throws InvalidRangeException
     *             invalid range
     */
	IArrayUtils section(final int[] origin, final int[] shape) throws InvalidRangeException;
    
    /**
     * Create a new Array as a subsection of this Array, with rank reduction. No
     * data is moved, so the new Array references the same backing store as the
     * original.
     * <p>
     * 
     * @param origin
     *            int array specifying the starting index. Must be same rank as
     *            original Array.
     * @param shape
     *            int array specifying the extents in each dimension. This
     *            becomes the shape of the returned Array. Must be same rank as
     *            original Array. If shape[dim] == 1, then the rank of the
     *            resulting Array is reduced at that dimension.
     * @param stride
     *            int array specifying the strides in each dimension. If null,
     *            assume all ones.
     * @return the new Array
     * @throws InvalidRangeException
     *             invalid range
     */
	IArrayUtils section(int[] origin, int[] shape, long[] stride) throws InvalidRangeException;

    /**
     * Create a new ArrayUtils as a subsection of this Array, without rank reduction.
     * No data is moved, so the new Array references the same backing store as
     * the original.
     * 
     * @param origin
     *            int array specifying the starting index. Must be same rank as
     *            original Array.
     * @param shape
     *            int array specifying the extents in each dimension. This
     *            becomes the shape of the returned Array. Must be same rank as
     *            original Array.
     * @param stride
     *            long array specifying the strides in each dimension. If null,
     *            assume all ones.
     * @return the new Array
     * @throws InvalidRangeException
     *             invalid range
     */
	IArrayUtils sectionNoReduce(int[] origin, int[] shape, long[] stride) throws InvalidRangeException;

    /**
     * Create a new ArrayUtils as a subsection of this Array, without rank reduction.
     * No data is moved, so the new Array references the same backing store as
     * the original.
     * 
     * @param ranges
     *            list of Ranges that specify the array subset. Must be same
     *            rank as original Array. A particular Range: 1) may be a
     *            subset, or 2) may be null, meaning use entire Range.
     * @return the new Array
     */
	IArrayUtils sectionNoReduce(List<IRange> ranges) throws InvalidRangeException;


	/**
	 * Create a new Array using same backing store as this Array, by fixing the
	 * specified dimension at the specified index value. This reduces rank by 1.
	 * 
	 * @param dim
	 *            which dimension to fix
	 * @param value
	 *            at what index value
	 * @return a new Array
	 */
	IArrayUtils slice(int dim, int value);
	
    /**
     * Create a new Array using same backing store as this Array, by transposing
     * two of the indices.
     * 
     * @param dim1
     *            transpose these two indices
     * @param dim2
     *            transpose these two indices
     * @return the new Array
     */
	IArrayUtils transpose(int dim1, int dim2);
            
    /**
     * Check if the two arrays are conformable. They must have exactly the same
     * shape (excluding dimensions of length 1)
     * 
     * @param array
     *            in Array type
     * @return Array with new storage Created on 14/07/2008
     */
	boolean isConformable(IArray array);

    /**
     * Element-wise apply a boolean map to the array. The values of the Array
     * will get updated. The map's rank must be smaller or equal to the rank of
     * the array. If the rank of the map is smaller, apply the map to subset of
     * the array in the lowest dimensions iteratively. For each element, if the
     * AND map value is true, return itself, otherwise return NaN.
     * 
     * @param booleanMap
     *            boolean Array
     * @return Array itself
     * @throws ShapeNotMatchException
     *             Created on 04/08/2008
     */
	IArrayUtils eltAnd(IArray booleanMap) throws ShapeNotMatchException;

    /**
     * Integrate on given dimension. The result array will be one dimensional
     * reduced from the given array.
     * 
     * @param dimension
     *            integer value
     * @param isVariance
     *            true if the array serves as variance
     * @return new Array object
     * @throws ShapeNotMatchException
     *             Created on 30/09/2008
     */
	IArrayUtils integrateDimension(int dimension, boolean isVariance) throws ShapeNotMatchException;
    
    /**
     * Integrate on given dimension. The result array will be one dimensional
     * reduced from the given array.
     * 
     * @param dimension
     *            integer value
     * @param isVariance
     *            true if the array serves as variance
     * @return new Array object
     * @throws ShapeNotMatchException
     *             Created on 30/09/2008
     */
	IArrayUtils enclosedIntegrateDimension(int dimension, boolean isVariance) throws ShapeNotMatchException;
	
	/**
	 * Create a new Array using same backing store as this Array, by flipping
	 * the index so that it runs from shape[index]-1 to 0.
	 * 
	 * @param dim
	 *            dimension to flip
	 * @return the new Array
	 */
	IArrayUtils flip(int dim);
	
	/**
	 * Create a new Array using same backing store as this Array, by permuting
	 * the indices.
	 * 
	 * @param dims
	 *            the old index dims[k] becomes the new kth index.
	 * @return the new Array
	 */
	IArrayUtils permute(int[] dims);


    
}
