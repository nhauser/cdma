package org.gumtree.data.math;

import org.gumtree.data.IFactory;
import org.gumtree.data.exception.DivideByZeroException;
import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.interfaces.IArray;

public interface IArrayMath {

	public IArray getArray();
    
    /**
     * Add two Array together, element-wisely. The two arrays must have the same
     * shape.
     * 
     * @param array
     *            in Array type
     * @return Array with new storage
     * @throws ShapeNotMatchException
     *             mismatching shape Created on 14/07/2008
     */
    IArrayMath toAdd(IArray array) throws ShapeNotMatchException;
    IArrayMath toAdd(IArrayMath array) throws ShapeNotMatchException;

    /**
     * Update the array with element-wise add values from another array to its
     * values.
     * 
     * @param array
     *            IArray object
     * @return Array itself
     * @throws ShapeNotMatchException
     *             mismatching shape Created on 14/07/2008
     */
	IArrayMath add(IArray array) throws ShapeNotMatchException;
	IArrayMath add(IArrayMath array) throws ShapeNotMatchException;

    /**
     * Add a value to the Array element-wisely.
     * 
     * @param value
     *            double type
     * @return Array with new storage Created on 14/07/2008
     */
	IArrayMath toAdd(double value);
    

    /**
     * Update the array with adding a constant to its values.
     * 
     * @param value
     *            double type
     * @return Array itself Created on 14/07/2008
     */
	IArrayMath add(double value);

    /**
     * Multiply the two arrays element-wisely. Xij = Aij * Bij. The two arrays
     * must have the same shape.
     * 
     * @param array
     *            in Array type
     * @return Array with new storage
     * @throws ShapeNotMatchException
     *             mismatching shape Created on 14/07/2008
     */
	IArrayMath toEltMultiply(IArray array) throws ShapeNotMatchException;
	IArrayMath toEltMultiply(IArrayMath array) throws ShapeNotMatchException;

    /**
     * Update the array with the element wise multiply of its values.
     * 
     * @param array
     *            IArray object
     * @return Array itself
     * @throws ShapeNotMatchException
     *             mismatching shape Created on 14/07/2008
     */
	IArrayMath eltMultiply(IArray array) throws ShapeNotMatchException;
	IArrayMath eltMultiply(IArrayMath array) throws ShapeNotMatchException;

    /**
     * Scale the array with a double value.
     * 
     * @param value
     *            double type
     * @return Array with new storage Created on 14/07/2008
     */
	IArrayMath toScale(double value);

    /**
     * Update the array with the scale of its values.
     * 
     * @param value
     *            double type
     * @return Array itself Created on 14/07/2008
     */
	IArrayMath scale(double value);

    /**
     * Multiple two arrays in matrix multiplication rule. The two arrays must
     * comply matrix multiply requirement.
     * 
     * @param array
     *            in Array type
     * @return Array with new storage
     * @throws ShapeNotMatchException
     *             Created on 14/07/2008
     */
	IArrayMath matMultiply(IArray array) throws ShapeNotMatchException;
	IArrayMath matMultiply(IArrayMath array) throws ShapeNotMatchException;

    /**
     * Inverse the array assume it's a matrix.
     * 
     * @param array
     *          in array type
     * @return Array with new storage
     * @throws ShapeNotMatchException
     *             Created on 14/07/2008
     */
	IArrayMath matInverse() throws ShapeNotMatchException;

    /**
     * Calculate the square root value of every element of the array.
     * 
     * @return Array with new storage Created on 14/07/2008
     */
	IArrayMath toSqrt();

    /**
     * Update the array with of the square root its value.
     * 
     * @return Array itself Created on 14/07/2008
     */
	IArrayMath sqrt();

    /**
     * Calculate the e raised to the power of double values in the Array
     * element-wisely.
     * 
     * @return Array with new storage Created on 14/07/2008
     */
	IArrayMath toExp();

    /**
     * Update the array with e raised to the power of its values.
     * 
     * @return Array itself Created on 11/12/2008
     */
	IArrayMath exp();

    /**
     * Calculate an element-wise natural logarithm of values of an
     * Array.
     * 
     * @return Array with new storage Created on 14/07/2008
     */
	IArrayMath toLn();

    /**
     * Update the array with element-wise natural logarithm of its
     * values.
     * 
     * @return Array itself Created on 11/12/2008
     */
	IArrayMath ln();

    /**
     * Calculate an element-wise logarithm (base 10) of values of an Array.
     * 
     * @return Array with new storage Created on 14/07/2008
     */
	IArrayMath toLog10();

    /**
     * Update the array with element-wise logarithm (base 10) of its values.
     * 
     * @return Array itself Created on 11/12/2008
     */
	IArrayMath log10();
    
    /**
     * Calculate the sine value of each elements in the Array.
     * 
     * @return Array with new storage Created on 14/07/2008
     */
	IArrayMath toSin();

    /**
     * Update the array with sine of its values.
     * 
     * @return Array itself Created on 11/12/2008
     */
	IArrayMath sin();

    /**
     * Calculate the arc sine value of each elements in the Array.
     * 
     * @return Array with new storage Created on 14/07/2008
     */
	IArrayMath toAsin();

    /**
     * Update the array with arc sine of its values.
     * 
     * @return Array itself Created on 11/12/2008
     */
	IArrayMath asin();

    /**
     * Calculate the cosine value of each elements in the Array.
     * 
     * @return Array with new storage Created on 14/07/2008
     */
	IArrayMath toCos();

    /**
     * Calculate the arc cosine value of each elements in the Array.
     * 
     * @return Array with new storage Created on 14/07/2008
     */
 	IArrayMath toAcos();

    /**
     * Update the array with cosine of its values.
     * 
     * @return Array itself Created on 11/12/2008
     */
	IArrayMath cos();

    /**
     * Update the array with arc cosine of its values.
     * 
     * @return Array itself Created on 11/12/2008
     */
	IArrayMath acos();

    /**
     * Calculate the trigonometric value of each elements in the Array.
     * 
     * @return Array with new storage Created on 14/07/2008
     */
 	IArrayMath toTan();

    /**
     * Update the array with trigonometric of its values.
     * 
     * @return Array itself Created on 11/12/2008
     */
	IArrayMath tan();

    /**
     * Calculate the arc trigonometric value of each elements in the Array.
     * 
     * @return Array with new storage Created on 14/07/2008
     */
	IArrayMath toAtan();

    /**
     * Update the array with arc trigonometric of its values.
     * 
     * @return Array itself Created on 11/12/2008
     */
	IArrayMath atan();

    /**
     * Do an element-wise power calculation of the array. Yij = Xij ^ power.
     * 
     * @param power
     *            double value
     * @return Array with new storage Created on 14/07/2008
     */
	IArrayMath toPower(double power);

    /**
     * Update the array with to a constant power of its values.
     * 
     * @param power
     *            double value
     * @return Array itself Created on 11/12/2008
     */
	IArrayMath power(double power);

    /**
     * Do a power-sum on a certain dimension. A power-sum will raise all element
     * of the array to a certain power, then do a sum on a certain dimension,
     * and put weight on the result.
     * 
     * @param axis
     *            IArray object
     * @param dimension
     *            integer
     * @param power
     *            double value
     * @return Array with new storage
     * @throws ShapeNotMatchException
     *             Created on 14/07/2008
     */
	double powerSum(IArray axis, int dimension, double power) throws ShapeNotMatchException;
	double powerSum(IArrayMath axis, int dimension, double power) throws ShapeNotMatchException;
     
	/**
	 * Calculate the sum value of the array. If an element is NaN, skip it.
	 * 
	 * @return a double value Created on 14/07/2008
	 */
	double sum();

	/**
	 * Calculate the sum value of the array. If an element is NaN, skip it. Then
	 * after calculation, normalise the result to the actual size of the array.
	 * For example, result = raw sum * size of array / (size of array - number
	 * of NaNs).
	 * 
	 * @return a double value Created on 04/08/2008
	 */
	double sumNormalise();

	/**
	 * Inverse every element of the array into a new storage.
	 * 
	 * @return Array with new storage
	 * @throws DivideByZeroException
	 *             Created on 14/07/2008
	 */
	IArrayMath toEltInverse() throws DivideByZeroException;

	/**
	 * Update the array with element-wise inverse of its values.
	 * 
	 * @return Array itself
	 * @throws DivideByZeroException
	 *             divided by zero Created on 11/12/2008
	 */
	IArrayMath eltInverse() throws DivideByZeroException;

	/**
	 * Do a element-wise inverse calculation that skip zero values. Yij = 1 /
	 * Xij.
	 * 
	 * @return Array with new storage Created on 14/07/2008
	 */
	IArrayMath toEltInverseSkipZero();

	/**
	 * Update the array with element-wise inverse of its values, skip zero
	 * values.
	 * 
	 * @return Array itself Created on 11/12/2008
	 */
	IArrayMath eltInverseSkipZero();

	/**
	 * Calculate the vector dot production of two arrays. Both array must have
	 * the same size.
	 * 
	 * @param array
	 *            in Array type
	 * @return Array with new storage
	 * @throws ShapeNotMatchException
	 *             Created on 14/07/2008
	 */
	double vecDot(IArray array) throws ShapeNotMatchException;

	double vecDot(IArrayMath array) throws ShapeNotMatchException;

	/**
	 * Do sum calculation for every slice of the array on a dimension. The
	 * result will be a one dimensional Array.
	 * 
	 * @param dimension
	 *            integer value
	 * @param isVariance
	 *            true if the array serves as variance
	 * @return IArray with new storage
	 * @throws ShapeNotMatchException
	 *             Created on 14/07/2008
	 */
	IArrayMath sumForDimension(int dimension, boolean isVariance)
			throws ShapeNotMatchException;

	/**
	 * Treat the array as a variance. Normalise the sum against the number of
	 * elements in the array.
	 * 
	 * @return double value
	 */
	double varianceSumNormalise();

	/**
	 * Do sum calculation for every slice of the array on a dimension. The
	 * result will be a one dimensional Array.
	 * 
	 * @param dimension
	 *            integer value
	 * @param isVariance
	 *            true if the array serves as variance
	 * @return Array with new storage
	 * @throws ShapeNotMatchException
	 *             Created on 14/07/2008
	 */
	IArrayMath enclosedSumForDimension(int dimension, boolean isVariance)
			throws ShapeNotMatchException;

	/**
	 * Get the L2 norm of the Array. The array must have only one dimension.
	 * 
	 * @return Array with new storage Created on 14/07/2008
	 */
	double getNorm();

	/**
	 * Normalise the vector to norm = 1.
	 * 
	 * @return Array with new storage Created on 14/07/2008
	 */
	IArrayMath normalise();

	/**
	 * Element-wise multiply another array, and put the result in a given array.
	 * 
	 * @param array
	 *            GDM Array object
	 * @param result
	 *            GDM Array object
	 * @throws ShapeNotMatchException
	 *             Created on 01/10/2008
	 */
	void eltMultiplyWithEqualSize(IArray array, IArray result)
			throws ShapeNotMatchException;

	void eltMultiplyWithEqualSize(IArrayMath array, IArrayMath result)
			throws ShapeNotMatchException;

	/**
	 * Element-wise divided by another array, and put the result in a given
	 * array.
	 * 
	 * @param array
	 *            GDM Array object
	 * @param result
	 *            GDM Array object
	 * @throws ShapeNotMatchException
	 *             Created on 01/10/2008
	 */
	void eltDivideWithEqualSize(IArray array, IArray result)
			throws ShapeNotMatchException;

	void eltDivideWithEqualSize(IArrayMath array, IArrayMath result)
			throws ShapeNotMatchException;

	/**
	 * Element wise divide the value by value from a given array.
	 * 
	 * @param array
	 *            IArray object
	 * @return new array
	 * @throws ShapeNotMatchException
	 *             mismatching shape
	 */
	IArrayMath toEltDivide(IArray array) throws ShapeNotMatchException;

	IArrayMath toEltDivide(IArrayMath array) throws ShapeNotMatchException;

	/**
	 * Element wise divide the value by value from a given array.
	 * 
	 * @param array
	 *            IArray object
	 * @return this array after modification
	 * @throws ShapeNotMatchException
	 *             mismatching shape
	 */
	IArrayMath eltDivide(IArray array) throws ShapeNotMatchException;

	IArrayMath eltDivide(IArrayMath array) throws ShapeNotMatchException;

	public IArrayMath eltRemainder(final IArray newArray)
	throws ShapeNotMatchException;

	public IArrayMath eltRemainder(IArrayMath array) throws ShapeNotMatchException;
	
	public IArrayMath toEltRemainder(final IArray newArray)
	throws ShapeNotMatchException;

	public IArrayMath toEltRemainder(IArrayMath array) throws ShapeNotMatchException;

	public void eltRemainderEqualSize(IArrayMath array, IArrayMath result) 
    throws ShapeNotMatchException;

    public void eltRemainderEqualSize(final IArray newArray,
            final IArray result) throws ShapeNotMatchException;
    
    public IArrayMath toMod(final double value);
    
    public IArrayMath mod(final double value);
	/**
	 * Calculate the determinant value.
	 * 
	 * @param array
	 *            in array type
	 * @return double value
	 * @throws ShapeNotMatchException
	 *             shape not match
	 */
	double getDeterminant() throws ShapeNotMatchException;

	/**
	 * Get maximum value of the array as a double type if it is a numeric array.
	 * 
	 * @param array
	 *            in array type
	 * @return maximum value in double type
	 */
	double getMaximum();

	/**
	 * Get minimum value of the array as a double type if it is a numeric array.
	 * 
	 * @param array
	 *            in array type
	 * @return minimum value in double type
	 */
	double getMinimum();
	
	/**
	 * Get the appropriate facgtory for this math object.
	 * 
	 * @return implementation of the factory object
	 */
	IFactory getFactory();
	
}
