//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_ARRAYMATH_H__
#define __CDMA_ARRAYMATH_H__

#include <yat/memory/SharedPtr.h>

#include <cdma/IFactory.h>
#include <cdma/exception/Exception.h>


namespace cdma
{
	class ArrayMath {
		public:
			//destructor
			~ArrayMath() {};

			yat::SharedPtr<Array , yat::Mutex> getArray();

			/**
			 * Add two Array together, element-wisely. The two arrays must have the same
			 * shape.
			 *
			 * @param array
			 *            in Array type
			 * @return Array with new storage
			 * @throw  Exception
			 *             mismatching shape Created on 14/07/2008
			 */
			yat::SharedPtr<ArrayMath, yat::Mutex> toAdd(Array& array) throw ( Exception );
			yat::SharedPtr<ArrayMath, yat::Mutex> toAdd(ArrayMath& array) throw ( Exception );

			/**
			 * Update the array with element-wise add values from another array to its
			 * values.
			 *
			 * @param array
			 *            Array object
			 * @return Array itself
			 * @throw  Exception
			 *             mismatching shape Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> add(Array& array) throw ( Exception );
		  yat::SharedPtr<ArrayMath, yat::Mutex> add(ArrayMath& array) throw ( Exception );

			/**
			 * Add a value to the Array element-wisely.
			 *
			 * @param value
			 *            double type
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toAdd(double value);


			/**
			 * Update the array with adding a constant to its values.
			 *
			 * @param value
			 *            double type
			 * @return Array itself Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> add(double value);

			/**
			 * Multiply the two arrays element-wisely. Xij = Aij * Bij. The two arrays
			 * must have the same shape.
			 *
			 * @param array
			 *            in Array type
			 * @return Array with new storage
			 * @throw  Exception
			 *             mismatching shape Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toEltMultiply(Array& array) throw ( Exception );
		  yat::SharedPtr<ArrayMath, yat::Mutex> toEltMultiply(ArrayMath& array) throw ( Exception );

			/**
			 * Update the array with the element wise multiply of its values.
			 *
			 * @param array
			 *            Array object
			 * @return Array itself
			 * @throw  Exception
			 *             mismatching shape Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> eltMultiply(Array& array) throw ( Exception );
		  yat::SharedPtr<ArrayMath, yat::Mutex> eltMultiply(ArrayMath& array) throw ( Exception );

			/**
			 * Scale the array with a double value.
			 *
			 * @param value
			 *            double type
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toScale(double value);

			/**
			 * Update the array with the scale of its values.
			 *
			 * @param value
			 *            double type
			 * @return Array itself Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> scale(double value);

			/**
			 * Multiple two arrays in matrix multiplication rule. The two arrays must
			 * comply matrix multiply requirement.
			 *
			 * @param array
			 *            in Array type
			 * @return Array with new storage
			 * @throw  Exception
			 *             Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> matMultiply(Array& array) throw ( Exception );
		  yat::SharedPtr<ArrayMath, yat::Mutex> matMultiply(ArrayMath& array) throw ( Exception );

			/**
			 * Inverse the array assume it's a matrix.
			 *
			 * @param array
			 *          in array type
			 * @return Array with new storage
			 * @throw  Exception
			 *             Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> matInverse() throw ( Exception );

			/**
			 * Calculate the square root value of every element of the array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toSqrt();

			/**
			 * Update the array with of the square root its value.
			 *
			 * @return Array itself Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> sqrt();

			/**
			 * Calculate the e raised to the power of double values in the Array
			 * element-wisely.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toExp();

			/**
			 * Update the array with e raised to the power of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> exp();

			/**
			 * Calculate an element-wise natural logarithm of values of an
			 * Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toLn();

			/**
			 * Update the array with element-wise natural logarithm of its
			 * values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> ln();

			/**
			 * Calculate an element-wise logarithm (base 10) of values of an Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toLog10();

			/**
			 * Update the array with element-wise logarithm (base 10) of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> log10();

			/**
			 * Calculate the sine value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toSin();

			/**
			 * Update the array with sine of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> sin();

			/**
			 * Calculate the arc sine value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toAsin();

			/**
			 * Update the array with arc sine of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> asin();

			/**
			 * Calculate the cosine value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toCos();

			/**
			 * Calculate the arc cosine value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toAcos();

			/**
			 * Update the array with cosine of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> cos();

			/**
			 * Update the array with arc cosine of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> acos();

			/**
			 * Calculate the trigonometric value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toTan();

			/**
			 * Update the array with trigonometric of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> tan();

			/**
			 * Calculate the arc trigonometric value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toAtan();

			/**
			 * Update the array with arc trigonometric of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> atan();

			/**
			 * Do an element-wise power calculation of the array. Yij = Xij ^ power.
			 *
			 * @param power
			 *            double value
			 * @return Array with new storage Created on 14/07/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> toPower(double power);

			/**
			 * Update the array with to a constant power of its values.
			 *
			 * @param power
			 *            double value
			 * @return Array itself Created on 11/12/2008
			 */
		  yat::SharedPtr<ArrayMath, yat::Mutex> power(double power);

			/**
			 * Do a power-sum on a certain dimension. A power-sum will raise all element
			 * of the array to a certain power, then do a sum on a certain dimension,
			 * and put weight on the result.
			 *
			 * @param axis
			 *            Array object
			 * @param dimension
			 *            integer
			 * @param power
			 *            double value
			 * @return Array with new storage
			 * @throw  Exception
			 *             Created on 14/07/2008
			 */
		  double powerSum(Array& axis, int dimension, double power) throw ( Exception );
		  double powerSum(ArrayMath& axis, int dimension, double power) throw ( Exception );

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
			* @throw  Exception
			*             Created on 14/07/2008
			*/
		  yat::SharedPtr<ArrayMath, yat::Mutex> toEltInverse() throw ( Exception );

		  /**
			* Update the array with element-wise inverse of its values.
			*
			* @return Array itself
			* @throw  Exception
			*             divided by zero Created on 11/12/2008
			*/
		  yat::SharedPtr<ArrayMath, yat::Mutex> eltInverse() throw ( Exception );

		  /**
			* Do a element-wise inverse calculation that skip zero values. Yij = 1 /
			* Xij.
			*
			* @return Array with new storage Created on 14/07/2008
			*/
		  yat::SharedPtr<ArrayMath, yat::Mutex> toEltInverseSkipZero();

		  /**
			* Update the array with element-wise inverse of its values, skip zero
			* values.
			*
			* @return Array itself Created on 11/12/2008
			*/
		  yat::SharedPtr<ArrayMath, yat::Mutex> eltInverseSkipZero();

		  /**
			* Calculate the vector dot production of two arrays. Both array must have
			* the same size.
			*
			* @param array
			*            in Array type
			* @return Array with new storage
			* @throw  Exception
			*             Created on 14/07/2008
			*/
		  double vecDot(Array& array) throw ( Exception );

		  double vecDot(ArrayMath& array) throw ( Exception );

		  /**
			* Do sum calculation for every slice of the array on a dimension. The
			* result will be a one dimensional Array.
			*
			* @param dimension
			*            integer value
			* @param isVariance
			*            true if the array serves as variance
			* @return Array with new storage
			* @throw  Exception
			*             Created on 14/07/2008
			*/
		  yat::SharedPtr<ArrayMath, yat::Mutex> sumForDimension(int dimension, bool isVariance) throw ( Exception );

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
			* @throw  Exception
			*             Created on 14/07/2008
			*/
		  yat::SharedPtr<ArrayMath, yat::Mutex> enclosedSumForDimension(int dimension, bool isVariance) throw ( Exception );

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
		  yat::SharedPtr<ArrayMath, yat::Mutex> normalise();

		  /**
			* Element-wise multiply another array, and put the result in a given array.
			*
			* @param array
			*            GDM Array object
			* @param result
			*            GDM Array object
			* @throw  Exception
			*             Created on 01/10/2008
			*/
		  void eltMultiplyWithEqualSize(Array& array, Array& result) throw ( Exception );

		  void eltMultiplyWithEqualSize(ArrayMath& array, ArrayMath& result) throw ( Exception );

		  /**
			* Element-wise divided by another array, and put the result in a given
			* array.
			*
			* @param array
			*            GDM Array object
			* @param result
			*            GDM Array object
			* @throw  Exception
			*             Created on 01/10/2008
			*/
		  void eltDivideWithEqualSize(Array& array, Array& result) throw ( Exception );

		  void eltDivideWithEqualSize(ArrayMath& array, ArrayMath& result) throw ( Exception );

		  /**
			* Element wise divide the value by value from a given array.
			*
			* @param array
			*            Array object
			* @return new array
			* @throw  Exception
			*             mismatching shape
			*/
		  yat::SharedPtr<ArrayMath, yat::Mutex> toEltDivide(Array& array) throw ( Exception );

		  yat::SharedPtr<ArrayMath, yat::Mutex> toEltDivide(ArrayMath& array) throw ( Exception );

		  /**
			* Element wise divide the value by value from a given array.
			*
			* @param array
			*            Array object
			* @return this array after modification
			* @throw  Exception
			*             mismatching shape
			*/
		  yat::SharedPtr<ArrayMath, yat::Mutex> eltDivide(Array& array) throw ( Exception );

		  yat::SharedPtr<ArrayMath, yat::Mutex> eltDivide(ArrayMath& array) throw ( Exception );

        yat::SharedPtr<ArrayMath, yat::Mutex> eltRemainder(const Array& newArray) throw ( Exception );

		  yat::SharedPtr<ArrayMath, yat::Mutex> eltRemainder(ArrayMath& array) throw ( Exception );

        yat::SharedPtr<ArrayMath, yat::Mutex> toEltRemainder(const Array& newArray) throw ( Exception );

		  yat::SharedPtr<ArrayMath, yat::Mutex> toEltRemainder(ArrayMath& array) throw ( Exception );

		  void eltRemainderEqualSize(ArrayMath& array, ArrayMath& result) throw ( Exception );

        void eltRemainderEqualSize(const Array& newArray, const Array& result) throw ( Exception );

			yat::SharedPtr<ArrayMath, yat::Mutex> toMod(const double value);

			yat::SharedPtr<ArrayMath, yat::Mutex> mod(const double value);
		  /**
			* Calculate the determinant value.
			*
			* @param array
			*            in array type
			* @return double value
			* @throw  Exception
			*             shape not match
			*/
		  double getDeterminant() throw ( Exception );

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
		  yat::SharedPtr<IFactory, yat::Mutex> getFactory();

	};
} //namespace CDMACore
#endif //__CDMA_ARRAYMATH_H__
