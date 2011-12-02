//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IARRAYMATH_H__
#define __CDMA_IARRAYMATH_H__

#include <yat/memory/SharedPtr.h>

#include <cdma/IFactory.h>
#include <cdma/exception/Exception.h>


namespace cdma
{
	class IArrayMath {
		public:
			//Virtual destructor
			virtual ~IArrayMath() {};

			virtual yat::SharedPtr<IArray , yat::Mutex> getArray() = 0;

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
			virtual yat::SharedPtr<IArrayMath, yat::Mutex> toAdd(IArray& array) throw ( Exception ) = 0;
			virtual yat::SharedPtr<IArrayMath, yat::Mutex> toAdd(IArrayMath& array) throw ( Exception ) = 0;

			/**
			 * Update the array with element-wise add values from another array to its
			 * values.
			 *
			 * @param array
			 *            IArray object
			 * @return Array itself
			 * @throw  Exception
			 *             mismatching shape Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> add(IArray& array) throw ( Exception ) = 0;
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> add(IArrayMath& array) throw ( Exception ) = 0;

			/**
			 * Add a value to the Array element-wisely.
			 *
			 * @param value
			 *            double type
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toAdd(double value) = 0;


			/**
			 * Update the array with adding a constant to its values.
			 *
			 * @param value
			 *            double type
			 * @return Array itself Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> add(double value) = 0;

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
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toEltMultiply(IArray& array) throw ( Exception ) = 0;
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toEltMultiply(IArrayMath& array) throw ( Exception ) = 0;

			/**
			 * Update the array with the element wise multiply of its values.
			 *
			 * @param array
			 *            IArray object
			 * @return Array itself
			 * @throw  Exception
			 *             mismatching shape Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> eltMultiply(IArray& array) throw ( Exception ) = 0;
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> eltMultiply(IArrayMath& array) throw ( Exception ) = 0;

			/**
			 * Scale the array with a double value.
			 *
			 * @param value
			 *            double type
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toScale(double value) = 0;

			/**
			 * Update the array with the scale of its values.
			 *
			 * @param value
			 *            double type
			 * @return Array itself Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> scale(double value) = 0;

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
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> matMultiply(IArray& array) throw ( Exception ) = 0;
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> matMultiply(IArrayMath& array) throw ( Exception ) = 0;

			/**
			 * Inverse the array assume it's a matrix.
			 *
			 * @param array
			 *          in array type
			 * @return Array with new storage
			 * @throw  Exception
			 *             Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> matInverse() throw ( Exception ) = 0;

			/**
			 * Calculate the square root value of every element of the array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toSqrt() = 0;

			/**
			 * Update the array with of the square root its value.
			 *
			 * @return Array itself Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> sqrt() = 0;

			/**
			 * Calculate the e raised to the power of double values in the Array
			 * element-wisely.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toExp() = 0;

			/**
			 * Update the array with e raised to the power of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> exp() = 0;

			/**
			 * Calculate an element-wise natural logarithm of values of an
			 * Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toLn() = 0;

			/**
			 * Update the array with element-wise natural logarithm of its
			 * values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> ln() = 0;

			/**
			 * Calculate an element-wise logarithm (base 10) of values of an Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toLog10() = 0;

			/**
			 * Update the array with element-wise logarithm (base 10) of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> log10() = 0;

			/**
			 * Calculate the sine value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toSin() = 0;

			/**
			 * Update the array with sine of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> sin() = 0;

			/**
			 * Calculate the arc sine value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toAsin() = 0;

			/**
			 * Update the array with arc sine of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> asin() = 0;

			/**
			 * Calculate the cosine value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toCos() = 0;

			/**
			 * Calculate the arc cosine value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toAcos() = 0;

			/**
			 * Update the array with cosine of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> cos() = 0;

			/**
			 * Update the array with arc cosine of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> acos() = 0;

			/**
			 * Calculate the trigonometric value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toTan() = 0;

			/**
			 * Update the array with trigonometric of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> tan() = 0;

			/**
			 * Calculate the arc trigonometric value of each elements in the Array.
			 *
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toAtan() = 0;

			/**
			 * Update the array with arc trigonometric of its values.
			 *
			 * @return Array itself Created on 11/12/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> atan() = 0;

			/**
			 * Do an element-wise power calculation of the array. Yij = Xij ^ power.
			 *
			 * @param power
			 *            double value
			 * @return Array with new storage Created on 14/07/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toPower(double power) = 0;

			/**
			 * Update the array with to a constant power of its values.
			 *
			 * @param power
			 *            double value
			 * @return Array itself Created on 11/12/2008
			 */
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> power(double power) = 0;

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
			 * @throw  Exception
			 *             Created on 14/07/2008
			 */
		  virtual double powerSum(IArray& axis, int dimension, double power) throw ( Exception ) = 0;
		  virtual double powerSum(IArrayMath& axis, int dimension, double power) throw ( Exception ) = 0;

		  /**
			* Calculate the sum value of the array. If an element is NaN, skip it.
			*
			* @return a double value Created on 14/07/2008
			*/
		  virtual double sum() = 0;

		  /**
			* Calculate the sum value of the array. If an element is NaN, skip it. Then
			* after calculation, normalise the result to the actual size of the array.
			* For example, result = raw sum * size of array / (size of array - number
			* of NaNs).
			*
			* @return a double value Created on 04/08/2008
			*/
		  virtual double sumNormalise() = 0;

		  /**
			* Inverse every element of the array into a new storage.
			*
			* @return Array with new storage
			* @throw  Exception
			*             Created on 14/07/2008
			*/
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toEltInverse() throw ( Exception ) = 0;

		  /**
			* Update the array with element-wise inverse of its values.
			*
			* @return Array itself
			* @throw  Exception
			*             divided by zero Created on 11/12/2008
			*/
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> eltInverse() throw ( Exception ) = 0;

		  /**
			* Do a element-wise inverse calculation that skip zero values. Yij = 1 /
			* Xij.
			*
			* @return Array with new storage Created on 14/07/2008
			*/
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toEltInverseSkipZero() = 0;

		  /**
			* Update the array with element-wise inverse of its values, skip zero
			* values.
			*
			* @return Array itself Created on 11/12/2008
			*/
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> eltInverseSkipZero() = 0;

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
		  virtual double vecDot(IArray& array) throw ( Exception ) = 0;

		  virtual double vecDot(IArrayMath& array) throw ( Exception ) = 0;

		  /**
			* Do sum calculation for every slice of the array on a dimension. The
			* result will be a one dimensional Array.
			*
			* @param dimension
			*            integer value
			* @param isVariance
			*            true if the array serves as variance
			* @return IArray with new storage
			* @throw  Exception
			*             Created on 14/07/2008
			*/
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> sumForDimension(int dimension, bool isVariance) throw ( Exception ) = 0;

		  /**
			* Treat the array as a variance. Normalise the sum against the number of
			* elements in the array.
			*
			* @return double value
			*/
		  virtual double varianceSumNormalise() = 0;

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
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> enclosedSumForDimension(int dimension, bool isVariance) throw ( Exception ) = 0;

		  /**
			* Get the L2 norm of the Array. The array must have only one dimension.
			*
			* @return Array with new storage Created on 14/07/2008
			*/
		  virtual double getNorm() = 0;

		  /**
			* Normalise the vector to norm = 1.
			*
			* @return Array with new storage Created on 14/07/2008
			*/
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> normalise() = 0;

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
		  virtual void eltMultiplyWithEqualSize(IArray& array, IArray& result) throw ( Exception ) = 0;

		  virtual void eltMultiplyWithEqualSize(IArrayMath& array, IArrayMath& result) throw ( Exception ) = 0;

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
		  virtual void eltDivideWithEqualSize(IArray& array, IArray& result) throw ( Exception ) = 0;

		  virtual void eltDivideWithEqualSize(IArrayMath& array, IArrayMath& result) throw ( Exception ) = 0;

		  /**
			* Element wise divide the value by value from a given array.
			*
			* @param array
			*            IArray object
			* @return new array
			* @throw  Exception
			*             mismatching shape
			*/
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toEltDivide(IArray& array) throw ( Exception ) = 0;

		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toEltDivide(IArrayMath& array) throw ( Exception ) = 0;

		  /**
			* Element wise divide the value by value from a given array.
			*
			* @param array
			*            IArray object
			* @return this array after modification
			* @throw  Exception
			*             mismatching shape
			*/
		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> eltDivide(IArray& array) throw ( Exception ) = 0;

		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> eltDivide(IArrayMath& array) throw ( Exception ) = 0;

        virtual yat::SharedPtr<IArrayMath, yat::Mutex> eltRemainder(const IArray& newArray) throw ( Exception ) = 0;

		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> eltRemainder(IArrayMath& array) throw ( Exception ) = 0;

        virtual yat::SharedPtr<IArrayMath, yat::Mutex> toEltRemainder(const IArray& newArray) throw ( Exception ) = 0;

		  virtual yat::SharedPtr<IArrayMath, yat::Mutex> toEltRemainder(IArrayMath& array) throw ( Exception ) = 0;

		  virtual void eltRemainderEqualSize(IArrayMath& array, IArrayMath& result) throw ( Exception ) = 0;

        virtual void eltRemainderEqualSize(const IArray& newArray, const IArray& result) throw ( Exception ) = 0;

			virtual yat::SharedPtr<IArrayMath, yat::Mutex> toMod(const double value) = 0;

			virtual yat::SharedPtr<IArrayMath, yat::Mutex> mod(const double value) = 0;
		  /**
			* Calculate the determinant value.
			*
			* @param array
			*            in array type
			* @return double value
			* @throw  Exception
			*             shape not match
			*/
		  virtual double getDeterminant() throw ( Exception ) = 0;

		  /**
			* Get maximum value of the array as a double type if it is a numeric array.
			*
			* @param array
			*            in array type
			* @return maximum value in double type
			*/
		  virtual double getMaximum() = 0;

		  /**
			* Get minimum value of the array as a double type if it is a numeric array.
			*
			* @param array
			*            in array type
			* @return minimum value in double type
			*/
		  virtual double getMinimum() = 0;

		  /**
			* Get the appropriate facgtory for this math object.
			*
			* @return implementation of the factory object
			*/
		  virtual yat::SharedPtr<IFactory, yat::Mutex> getFactory() = 0;

	};
} //namespace CDMACore
#endif //__CDMA_IARRAYMATH_H__
