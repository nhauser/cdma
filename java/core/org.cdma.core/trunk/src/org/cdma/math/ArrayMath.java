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
// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
//
// Contributors:
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
//    ClÃ©ment Rodriguez (clement.rodriguez@synchrotron-soleil.fr) - API evolution
// ****************************************************************************
package org.cdma.math;

/// @cond internal

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.exception.DivideByZeroException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.ISliceIterator;

public abstract class ArrayMath implements IArrayMath {

    private IArray m_array;

    private IFactory factory;

    public ArrayMath(final IArray array, final IFactory factory) {
        m_array = array;
        this.factory = factory;
        if (this.factory == null) {
            this.factory = Factory.getFactory();
        }
    }

    @Override
    public IArray getArray() {
        return m_array;
    }

    /**
     * Add two IArray together, element-wisely. The two arrays must have the same
     * shape.
     * 
     * @param array in IArray type
     * @return IArray with new storage
     * @throws ShapeNotMatchException
     *             mismatching shape
     */
    @Override
    public IArrayMath toAdd(final IArrayMath array) throws ShapeNotMatchException {
        return toAdd(array.getArray());
    }

    @Override
    public IArrayMath add(final IArrayMath array) throws ShapeNotMatchException {
        return add(array.getArray());
    }

    /**
     * Add a value to the IArray element-wisely.
     * 
     * @param value double type
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toAdd(final double value) {
        // [ANSTO][Tony][2011-08-30] Resulting array should be typed as double
        IArray result = getFactory().createArray(double.class, getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.next();
            newIterator.setDouble(oldIterator.getDoubleNext() + value);
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with adding a constant to its values.
     * 
     * @param value double type
     * @return IArray itself
     */
    @Override
    public IArrayMath add(final double val) {
        IArrayIterator iter = getArray().getIterator();
        while (iter.hasNext()) {
            iter.next();
            iter.setDouble(iter.getDoubleNext() + val);
        }
        getArray().setDirty(true);
        return this;
    }

    @Override
    public IArrayMath eltRemainder(final IArray newArray)
            throws ShapeNotMatchException {
        getArray().getArrayUtils().checkShape(newArray);
        if (getArray().getRank() == newArray.getRank()) {
            eltRemainderEqualSize(newArray, getArray());
        } else {
            ISliceIterator sourceSliceIterator = null;
            try {
                sourceSliceIterator = getArray().getSliceIterator(
                        newArray.getRank());
                while (sourceSliceIterator.hasNext()) {
                    IArray sourceSlice = sourceSliceIterator.getArrayNext();
                    sourceSlice.getArrayMath().eltRemainderEqualSize(newArray,
                            sourceSlice);
                }
            } catch (InvalidRangeException e) {
                throw new ShapeNotMatchException("shape is invalid");
            }
        }
        getArray().setDirty(true);
        return this;
    }

    @Override
    public IArrayMath eltRemainder(final IArrayMath array)
            throws ShapeNotMatchException {
        return eltRemainder(array.getArray());
    }

    /**
     * Multiply the two arrays element-wisely. Xij = Aij * Bij. The two arrays
     * must have the same shape.
     * 
     * @param array in IArray type
     * @return IArray with new storage
     * @throws ShapeNotMatchException
     *             mismatching shape
     */
    @Override
    public IArrayMath toEltRemainder(final IArray newArray)
            throws ShapeNotMatchException {
        getArray().getArrayUtils().checkShape(newArray);
        IArrayMath arrMath = newArray.getArrayMath();
        IArray result = getFactory().createArray(getArray().getElementType(),
                getArray().getShape());
        if (getArray().getRank() == newArray.getRank()) {
            eltRemainderEqualSize(newArray, result);
        } else {
            ISliceIterator sourceSliceIterator = null;
            ISliceIterator resultSliceIterator = null;
            try {
                sourceSliceIterator = getArray().getSliceIterator(
                        newArray.getRank());
                resultSliceIterator = result.getSliceIterator(newArray
                        .getRank());
                while (sourceSliceIterator.hasNext()
                        && resultSliceIterator.hasNext()) {
                    IArray sourceSlice = sourceSliceIterator.getArrayNext();
                    IArray resultSlice = resultSliceIterator.getArrayNext();
                    arrMath.eltRemainderEqualSize(sourceSlice, resultSlice);
                }
            } catch (InvalidRangeException e) {
                throw new ShapeNotMatchException("shape is invalid");
            }
        }
        return result.getArrayMath();
    }

    @Override
    public IArrayMath toEltRemainder(final IArrayMath array)
            throws ShapeNotMatchException {
        return toEltRemainder(array.getArray());
    }

    @Override
    public IArrayMath toEltMultiply(final IArray newArray)
            throws ShapeNotMatchException {
        getArray().getArrayUtils().checkShape(newArray);
        IArrayMath arrMath = newArray.getArrayMath();
        IArray result = getFactory().createArray(getArray().getElementType(),
                getArray().getShape());
        if (getArray().getRank() == newArray.getRank()) {
            eltMultiplyWithEqualSize(newArray, result);
        } else {
            ISliceIterator sourceSliceIterator = null;
            ISliceIterator resultSliceIterator = null;
            try {
                sourceSliceIterator = getArray().getSliceIterator(
                        newArray.getRank());
                resultSliceIterator = result.getSliceIterator(newArray
                        .getRank());
                while (sourceSliceIterator.hasNext()
                        && resultSliceIterator.hasNext()) {
                    IArray sourceSlice = sourceSliceIterator.getArrayNext();
                    IArray resultSlice = resultSliceIterator.getArrayNext();
                    arrMath.eltMultiplyWithEqualSize(sourceSlice, resultSlice);
                }
            } catch (InvalidRangeException e) {
                throw new ShapeNotMatchException("shape is invalid");
            }
        }
        return result.getArrayMath();
    }

    @Override
    public IArrayMath toEltMultiply(final IArrayMath array)
            throws ShapeNotMatchException {
        return toEltMultiply(array.getArray());
    }

    /**
     * Update the array with the element wise multiply of its values.
     * 
     * @param array IArray object
     * @return IArray itself
     * @throws ShapeNotMatchException
     *             mismatching shape
     */
    @Override
    public IArrayMath eltMultiply(final IArray newArray)
            throws ShapeNotMatchException {
        getArray().getArrayUtils().checkShape(newArray);
        if (getArray().getRank() == newArray.getRank()) {
            eltMultiplyWithEqualSize(newArray, getArray());
        } else {
            ISliceIterator sourceSliceIterator = null;
            try {
                sourceSliceIterator = getArray().getSliceIterator(
                        newArray.getRank());
                while (sourceSliceIterator.hasNext()) {
                    IArray sourceSlice = sourceSliceIterator.getArrayNext();
                    sourceSlice.getArrayMath().eltMultiplyWithEqualSize(
                            newArray, sourceSlice);
                }
            } catch (InvalidRangeException e) {
                throw new ShapeNotMatchException("shape is invalid");
            }
        }
        getArray().setDirty(true);
        return this;
    }

    @Override
    public IArrayMath eltMultiply(final IArrayMath array)
            throws ShapeNotMatchException {
        return eltMultiply(array.getArray());
    }

    /**
     * Scale the array with a double value.
     * 
     * @param value double type
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toScale(final double value) {
        System.out.println("ArrayMath.toScale()");
        IArray result = getFactory().createArray(double.class, getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.next();
            newIterator.setDouble(oldIterator.getDoubleNext() * value);
        }
        return result.getArrayMath();
    }

    /**
     * Modulo the array with a double value.
     * 
     * @param value double type
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toMod(final double value) {
        IArray result = getFactory().createArray(double.class, getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.next();
            newIterator.setDouble(oldIterator.getDoubleNext() % value);
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with the scale of its values.
     * 
     * @param value double type
     * @return IArray itself
     */
    @Override
    public IArrayMath scale(final double value) {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            oldIterator.setDouble(oldIterator.getDoubleNext() * value);
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Update the array with the mod of a value.
     * 
     * @param value double type
     * @return IArray itself
     */
    @Override
    public IArrayMath mod(final double value) {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            oldIterator.setDouble(oldIterator.getDoubleNext() % value);
        }
        getArray().setDirty(true);
        return this;
    }

    @Override
    public IArrayMath matMultiply(final IArrayMath array)
            throws ShapeNotMatchException {
        return matMultiply(array.getArray());
    }

    /**
     * Calculate the square root value of every element of the array.
     * 
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toSqrt() {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.setDouble(Math.sqrt(oldIterator.getDoubleNext()));
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with of the square root its value.
     * 
     * @return IArray itself
     */
    @Override
    public IArrayMath sqrt() {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            oldIterator.setDouble(Math.sqrt(oldIterator.getDoubleNext()));
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Calculate the e raised to the power of double values in the IArray
     * element-wisely.
     * 
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toExp() {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.next();
            newIterator.setDouble(Math.exp(oldIterator.getDoubleNext()));
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with e raised to the power of its values.
     * 
     * @return IArray itself
     */
    @Override
    public IArrayMath exp() {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            oldIterator.setDouble(Math.exp(oldIterator.getDoubleNext()));
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Calculate an element-wise natural logarithm of values of an IArray.
     * 
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toLn() {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            double value = oldIterator.getDoubleNext();
            newIterator.next();
            if (value == 0) {
                newIterator.setDouble(Double.NaN);
            } else {
                newIterator.setDouble(Math.log(value));
            }
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with element-wise natural logarithm of its values.
     * 
     * @return IArray itself
     */
    @Override
    public IArrayMath ln() {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            double value = oldIterator.getDoubleNext();
            if (value == 0) {
                oldIterator.setDouble(Double.NaN);
            } else {
                oldIterator.setDouble(Math.log(value));
            }
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Calculate an element-wise logarithm (base 10) of values of an IArray.
     * 
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toLog10() {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            double value = oldIterator.getDoubleNext();
            newIterator.next();
            if (value == 0) {
                newIterator.setDouble(Double.NaN);
            } else {
                newIterator.setDouble(Math.log10(value));
            }
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with element-wise logarithm (base 10) of its values.
     * 
     * @return IArray itself
     */
    @Override
    public IArrayMath log10() {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            double value = oldIterator.getDoubleNext();
            if (value == 0) {
                oldIterator.setDouble(Double.NaN);
            } else {
                oldIterator.setDouble(Math.log10(value));
            }
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Calculate the sine value of each elements in the IArray.
     * 
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toSin() {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.next();
            newIterator.setDouble(Math.sin(oldIterator.getDoubleNext()));
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with sine of its values.
     * 
     * @return IArray itself
     */
    @Override
    public IArrayMath sin() {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            oldIterator.setDouble(Math.sin(oldIterator.getDoubleNext()));
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Calculate the arc sine value of each elements in the IArray.
     * 
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toAsin() {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.next();
            newIterator.setDouble(Math.asin(oldIterator.getDoubleNext()));
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with arc sine of its values.
     * 
     * @return IArray itself
     */
    @Override
    public IArrayMath asin() {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            oldIterator.setDouble(Math.asin(oldIterator.getDoubleNext()));
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Calculate the cosine value of each elements in the IArray.
     * 
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toCos() {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.next();
            newIterator.setDouble(Math.cos(oldIterator.getDoubleNext()));
        }
        return result.getArrayMath();
    }

    /**
     * Calculate the arc cosine value of each elements in the IArray.
     * 
     * @param array in array type
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toAcos() {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.next();
            newIterator.setDouble(Math.acos(oldIterator.getDoubleNext()));
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with cosine of its values.
     * 
     * @param array in array type
     * @return IArray itself
     */
    @Override
    public IArrayMath cos() {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            oldIterator.setDouble(Math.cos(oldIterator.getDoubleNext()));
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Update the array with arc cosine of its values.
     * 
     * @return IArray itself
     */
    @Override
    public IArrayMath acos() {
        IArrayIterator iterator = getArray().getIterator();
        while (iterator.hasNext()) {
            iterator.setDouble(Math.acos(iterator.getDoubleNext()));
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Calculate the trigonometric value of each elements in the IArray.
     * 
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toTan() {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.next();
            newIterator.setDouble(Math.tan(oldIterator.getDoubleNext()));
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with trigonometric of its values.
     * 
     * @return IArray itself
     */
    @Override
    public IArrayMath tan() {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            oldIterator.setDouble(Math.tan(oldIterator.getDoubleNext()));
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Calculate the arc trigonometric value of each elements in the IArray.
     * 
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toAtan() {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.next();
            newIterator.setDouble(Math.atan(oldIterator.getDoubleNext()));
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with arc trigonometric of its values.
     * 
     * @return IArray itself
     */
    @Override
    public IArrayMath atan() {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            oldIterator.setDouble(Math.atan(oldIterator.getDoubleNext()));
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Do an element-wise power calculation of the array. Yij = Xij ^ power.
     * 
     * @param power double value
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toPower(final double value) {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            newIterator.next();
            newIterator.setDouble(Math.pow(oldIterator.getDoubleNext(),  value));
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with to a constant power of its values.
     * 
     * @param power double value
     * @return IArray itself
     */
    @Override
    public IArrayMath power(final double value) {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            oldIterator.setDouble(Math.pow(oldIterator.getDoubleNext(),value));
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Do a power-sum on a certain dimension. A power-sum will raise all element
     * of the array to a certain power, then do a sum on a certain dimension,
     * and put weight on the result.
     * 
     * @param axis IArray object
     * @param dimension integer
     * @param power double value
     * @return IArray with new storage
     * @throws ShapeNotMatchException
     */
    @Override
    public double powerSum(final IArray axis, final int dimension,
            final double power) throws ShapeNotMatchException {
        if (dimension >= getArray().getRank()) {
            throw new ShapeNotMatchException(dimension
                    + " dimension is not available");
        }
        int[] shape = getArray().getShape();
        if (axis != null && axis.getSize() < shape[dimension]) {
            throw new ShapeNotMatchException("axis size not match");
        }
        IArray result = getFactory().createArray(getArray().getElementType(), getArray().getShape());
        result = power(power).getArray();
        double powerSum = 0;
        for (int i = 0; i < shape[dimension]; i++) {
            IArray sumOnDimension;
            if (axis == null) {
                sumOnDimension = result.getArrayMath().sumForDimension(i, false).scale(i).getArray();
            } else {
                sumOnDimension = result.getArrayMath().sumForDimension(i, false).eltMultiply(axis).getArray();
            }
            powerSum += sumOnDimension.getArrayMath().sum();
        }
        return powerSum;
    }

    @Override
    public double powerSum(final IArrayMath axis, final int dimension, final double power)
            throws ShapeNotMatchException {
        return powerSum(axis.getArray(), dimension, power);
    }

    /**
     * Calculate the sum value of the array. If an element is NaN, skip it.
     * 
     * @return a double value
     */
    @Override
    public double sum() {
        double sum = Double.NaN;
        IArrayIterator iterator = getArray().getIterator();
        while (iterator.hasNext()) {
            Double value = iterator.getDoubleNext();
            if (!value.isNaN()) {
                sum = value;
                break;
            }
        }
        while (iterator.hasNext()) {
            Double value = iterator.getDoubleNext();
            if (!value.isNaN()) {
                sum += value;
            }
        }
        return sum;
    }

    /**
     * Calculate the sum value of the array. If an element is NaN, skip it. Then
     * after calculation, normalise the result to the actual size of the array.
     * For example, result = raw sum * size of array / (size of array - number
     * of NaNs).
     * 
     * @return a double value
     */
    @Override
    public double sumNormalise() {
        double sum = Double.NaN;
        int countNaN = 0;
        IArrayIterator iterator = getArray().getIterator();
        while (iterator.hasNext()) {
            Double value = iterator.getDoubleNext();
            if (value.isNaN()) {
                countNaN++;
            } else {
                sum = value;
                break;
            }
        }
        while (iterator.hasNext()) {
            Double value = iterator.getDoubleNext();
            if (value.isNaN()) {
                countNaN++;
            } else {
                sum += value;
            }
        }
        if (Double.isNaN(sum)) {
            return sum;
        }
        return Double.valueOf(sum) * getArray().getSize()
                / Double.valueOf(getArray().getSize() - countNaN);
    }

    /**
     * Inverse every element of the array into a new storage.
     * 
     * @return IArray with new storage
     * @throws DivideByZeroException
     */
    @Override
    public IArrayMath toEltInverse() throws DivideByZeroException {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            try {
                newIterator.next();
                newIterator.setDouble(1 / oldIterator.getDoubleNext());
            } catch (Exception e) {
                throw new DivideByZeroException(e);
            }
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with element-wise inverse of its values.
     * 
     * @return IArray itself
     * @throws DivideByZeroException
     *             divided by zero
     */
    @Override
    public IArrayMath eltInverse() throws DivideByZeroException {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            try {
                oldIterator.setDouble(1 / oldIterator.getDoubleNext());
            } catch (Exception e) {
                throw new DivideByZeroException(e);
            }
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Do a element-wise inverse calculation that skip zero values. Yij = 1 /
     * Xij.
     * 
     * @return IArray with new storage
     */
    @Override
    public IArrayMath toEltInverseSkipZero() {
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        IArrayIterator oldIterator = getArray().getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (oldIterator.hasNext()) {
            double det = oldIterator.getDoubleNext();
            newIterator.next();
            newIterator.setDouble(det == 0 ? 0 : 1 / det);
        }
        return result.getArrayMath();
    }

    /**
     * Update the array with element-wise inverse of its values, skip zero
     * values.
     * 
     * @return IArray itself
     */
    @Override
    public IArrayMath eltInverseSkipZero() {
        IArrayIterator oldIterator = getArray().getIterator();
        while (oldIterator.hasNext()) {
            double det = oldIterator.getDoubleNext();
            oldIterator.setDouble(det == 0 ? 0 : 1 / det);
        }
        getArray().setDirty(true);
        return this;
    }

    /**
     * Calculate the vector dot production of two arrays. Both array must have
     * the same size.
     * 
     * @param array in IArray type
     * @return IArray with new storage
     * @throws ShapeNotMatchException
     */
    @Override
    public double vecDot(final IArray newArray) throws ShapeNotMatchException {
        try {
            return toEltMultiply(newArray).sum();
        } catch (Exception e) {
            throw new ShapeNotMatchException(e);
        }
    }

    @Override
    public double vecDot(final IArrayMath array) throws ShapeNotMatchException {
        return vecDot(array.getArray());
    }

    /**
     * Treat the array as a variance. Normalize the sum against the number of
     * elements in the array.
     * 
     * @return double value
     */
    @Override
    public double varianceSumNormalise() {
        double sum = Double.NaN;
        int countNaN = 0;
        IArrayIterator iterator = getArray().getIterator();
        while (iterator.hasNext()) {
            Double value = iterator.getDoubleNext();
            if (value.isNaN()) {
                countNaN++;
            } else {
                sum = value;
                break;
            }
        }
        while (iterator.hasNext()) {
            Double value = iterator.getDoubleNext();
            if (value.isNaN()) {
                countNaN++;
            } else {
                sum += value;
            }
        }
        if (Double.isNaN(sum)) {
            return sum;
        }
        double normaliseFactor = getArray().getSize()
                / Double.valueOf(getArray().getSize() - countNaN);
        return Double.valueOf(sum) * normaliseFactor * normaliseFactor;
    }

    /**
     * Element-wise multiply another array, and put the result in a given array.
     * 
     * @param array CDMA IArray object
     * @param result CDMA IArray object
     * @throws ShapeNotMatchException
     */
    @Override
    public void eltMultiplyWithEqualSize(final IArray newArray, final IArray result)
            throws ShapeNotMatchException {
        if (getArray().getSize() != newArray.getSize()) {
            throw new ShapeNotMatchException("the size of the arrays not match");
        }
        IArrayIterator iterator1 = getArray().getIterator();
        IArrayIterator iterator2 = newArray.getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (iterator1.hasNext()) {
            newIterator.next();
            newIterator.setDouble(iterator1.getDoubleNext()
                    * iterator2.getDoubleNext());
        }
        getArray().setDirty(true);
    }

    @Override
    public void eltMultiplyWithEqualSize(final IArrayMath array, final IArrayMath result)
            throws ShapeNotMatchException {
        eltMultiplyWithEqualSize(array.getArray(), result.getArray());
    }

    @Override
    public void eltRemainderEqualSize(final IArrayMath array, final IArrayMath result)
            throws ShapeNotMatchException {
        eltRemainderEqualSize(array.getArray(), result.getArray());
    }

    @Override
    public void eltRemainderEqualSize(final IArray newArray, final IArray result)
            throws ShapeNotMatchException {
        if (getArray().getSize() != newArray.getSize()) {
            throw new ShapeNotMatchException("the size of the arrays not match");
        }
        IArrayIterator iterator1 = getArray().getIterator();
        IArrayIterator iterator2 = newArray.getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (iterator1.hasNext()) {
            newIterator.next();
            newIterator.setDouble(iterator1.getDoubleNext()
                    % iterator2.getDoubleNext());
        }
        getArray().setDirty(true);
    }

    /**
     * Element-wise divided by another array, and put the result in a given
     * array.
     * 
     * @param array CDMA IArray object
     * @param result CDMA IArray object
     * @throws ShapeNotMatchException
     */
    @Override
    public void eltDivideWithEqualSize(final IArray newArray, final IArray result)
            throws ShapeNotMatchException {
        if (getArray().getSize() != newArray.getSize()) {
            throw new ShapeNotMatchException("the size of the arrays not match");
        }
        IArrayIterator iterator1 = getArray().getIterator();
        IArrayIterator iterator2 = newArray.getIterator();
        IArrayIterator newIterator = result.getIterator();
        while (iterator1.hasNext()) {
            double newValue = iterator2.getDoubleNext();
            newIterator.next();
            if (newValue != 0) {
                newIterator.setDouble(iterator1.getDoubleNext() / newValue);
            } else {
                newIterator.setDouble(iterator1.getDoubleNext());
            }
        }
        getArray().setDirty(true);
    }

    @Override
    public void eltDivideWithEqualSize(final IArrayMath array, final IArrayMath result)
            throws ShapeNotMatchException {
        eltDivideWithEqualSize(array.getArray(), result.getArray());
    }

    /**
     * Element wise divide the value by value from a given array.
     * 
     * @param array IArray object
     * @return new array
     * @throws ShapeNotMatchException
     *             mismatching shape
     */
    @Override
    public IArrayMath toEltDivide(final IArray newArray)
            throws ShapeNotMatchException {
        getArray().getArrayUtils().checkShape(newArray);
        IArray result = getFactory().createArray(Double.TYPE,
                getArray().getShape());
        if (getArray().getRank() == newArray.getRank()) {
            eltDivideWithEqualSize(newArray, result);
        } else {
            ISliceIterator sourceSliceIterator = null;
            ISliceIterator resultSliceIterator = null;
            try {
                sourceSliceIterator = getArray().getSliceIterator(
                        newArray.getRank());
                resultSliceIterator = result.getSliceIterator(newArray
                        .getRank());
                while (sourceSliceIterator.hasNext()
                        && resultSliceIterator.hasNext()) {
                    IArray sourceSlice = sourceSliceIterator.getArrayNext();
                    IArray resultSlice = resultSliceIterator.getArrayNext();
                    sourceSlice.getArrayMath().eltDivideWithEqualSize(newArray,
                            resultSlice);
                }
            } catch (InvalidRangeException e) {
                throw new ShapeNotMatchException("shape is invalid");
            }
        }
        return result.getArrayMath();
    }

    @Override
    public IArrayMath toEltDivide(final IArrayMath array)
            throws ShapeNotMatchException {
        return toEltDivide(array.getArray());
    }

    /**
     * Element wise divide the array1 value by value from given array2.
     * 
     * @param array IArray object
     * @return this array1 after modification
     * @throws ShapeNotMatchException
     *             mismatching shape
     */
    @Override
    public IArrayMath eltDivide(final IArray newArray) throws ShapeNotMatchException {
        getArray().getArrayUtils().checkShape(newArray);
        if (getArray().getRank() == newArray.getRank()) {
            eltDivideWithEqualSize(newArray, getArray());
        } else {
            ISliceIterator sourceSliceIterator = null;
            try {
                sourceSliceIterator = getArray().getSliceIterator(
                        newArray.getRank());
                while (sourceSliceIterator.hasNext()) {
                    IArray sourceSlice = sourceSliceIterator.getArrayNext();
                    sourceSlice.getArrayMath().eltDivideWithEqualSize(newArray,
                            sourceSlice);
                }
            } catch (InvalidRangeException e) {
                throw new ShapeNotMatchException("shape is invalid");
            }
        }
        getArray().setDirty(true);
        return this;
    }

    @Override
    public IArrayMath eltDivide(final IArrayMath array) throws ShapeNotMatchException {
        return eltDivide(array.getArray());
    }

    @Override
    public IFactory getFactory() {
        return factory;
    }

    @Override
    public double getMaximum() {
        IArrayIterator iter = getArray().getIterator();
        double max = -Double.MAX_VALUE;
        while (iter.hasNext()) {
            double val = iter.getDoubleNext();
            if (Double.isNaN(val))
                continue;
            if (val > max)
                max = val;
        }
        return max;
    }

    @Override
    public double getMinimum() {
        IArrayIterator iter = getArray().getIterator();
        double min = Double.MAX_VALUE;
        while (iter.hasNext()) {
            double val = iter.getDoubleNext();
            if (Double.isNaN(val))
                continue;
            if (val < min)
                min = val;
        }
        return min;
    }
}

/// @endcond internal
