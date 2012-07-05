package org.cdma.plugin.soleil.utils;

import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.math.ArrayMath;
import org.cdma.math.IArrayMath;
import org.cdma.plugin.soleil.NxsFactory;

public final class NxsArrayMath extends ArrayMath {

    public NxsArrayMath(IArray array) {
        super(array, NxsFactory.getInstance());
    }

    @Override
    public IArrayMath toAdd(IArray array) throws ShapeNotMatchException {
        throw new UnsupportedOperationException();
    }

    @Override
    public IArrayMath add(IArray array) throws ShapeNotMatchException {
        throw new UnsupportedOperationException();
    }

    @Override
    public IArrayMath matMultiply(IArray array) throws ShapeNotMatchException {
        throw new UnsupportedOperationException();
    }

    @Override
    public IArrayMath matInverse() throws ShapeNotMatchException {
        throw new UnsupportedOperationException();
    }

    @Override
    public IArrayMath sumForDimension(int dimension, boolean isVariance)
            throws ShapeNotMatchException {
        throw new UnsupportedOperationException();
    }

    @Override
    public IArrayMath enclosedSumForDimension(int dimension, boolean isVariance)
            throws ShapeNotMatchException {
        throw new UnsupportedOperationException();
    }

    @Override
    public double getNorm() {
        throw new UnsupportedOperationException();
    }

    @Override
    public IArrayMath normalise() {
        throw new UnsupportedOperationException();
    }

    @Override
    public double getDeterminant() throws ShapeNotMatchException {
        throw new UnsupportedOperationException();
    }

}
