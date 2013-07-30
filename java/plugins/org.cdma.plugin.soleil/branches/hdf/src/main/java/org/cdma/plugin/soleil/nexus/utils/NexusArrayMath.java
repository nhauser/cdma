//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.soleil.nexus.utils;

import org.cdma.Factory;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.math.ArrayMath;
import org.cdma.math.IArrayMath;

public final class NexusArrayMath extends ArrayMath {

    public NexusArrayMath(IArray array) {
	super(array, Factory.getFactory(array.getFactoryName()));
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
