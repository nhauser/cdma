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
    public IArrayMath sumForDimension(int dimension, boolean isVariance) throws ShapeNotMatchException {
        throw new UnsupportedOperationException();
    }

    @Override
    public IArrayMath enclosedSumForDimension(int dimension, boolean isVariance) throws ShapeNotMatchException {
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
