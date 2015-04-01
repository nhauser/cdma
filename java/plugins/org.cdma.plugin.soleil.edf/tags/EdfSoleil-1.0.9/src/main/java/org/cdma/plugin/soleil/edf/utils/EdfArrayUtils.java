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
package org.cdma.plugin.soleil.edf.utils;

import java.util.List;

import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IRange;
import org.cdma.utils.ArrayUtils;
import org.cdma.utils.IArrayUtils;

public class EdfArrayUtils extends ArrayUtils {

    public EdfArrayUtils(IArray array) {
        super(array);
    }

    @Override
    public IArrayUtils createArrayUtils(IArray array) {
        return new EdfArrayUtils(array);
    }

    @Override
    public Object get1DJavaArray(Class<?> wantType) {
        throw new NotImplementedException();
    }

    @Override
    public IArrayUtils sectionNoReduce(List<IRange> ranges) throws InvalidRangeException {
        throw new NotImplementedException();
    }

    @Override
    public IArrayUtils transpose(int dim1, int dim2) {
        throw new NotImplementedException();
    }

    @Override
    public boolean isConformable(IArray array) {
        throw new NotImplementedException();
    }

    @Override
    public IArrayUtils flip(int dim) {
        throw new NotImplementedException();
    }

    @Override
    public IArrayUtils permute(int[] dims) {
        throw new NotImplementedException();
    }

}
