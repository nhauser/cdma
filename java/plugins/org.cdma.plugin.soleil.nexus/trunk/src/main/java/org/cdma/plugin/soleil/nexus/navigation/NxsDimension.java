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
package org.cdma.plugin.soleil.nexus.navigation;

import java.io.IOException;

import org.cdma.exception.NotImplementedException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDimension;

public final class NxsDimension implements IDimension {

    private IArray mArray;
    private String mLongName;
    private boolean mIsVariableLength;
    private boolean mIsUnlimited;
    private boolean mIsShared;
    private final String mFactory;

    public NxsDimension(String factoryName, IDataItem item) {
        mFactory = factoryName;
        mLongName = item.getName();
        mIsUnlimited = item.isUnlimited();
        try {
            mArray = item.getData();
        } catch (IOException e) {
            mArray = null;
        }
    }

    public NxsDimension(NxsDimension dim) {
        mFactory = dim.mFactory;
        mLongName = dim.mLongName;
        mArray = dim.mArray;
        mIsVariableLength = dim.mIsVariableLength;
        mIsUnlimited = dim.mIsUnlimited;
    }

    @Override
    public int compareTo(Object o) {
        if (this.equals(o)) {
            return 0;
        } else {
            IDimension dim = (IDimension) o;
            return mLongName.compareTo(dim.getName());
        }
    }

    @Override
    public boolean equals(Object dim) {
        boolean result;
        if (dim instanceof IDimension) {
            result = mLongName.equals(((IDimension) dim).getName());
        } else {
            result = false;
        }
        return result;
    }

    @Override
    public int hashCode() {
        return mLongName.hashCode();
    }

    @Override
    public IArray getCoordinateVariable() {
        return mArray;
    }

    @Override
    public int getLength() {
        return Long.valueOf(mArray.getSize()).intValue();
    }

    @Override
    public String getName() {
        return mLongName;
    }

    @Override
    public boolean isShared() {
        return mIsShared;
    }

    @Override
    public boolean isUnlimited() {
        return mIsUnlimited;
    }

    @Override
    public boolean isVariableLength() {
        return mIsVariableLength;
    }

    @Override
    public void setLength(int n) {
        try {
            mArray.getArrayUtils().reshape(new int[] { n });
        } catch (ShapeNotMatchException e) {
        }
    }

    @Override
    public void setName(String name) {
        mLongName = name;
    }

    @Override
    public void setShared(boolean b) {
        mIsShared = b;
    }

    @Override
    public void setUnlimited(boolean b) {
        mIsUnlimited = b;
    }

    @Override
    public void setVariableLength(boolean b) {
        mIsVariableLength = b;
    }

    @Override
    public void setCoordinateVariable(IArray array) throws ShapeNotMatchException {
        if (java.util.Arrays.equals(mArray.getShape(), array.getShape())) {
            throw new ShapeNotMatchException("Arrays must have same shape!");
        }
        mArray = array;
    }

    @Override
    public String writeCDL(boolean strict) {
        throw new NotImplementedException();
    }

    @Override
    public String getFactoryName() {
        return mFactory;
    }
}
