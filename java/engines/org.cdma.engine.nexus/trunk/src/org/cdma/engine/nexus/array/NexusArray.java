/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.engine.nexus.array;

import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.arrays.DefaultArrayInline;
import org.cdma.arrays.DefaultIndex;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IArray;

import fr.soleil.nexus.DataItem;

public final class NexusArray extends DefaultArrayInline {
    private final DataItem mN4TDataItem;

    public NexusArray(NexusArray array) throws InvalidArrayTypeException {
        super(array);
        mN4TDataItem = array.mN4TDataItem;
    }

    public NexusArray(String factoryName, Object inlineArray, int[] iShape) throws InvalidArrayTypeException {
        super(factoryName, inlineArray, iShape);
        mN4TDataItem = null;
    }

    public NexusArray(String mFactory, DataItem mn4tDataItem) throws InvalidArrayTypeException {
        super(mFactory, mn4tDataItem.getDataClass(), mn4tDataItem.getSize());
        mN4TDataItem = mn4tDataItem;
    }

    @Override
    public IArray copy(boolean data) {
        NexusArray result;
        try {
            result = new NexusArray(this);
        } catch (InvalidArrayTypeException e) {
            result = null;
            Factory.getLogger().log(Level.SEVERE, "Unable to copy the NeXus array: " + this, e);
        }
        return result;
    }

    /**
     * Override of the super method: the loading of data is managed by the NeXus data item
     * The data is softly referenced and no loading is necessary.
     * In case of small data (String values), there are no soft reference, thus we return what
     * the storage of inherited DefaultArray.
     */
    @Override
    protected Object loadData() {
        return super.getData();
    }

    /**
     * Ask the underlying data item (NeXus) the portion of the corresponding view.
     * The access is done through a soft reference which is reloaded if lost or if the
     * visible part of the item has changed.
     */
    @Override
    protected Object getData() {
        Object result = loadData();
        if (result == null && mN4TDataItem != null) {
            DefaultIndex index = getIndex();
            result = mN4TDataItem.getData(index.getProjectionOrigin(), index.getProjectionShape());
        }
        return result;
    }

    public DataItem getDataItem() {
        return mN4TDataItem;
    }
}
