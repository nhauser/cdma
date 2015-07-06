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
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.engine.hdf.array;

import java.lang.ref.WeakReference;
import java.util.logging.Level;

import ncsa.hdf.object.h5.H5ScalarDS;

import org.cdma.Factory;
import org.cdma.arrays.DefaultArrayInline;
import org.cdma.arrays.DefaultIndex;
import org.cdma.engine.hdf.navigation.HdfDataItem;
import org.cdma.engine.hdf.utils.HdfObjectUtils;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IArray;

public class HdfArray extends DefaultArrayInline {

    private HdfDataItem dataItem;

    public HdfArray(String factoryName, Class<?> clazz, int[] iShape, HdfDataItem dataItem)
            throws InvalidArrayTypeException {
        super(factoryName, clazz, iShape);
        this.dataItem = dataItem;
    }

    public HdfArray(String factoryName, Object array, int[] iShape) throws InvalidArrayTypeException {
        super(factoryName, new WeakReference<Object>(array), iShape);
        this.lock();
    }

    public HdfArray(String factoryName, HdfDataItem dataItem) throws InvalidArrayTypeException {
        super(factoryName, dataItem.getType(), dataItem.getShape());
        this.dataItem = dataItem;
    }

    public HdfArray(HdfArray array) throws InvalidArrayTypeException {
        super(array);
        this.dataItem = array.dataItem;
    }

    public HdfArray(String factoryName, H5ScalarDS ds) throws OutOfMemoryError, Exception {
        super(factoryName, ds.getData(), HdfObjectUtils.convertLongToInt(ds.getDims()));
        this.dataItem = null;
    }

    @Override
    protected Object loadData() {
        Object data = null;
        WeakReference<Object> result = new WeakReference<Object>(data);

        if (dataItem != null) {
            DefaultIndex index = getIndex();
            try {
                data = dataItem.load(index.getProjectionOrigin(), index.getProjectionShape());
                result = new WeakReference<Object>(data);
            } catch (OutOfMemoryError e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to loadData()", e);
            } catch (Exception e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to loadData()", e);
            }
        }
        return result;
    }

    @Override
    public IArray copy(boolean data) {
        HdfArray result;
        try {
            result = new HdfArray(this);
        } catch (InvalidArrayTypeException e) {
            result = null;
            Factory.getLogger().log(Level.SEVERE, "Unable to copy the HdfArray array: " + this, e);
        }
        return result;
    }

    @Override
    protected Object getData() {
        WeakReference<Object> reference = (WeakReference<Object>) super.getData();

        if (reference == null) {
            reference = (WeakReference<Object>) loadData();
        }
        Object result = reference.get();

        if (result == null) {
            reference = (WeakReference<Object>) loadData();
        }

        if (reference != null) {
            result = reference.get();
        }
        return result;
    }

}
