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
 *        St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.engine.hdf.array;

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

    public HdfArray(String factoryName, Object array, int[] iShape)
            throws InvalidArrayTypeException {
        super(factoryName, array, iShape);
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
        Object result = null;
        if (dataItem != null) {
            DefaultIndex index = getIndex();
            try {
                result = dataItem.load(index.getProjectionOrigin(), index.getProjectionShape());
            }
            catch (OutOfMemoryError e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to loadData()", e);
            }
            catch (Exception e) {
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
        }
        catch (InvalidArrayTypeException e) {
            result = null;
            Factory.getLogger().log(Level.SEVERE, "Unable to copy the HdfArray array: " + this, e);
        }
        return result;
    }

    @Override
    protected Object getData() {
        Object result = super.getData();
        if( result == null ) {
            result = loadData();
        }
        return result;
    }

}
