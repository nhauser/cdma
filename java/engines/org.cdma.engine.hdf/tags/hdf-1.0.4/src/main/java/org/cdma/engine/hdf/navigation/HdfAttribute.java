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
package org.cdma.engine.hdf.navigation;

import java.util.logging.Level;

import ncsa.hdf.object.Attribute;
import ncsa.hdf.object.Datatype;
import ncsa.hdf.object.HObject;
import ncsa.hdf.object.h5.H5Datatype;

import org.cdma.Factory;
import org.cdma.engine.hdf.array.HdfArray;
import org.cdma.engine.hdf.utils.HdfObjectUtils;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;

public class HdfAttribute implements IAttribute {

    private final String factoryName; // factory name that attribute depends on
    private IArray value; // Attribute's value
    private final String name;
    private boolean dirty;

    public HdfAttribute(final String factoryName, final Attribute attribute) {
        this(factoryName, attribute.getName(), attribute.getValue());
        // This constructor is used when data is read on from files
        this.dirty = false;
    }

    public HdfAttribute(final String factoryName, final String name, final Object value) {
        this.factoryName = factoryName;
        this.name = name;
        int[] shape;
        Object data = value;

        if (value.getClass().isArray()) {
            if (data instanceof char[]) {
                data = new String[] { new String((char[]) value) };
            }
            shape = new int[] { java.lang.reflect.Array.getLength(data) };
        } else {
            shape = new int[] {};
            data = java.lang.reflect.Array.newInstance(value.getClass(), 1);
            java.lang.reflect.Array.set(data, 0, value);
        }

        try {
            this.value = new HdfArray(factoryName, data, shape);
            this.value.lock();
            this.dirty = true;
        } catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        }
    }

    @Override
    public String getFactoryName() {
        return factoryName;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public Class<?> getType() {
        Class<?> result = null;
        if (value != null) {
            result = value.getElementType();
        }
        return result;
    }

    @Override
    public boolean isString() {
        boolean result = false;
        if (value != null) {
            Class<?> container = value.getElementType();
            result = (container.equals(Character.TYPE) || container.equals(String.class));
        }
        return result;
    }

    @Override
    public boolean isArray() {
        return value.getSize() > 1;
    }

    @Override
    public int getLength() {
        Long length = value.getSize();
        return length.intValue();
    }

    @Override
    public IArray getValue() {
        return value;
    }

    @Override
    public String getStringValue() {
        String result;
        if (isString()) {
            if (Character.TYPE.equals(getType())) {
                result = new String((char[]) value.getStorage());
            } else {
                result = getStringValue(0);
            }
        } else {
            result = getNumericValue().toString();
        }
        return result;
    }

    @Override
    public String getStringValue(final int index) {
        if (isString()) {
            Object data = value.getStorage();
            return ((String[]) data)[0];
            // return ((String) java.lang.reflect.Array.get(value.getStorage(), index));
        } else {
            return null;
        }
    }

    @Override
    public Number getNumericValue() {
        Number result = null;
        if (!isString()) {
            result = getNumericValue(0);
        }
        return result;
    }

    @Override
    public Number getNumericValue(final int index) {
        Object localValue;
        if (isArray()) {
            localValue = java.lang.reflect.Array.get(value.getArrayUtils().copyTo1DJavaArray(), index);
        } else {
            localValue = java.lang.reflect.Array.get(value.getStorage(), index);
        }

        if (isString()) {
            localValue = Double.parseDouble((String) localValue);
        }

        return (Number) localValue;
    }

    @Override
    public void setStringValue(final String val) {
        try {
            value = new HdfArray(factoryName, new String[] { val }, new int[] { 1 });
            dirty = true;
        } catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        }
    }

    @Override
    public void setValue(final IArray value) {
        this.value = value;
        this.dirty = true;
    }

    public void save(final HObject parent) throws Exception {

        H5Datatype dataType;
        if (dirty) {
            Object valueObject = getValue().getStorage();
            if (valueObject instanceof String[]) {
                String[] strArray = (String[]) valueObject;
                // valueObject = strArray[0];
                dataType = new H5Datatype(Datatype.CLASS_STRING, strArray[0].length() + 1, -1, -1);
            } else {
                int type_id = HdfObjectUtils.getNativeHdfDataTypeForClass(value.getElementType());
                dataType = new H5Datatype(type_id);
            }

            Attribute newAttribute = new Attribute(getName(), dataType, null, valueObject);
            parent.writeMetadata(newAttribute);
        }
        dirty = false;
    }
}
