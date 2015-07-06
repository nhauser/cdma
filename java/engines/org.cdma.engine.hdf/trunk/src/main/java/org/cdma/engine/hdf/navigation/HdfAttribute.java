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

import java.lang.reflect.Array;
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
import org.cdma.utils.ArrayTools;

public class HdfAttribute implements IAttribute {

    private final String factoryName; // factory name that attribute depends on
    private final String name;
    private boolean dirty;
    private Object value;

    public HdfAttribute(final String factoryName, final Attribute attribute) {
        this(factoryName, attribute.getName(), attribute.getValue());
        // This constructor is used when data is read on from files
        this.dirty = false;
    }

    public HdfAttribute(final String factoryName, final String name, final Object value) {
        // This constructor is used when data is supplied by the API user
        this.factoryName = factoryName;
        this.name = name;
        this.value = value;

        if (value.getClass().isArray()) {
            if (value instanceof char[]) {
                this.value = new String[] { new String((char[]) value) };
            }
        } else {
            this.value = java.lang.reflect.Array.newInstance(value.getClass(), 1);
            java.lang.reflect.Array.set(this.value, 0, value);
        }

        this.dirty = true;

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
        if (value != null && value.getClass().isArray()) {
            result = value.getClass().getComponentType();
        }
        return result;
    }

    @Override
    public boolean isString() {
        boolean result = false;
        if (value != null && value.getClass().isArray()) {
            Class<?> container = value.getClass().getComponentType();
            result = (container.equals(Character.TYPE) || container.equals(String.class));
        }
        return result;
    }

    @Override
    public boolean isArray() {
        return value.getClass().isArray();
    }

    @Override
    public int getLength() {
        int result = Array.getLength(value);
        return result;
    }

    @Override
    public IArray getValue() {
        IArray result = null;
        try {
            result = new HdfArray(getFactoryName(), value, ArrayTools.detectShape(value));
        } catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to load attribute value", e);
        }
        return result;
    }

    @Override
    public String getStringValue() {
        String result;
        if (isString()) {
            if (Character.TYPE.equals(getType())) {
                result = new String((char[]) value);
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
            return ((String[]) value)[0];
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
        localValue = java.lang.reflect.Array.get(value, index);

        if (isString()) {
            localValue = Double.parseDouble((String) localValue);
        }

        return (Number) localValue;
    }

    @Override
    public void setStringValue(final String val) {
        value = new String[] { val };
        dirty = true;
    }

    @Override
    public void setValue(final IArray value) {
        this.value = value.getStorage();
        this.dirty = true;
    }

    public void save(final HObject parent, boolean forceSave) throws Exception {
        save(parent, false, forceSave);
    }

    public void save(final HObject parent, boolean isLink, boolean forceSave) throws Exception {
        if (dirty || forceSave) {
            H5Datatype dataType;

            if (value instanceof String[]) {
                String[] strArray = (String[]) value;
                String valueObject = strArray[0];
                dataType = new H5Datatype(Datatype.CLASS_STRING, valueObject.length() + 1, -1, -1);
            } else {
                int type_id = HdfObjectUtils.getNativeHdfDataTypeForClass(value.getClass().getComponentType());
                dataType = new H5Datatype(type_id);
            }

            Attribute attr = HdfObjectUtils.getAttribute(parent, getName());

            if (isLink && attr != null) {
                return;
            }

            if (attr == null) {
                attr = new Attribute(getName(), dataType, null, value);
            } else {
                attr.setValue(value);
            }

            parent.writeMetadata(attr);
        }
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((name == null) ? 0 : name.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        HdfAttribute other = (HdfAttribute) obj;
        if (name == null) {
            if (other.name != null)
                return false;
        } else if (!name.equals(other.name))
            return false;
        return true;
    }
}
