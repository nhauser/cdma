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
package org.cdma.engine.nexus.navigation;

import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.nexus.array.NexusArray;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;

public final class NexusAttribute implements IAttribute {

    // / Members
    private final String mName; // Attribute's name
    private IArray mValue; // Attribute's value
    private final String mFactory; // factory name that attribute depends on

    // / Constructors
    public NexusAttribute(String factoryName, String name, char[] value) {
        this(factoryName, name, new String(value));
    }

    public NexusAttribute(String factoryName, String name, Object value) {
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
        mFactory = factoryName;
        mName = name;
        try {
            mValue = new NexusArray(mFactory, data, shape);
        } catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        }
    }

    @Override
    public int getLength() {
        Long length = mValue.getSize();
        return length.intValue();
    }

    @Override
    public String getName() {
        return mName;
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
    public Number getNumericValue(int index) {
        Object value;
        if (isArray()) {
            value = java.lang.reflect.Array.get(mValue.getArrayUtils().copyTo1DJavaArray(), index);
        } else {
            value = java.lang.reflect.Array.get(mValue.getStorage(), index);
        }

        if (isString()) {
            value = Double.parseDouble((String) value);
        }

        return (Number) value;
    }

    @Override
    public String getStringValue() {
        String result;
        if (isString()) {
            if (Character.TYPE.equals(getType())) {
                result = new String((char[]) mValue.getStorage());
            } else {
                result = getStringValue(0);
            }
        } else {
            result = getNumericValue().toString();
        }
        return result;
    }

    @Override
    public String getStringValue(int index) {
        if (isString()) {
            return ((String) java.lang.reflect.Array.get(mValue.getStorage(), index));
        } else {
            return null;
        }
    }

    @Override
    public Class<?> getType() {
        return mValue.getElementType();
    }

    @Override
    public IArray getValue() {
        return mValue;
    }

    @Override
    public boolean isArray() {
        return mValue.getSize() > 1;
    }

    @Override
    public boolean isString() {
        Class<?> container = mValue.getElementType();
        return (container.equals(Character.TYPE) || container.equals(String.class));
    }

    @Override
    public void setStringValue(String val) {
        try {
            mValue = new NexusArray(mFactory, new String[] { val }, new int[] { 1 });
        } catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        }
    }

    @Override
    public void setValue(IArray value) {
        mValue = value;
    }

    @Override
    public String toString() {
        String result = mName + "=";
        result += isString() ? getStringValue() : getNumericValue();
        return result;
    }

    @Override
    public String getFactoryName() {
        return mFactory;
    }
}
