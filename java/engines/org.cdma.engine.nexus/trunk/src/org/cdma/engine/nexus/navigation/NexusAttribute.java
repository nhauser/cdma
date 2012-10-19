//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.engine.nexus.navigation;

import org.cdma.engine.nexus.array.NexusArray;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;

public final class NexusAttribute implements IAttribute {

    /// Members
    private String  mName;    // Attribute's name
    private IArray  mValue;   // Attribute's value
    private String  mFactory; // factory name that attribute depends on

    /// Constructors
    public NexusAttribute(String factoryName, String name, String value) {
    	this( factoryName, name, value.toCharArray());
    }
    
    public NexusAttribute(String factoryName, String name, Object value) {
        int[] shape;
        if (value.getClass().isArray()) {
        	shape = new int[] { java.lang.reflect.Array.getLength(value) };
        }
        else {
        	shape = new int[] {};
        }
        mFactory = factoryName;
        mName = name;
        mValue = new NexusArray(mFactory, value, shape);
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
        if (isString()) {
            return null;
        }

        if (isArray()) {
            return getNumericValue(0);
        } else {
            return (Number) mValue.getStorage();

        }
    }

    @Override
    public Number getNumericValue(int index) {
        Object value;
        if (isArray()) {
            value = java.lang.reflect.Array.get(mValue.getStorage(), index);
        } else {
            value = mValue.getStorage();
        }

        if (isString()) {
            return (Double) value;
        }

        return (Number) value;
    }

    @Override
    public String getStringValue() {
        if (isString()) {
            return new String( (char[]) mValue.getStorage() );
        } else {
            return getNumericValue().toString();
        }
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
        return (container.equals( Character.TYPE ));
    }

    @Override
    public void setStringValue(String val) {
        mValue = new NexusArray(mFactory, val, new int[] { 1 });
    }

    @Override
    public void setValue(IArray value) {
        mValue = value;
    }

    public String toString() {
        return mName + "=" + mValue;
    }

    @Override
    public String getFactoryName() {
        return mFactory;
    }
}
