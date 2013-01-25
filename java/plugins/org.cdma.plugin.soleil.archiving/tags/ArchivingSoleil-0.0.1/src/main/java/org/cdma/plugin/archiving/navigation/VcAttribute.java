package org.cdma.plugin.archiving.navigation;


//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************

import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.arrays.DefaultArray;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.IAttribute;
import org.cdma.plugin.archiving.VcFactory;

public class VcAttribute implements IAttribute {

    /// Members
    private String  mName;    // Attribute's name
    private IArray  mValue;   // Attribute's value

    /// Constructors
    public VcAttribute(String name, char[] value) {
    	this( name, new String(value) );
    }
    
    public VcAttribute(String name, Object value) {
        int[] shape;
        Object data = value;
        
        if (value.getClass().isArray()) {
            if( data instanceof char[] ) {
            	data = new String[] {new String( (char[]) value)};
            }
        	shape = new int[] { java.lang.reflect.Array.getLength(data) };
        }
        else {
        	shape = new int[] {};
        	data = java.lang.reflect.Array.newInstance( value.getClass(), 1);
        	java.lang.reflect.Array.set(data, 0, value);
        }
        mName = name;
        try {
			mValue = DefaultArray.instantiateDefaultArray(VcFactory.NAME, data, shape);
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
        if (isString()) {
            return null;
        }

        return getNumericValue(0);
    }

    @Override
    public Number getNumericValue(int index) {
        Object value = null;
        IArrayIterator p = mValue.getIterator();
        int i = 0;
        while( i < index && p.hasNext() ) {
        	p.next();
        }
        if( p.hasNext() ) {
        	value = p.getObjectNext();
        }

        if (isString()) {
            value = Double.parseDouble( (String) value );
        }

        return (Number) value;
    }

    @Override
    public String getStringValue() {
    	String result;
        if (isString()) {
        	if( Character.TYPE.equals( getType() ) ) {
        		result = new String( (char[]) mValue.getStorage() );
        	}
        	else {
        		result = mValue.getStorage().toString();
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
        return (container.equals( Character.TYPE ) || container.equals( String.class ) );
    }

    @Override
    public void setStringValue(String val) {
        try {
        	mValue = DefaultArray.instantiateDefaultArray(VcFactory.NAME, val, new int[] { 1 });
		} catch (InvalidArrayTypeException e) {
        	Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
		}
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
        return VcFactory.NAME;
    }
}
