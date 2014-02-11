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
package org.cdma.plugin.xml.array;

import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;

public class XmlAttribute implements IAttribute {

    private final String mFactoryName;
    private final String mName;
    private String mValue;
    private IArray mArrayValue;

    public XmlAttribute(String factory, String name, String value) {
        mFactoryName = factory;
        mName = name;
        mValue = value;
        mArrayValue = null;
    }

    @Override
    public String getFactoryName() {
        return mFactoryName;
    }

    @Override
    public String getName() {
        return mName;
    }

    @Override
    public Class<?> getType() {
        if (mArrayValue == null) {
            return String.class;
        }
        return null;
    }

    @Override
    public boolean isString() {
        if (mArrayValue == null) {
            return true;
        }
        return false;
    }

    @Override
    public boolean isArray() {
        if (mArrayValue != null) {
            return true;
        }
        return false;
    }

    @Override
    public int getLength() {
        if (mArrayValue == null) {
            return 1;
        }
        return 0;
    }

    @Override
    public IArray getValue() {
        return mArrayValue;
    }

    @Override
    public String getStringValue() {
        return mValue;
    }

    @Override
    public String getStringValue(int index) {
        return mValue;
    }

    @Override
    public Number getNumericValue() {
        Double value = null;
        try {
            value = Double.parseDouble(mValue);
        } catch (NumberFormatException e) {
            // nothing to do
        } catch (Exception e) {
            e.printStackTrace();
        }
        return value;
    }

    @Override
    public Number getNumericValue(int index) {
        return getNumericValue();
    }

    @Override
    public void setStringValue(String val) {
        mValue = val;
    }

    @Override
    public void setValue(IArray value) {
        mArrayValue = value;
    }

    @Override
    public String toString() {
        return mName + "=" + mValue;
    }
}
