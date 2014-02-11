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

import org.cdma.arrays.DefaultArrayInline;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IIndex;

public class XmlArray extends DefaultArrayInline {

    public XmlArray(String factory, String value) throws InvalidArrayTypeException {
        super(factory, new String[] { value }, new int[1]);
    }

    @Override
    public boolean getBoolean(IIndex ima) {
        boolean result = false;
        try {
            result = Boolean.parseBoolean((String) getData());
        } catch (NumberFormatException e) {
            // nothing to do
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public byte getByte(IIndex ima) {
        byte result = 0;
        try {
            result = Byte.parseByte((String) getData());
        } catch (NumberFormatException e) {
            // nothing to do
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public char getChar(IIndex ima) {
        char result = 0;
        String value = (String) getData();
        try {
            if (value.length() == 1) {
                result = value.charAt(0);
            }
        } catch (IndexOutOfBoundsException e) {
            // nothing to be done
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public double getDouble(IIndex ima) {
        double result = 0.0;
        try {
            result = Double.parseDouble((String) getData());
        } catch (NumberFormatException e) {
            // nothing to do
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public float getFloat(IIndex ima) {
        float result = 0;
        try {
            result = Float.parseFloat((String) getData());
        } catch (NumberFormatException e) {
            // nothing to do
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public int getInt(IIndex ima) {
        int result = 0;
        try {
            result = Integer.parseInt((String) getData());
        } catch (NumberFormatException e) {
            // nothing to do
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public long getLong(IIndex ima) {
        long result = 0;
        try {
            result = Long.parseLong((String) getData());
        } catch (NumberFormatException e) {
            // nothing to do
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public short getShort(IIndex ima) {
        short result = 0;
        try {
            result = Short.parseShort((String) getData());
        } catch (NumberFormatException e) {
            // nothing to do
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

}
