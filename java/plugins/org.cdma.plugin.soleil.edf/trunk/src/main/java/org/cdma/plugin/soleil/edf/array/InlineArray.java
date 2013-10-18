package org.cdma.plugin.soleil.edf.array;

import org.cdma.arrays.DefaultArrayInline;
import org.cdma.exception.InvalidArrayTypeException;

public class InlineArray extends DefaultArrayInline{

    public InlineArray(String factoryName, Object inlineArray, int[] iShape) throws InvalidArrayTypeException {
        super(factoryName, inlineArray, iShape);
       
    }

}
