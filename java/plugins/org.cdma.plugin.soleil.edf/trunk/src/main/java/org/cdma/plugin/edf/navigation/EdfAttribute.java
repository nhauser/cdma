package org.cdma.plugin.edf.navigation;

import org.cdma.plugin.edf.abstraction.AbstractAttribute;
import org.cdma.plugin.edf.array.BasicArray;

public class EdfAttribute extends AbstractAttribute {

    public EdfAttribute(String name, Object value) {
        super(name);
        int i = 1;
        if (value.getClass().isArray()) {
            i = java.lang.reflect.Array.getLength(value);
        }
        setValue(new BasicArray(value, new int[] { i }));
    }

    @Override
    public String getStringValue(int index) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Number getNumericValue() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Number getNumericValue(int index) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void setStringValue(String val) {
        // TODO Auto-generated method stub

    }

    @Override
    public String getFactoryName() {
        // TODO Auto-generated method stub
        return null;
    }

}
