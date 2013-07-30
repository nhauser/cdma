package org.cdma.plugin.soleil.edf.navigation;

import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.arrays.DefaultArrayMatrix;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.plugin.soleil.edf.EdfFactory;

public class EdfAttribute implements IAttribute {

    private final String factoryName; // factory name that attribute depends on
    private final String name;
    private IArray value;

    public EdfAttribute(String name, Object value) {
        this.factoryName = EdfFactory.NAME;
        this.name = name;

        Object data = value;
        if (value != null) {
            if (value.getClass().isArray()) {
                if (data instanceof char[]) {
                    data = new String[] { new String((char[]) value) };
                }
            }
            else {
                data = java.lang.reflect.Array.newInstance(value.getClass(), 1);
                java.lang.reflect.Array.set(data, 0, value);
            }

            try {
                this.value = new DefaultArrayMatrix(factoryName, data);
                this.value.lock();

            }

            catch (InvalidArrayTypeException e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
            }
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
            }
            else {
                result = getStringValue(0);
            }
        }
        else {
            result = getNumericValue().toString();
        }
        return result;
    }

    @Override
    public String getStringValue(int index) {
        if (isString()) {
            Object data = value.getStorage();
            return ((String[]) data)[0];
            // return ((String) java.lang.reflect.Array.get(value.getStorage(), index));
        }
        else {
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
    public Number getNumericValue(int index) {
        Object localValue;
        if (isArray()) {
            localValue = java.lang.reflect.Array.get(value.getArrayUtils().copyTo1DJavaArray(),
                    index);
        }
        else {
            localValue = java.lang.reflect.Array.get(value.getStorage(), index);
        }

        if (isString()) {
            localValue = Double.parseDouble((String) localValue);
        }

        return (Number) localValue;
    }

    @Override
    public void setStringValue(String val) {
        try {
            value = new DefaultArrayMatrix(factoryName, new String[] { val });
        }
        catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        }
    }

    @Override
    public void setValue(IArray value) {
        this.value = value;
    }

}

