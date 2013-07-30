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


    public HdfAttribute(String factoryName, Attribute attribute) {
        this(factoryName, attribute.getName(), attribute.getValue());
    }

    public HdfAttribute(String factoryName, String name, Object value) {
        this.factoryName = factoryName;
        this.name = name;
        int[] shape;
        Object data = value;

        if (value.getClass().isArray()) {
            if (data instanceof char[]) {
                data = new String[] { new String((char[]) value) };
            }
            shape = new int[] { java.lang.reflect.Array.getLength(data) };
        }
        else {
            shape = new int[] {};
            data = java.lang.reflect.Array.newInstance(value.getClass(), 1);
            java.lang.reflect.Array.set(data, 0, value);
        }

        try {
            this.value = new HdfArray(factoryName, data, shape, null);
            this.value.lock();
        }
        catch (InvalidArrayTypeException e) {
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
            value = new HdfArray(factoryName, new String[] { val }, new int[] { 1 }, null);
        }
        catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
        }
    }

    @Override
    public void setValue(IArray value) {
        this.value = value;
    }

    public void save(HObject parent) throws Exception {
        int dataType = HdfObjectUtils.getHdfDataTypeForClass(getType());
        Attribute newAttribute = new Attribute(getName(), new H5Datatype(dataType), null,
                getValue().getStorage());

        if (getValue().getStorage() instanceof String[]) {
            String[] value = (String[]) getValue().getStorage();
            Datatype newType = new H5Datatype(Datatype.CLASS_STRING, value[0].length() + 1, -1, -1);
            // Create a new attribute
            newAttribute = new Attribute(getName(), newType, null, getValue().getStorage());

        }
        // parent.writeMetadata(newAttribute);
    }
}
