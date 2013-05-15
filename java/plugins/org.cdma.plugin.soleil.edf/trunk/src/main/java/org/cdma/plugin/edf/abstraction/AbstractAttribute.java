package org.cdma.plugin.edf.abstraction;

import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.IAttribute;

public abstract class AbstractAttribute implements IAttribute {

    protected String name;
    protected IArray value;

    public AbstractAttribute(String name) {
        super();
        this.name = name;
        value = null;
    }

    @Override
    public int getLength() {
        if (value != null) {
            long size = value.getSize();
            if (size < 0) {
                size = 0;
            }
            else if (size > Integer.MAX_VALUE) {
                size = Integer.MAX_VALUE;
            }
            return (int) size;
        }
        return 0;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public Class<?> getType() {
        if (value != null) {
            return value.getElementType();
        }
        return null;
    }

    @Override
    public IArray getValue() {
        return value;
    }

    @Override
    public void setValue(IArray value) {
        this.value = value;
    }

    @Override
    public boolean isArray() {
        Class<?> type = getType();
        if (type != null) {
            return type.isArray();
        }
        return false;
    }

    @Override
    public boolean isString() {
        return String.class.isAssignableFrom(getType());
    }

    @Override
    public String getStringValue() {
        if (isString()) {
            StringBuffer buffer = new StringBuffer();
            if (value.getSize() > 0) {
                IArrayIterator iterator = value.getIterator();
                // buffer.append(iterator.getCharNext());
                while (iterator.hasNext()) {
                    buffer.append(iterator.getCharNext());
                }
                iterator = null;
            }
            return buffer.toString();
        }
        return null;
    }

}
