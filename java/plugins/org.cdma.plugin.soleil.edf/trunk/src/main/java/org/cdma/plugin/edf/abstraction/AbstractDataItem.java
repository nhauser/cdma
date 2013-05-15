package org.cdma.plugin.edf.abstraction;

import java.io.IOException;

import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.plugin.edf.utils.StringUtils;
import org.cdma.utils.Utilities.ModelType;

public abstract class AbstractDataItem extends AbstractObject implements IDataItem {

    protected IArray data;

    public AbstractDataItem() {
        super();
        data = null;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.DataItem;
    }

    @Override
    public IAttribute findAttributeIgnoreCase(String name) {
        if (attributes != null) {
            for (IAttribute attribute : attributes) {
                if (StringUtils.isSameStringIgnoreCase(name, attribute.getName())) {
                    return attribute;
                }
            }
        }
        return null;
    }

    @Override
    public IDataItem getASlice(int dimension, int value) throws InvalidRangeException {
        return getSlice(dimension, value);
    }

    @Override
    public IArray getData() throws IOException {
        return data;
    }

    @Override
    public IArray getData(int[] origin, int[] shape) throws IOException, InvalidRangeException {
        if (data != null) {
            return data.getArrayUtils().section(origin, shape).getArray();
        }
        return null;
    }

    @Override
    public IDataItem clone() {
        return (IDataItem) super.clone();
    }

}
