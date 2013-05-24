package org.cdma.plugin.edf.utils;

import java.util.List;

import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IRange;
import org.cdma.utils.ArrayUtils;
import org.cdma.utils.IArrayUtils;

public class EdfArrayUtils extends ArrayUtils {

    public EdfArrayUtils(IArray array) {
        super(array);
    }

    @Override
    public IArrayUtils createArrayUtils(IArray array) {
        return new EdfArrayUtils(array);
    }

    @Override
    public Object get1DJavaArray(Class<?> wantType) {
        throw new NotImplementedException();
    }

    @Override
    public IArrayUtils sectionNoReduce(List<IRange> ranges) throws InvalidRangeException {
        throw new NotImplementedException();
    }

    @Override
    public IArrayUtils transpose(int dim1, int dim2) {
        throw new NotImplementedException();
    }

    @Override
    public boolean isConformable(IArray array) {
        throw new NotImplementedException();
    }

    @Override
    public IArrayUtils flip(int dim) {
        throw new NotImplementedException();
    }

    @Override
    public IArrayUtils permute(int[] dims) {
        throw new NotImplementedException();
    }

}
