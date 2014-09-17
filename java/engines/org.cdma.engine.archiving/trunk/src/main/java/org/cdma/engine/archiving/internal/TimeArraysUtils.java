package org.cdma.engine.archiving.internal;

import java.util.List;

import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IRange;
import org.cdma.utils.IArrayUtils;

public class TimeArraysUtils implements IArrayUtils {

    private final IArrayUtils arrayUtils;

    public TimeArraysUtils(final IArrayUtils arrayUtils) {
        this.arrayUtils = arrayUtils;
    }
    @Override
    public IArray getArray() {
        return arrayUtils.getArray();
    }

    @Override
    public void copyTo(IArray newArray) throws ShapeNotMatchException {
        arrayUtils.copyTo(newArray);
    }

    @Override
    public Object copyTo1DJavaArray() {
        long[] timeValue = null;
        Object tab = arrayUtils.copyTo1DJavaArray();
        if(tab instanceof String[]) {
            String[] stringArray = (String[])tab;
            timeValue = new long[stringArray.length];
            String dateString = null;
            for (int i = 0; i < stringArray.length; i++) {
                dateString = stringArray[i];
                Long longObject = TimeArray.convertStringDateToMs(dateString);
                if (longObject != null) {
                    timeValue[i] = longObject.longValue();
                }
            }
        }
        return timeValue;
    }

    @Override
    public Object get1DJavaArray(Class<?> wantType) {
        return arrayUtils.get1DJavaArray(wantType);
    }

    @Override
    public Object copyToNDJavaArray() {
        return arrayUtils.copyToNDJavaArray();
    }

    @Override
    public void checkShape(IArray newArray) throws ShapeNotMatchException {
        arrayUtils.checkShape(newArray);
    }

    @Override
    public IArrayUtils concatenate(IArray array) throws ShapeNotMatchException {
        return arrayUtils.concatenate(array);
    }

    @Override
    public IArrayUtils reduce() {
        return arrayUtils.reduce();
    }

    @Override
    public IArrayUtils reduce(int dim) {
        return arrayUtils.reduce(dim);
    }

    @Override
    public IArrayUtils reduceTo(int rank) {
        return arrayUtils.reduceTo(rank);
    }

    @Override
    public IArrayUtils reshape(int[] shape) throws ShapeNotMatchException {
        return arrayUtils.reshape(shape);
    }

    @Override
    public IArrayUtils section(int[] origin, int[] shape) throws InvalidRangeException {
        return arrayUtils.section(origin, shape);
    }

    @Override
    public IArrayUtils section(int[] origin, int[] shape, long[] stride) throws InvalidRangeException {
        return arrayUtils.section(origin, shape, stride);
    }

    @Override
    public IArrayUtils sectionNoReduce(int[] origin, int[] shape, long[] stride) throws InvalidRangeException {
        return arrayUtils.sectionNoReduce(origin, shape, stride);
    }

    @Override
    public IArrayUtils sectionNoReduce(List<IRange> ranges) throws InvalidRangeException {
        return arrayUtils.sectionNoReduce(ranges);
    }

    @Override
    public IArrayUtils slice(int dim, int value) {
        return arrayUtils.slice(dim, value);
    }

    @Override
    public IArrayUtils transpose(int dim1, int dim2) {
        return arrayUtils.transpose(dim1, dim2);
    }

    @Override
    public boolean isConformable(IArray array) {
        return arrayUtils.isConformable(array);
    }

    @Override
    public IArrayUtils eltAnd(IArray booleanMap) throws ShapeNotMatchException {
        return arrayUtils.eltAnd(booleanMap);
    }

    @Override
    public IArrayUtils integrateDimension(int dimension, boolean isVariance) throws ShapeNotMatchException {
        return arrayUtils.integrateDimension(dimension, isVariance);
    }

    @Override
    public IArrayUtils enclosedIntegrateDimension(int dimension, boolean isVariance) throws ShapeNotMatchException {
        return arrayUtils.enclosedIntegrateDimension(dimension, isVariance);
    }

    @Override
    public IArrayUtils flip(int dim) {
        return arrayUtils.flip(dim);
    }

    @Override
    public IArrayUtils permute(int[] dims) {
        return arrayUtils.permute(dims);
    }

}
