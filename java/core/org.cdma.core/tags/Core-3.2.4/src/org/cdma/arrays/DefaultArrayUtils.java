// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr)
// ****************************************************************************
package org.cdma.arrays;

import java.util.List;

import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.IRange;
import org.cdma.utils.ArrayUtils;
import org.cdma.utils.IArrayUtils;

public class DefaultArrayUtils extends ArrayUtils {

    public DefaultArrayUtils(DefaultArray array) {
        super(array);
    }

    @Override
    public Object copyTo1DJavaArray() {
    	DefaultArray array = (DefaultArray) getArray();

        return array.copyTo1DJavaArray();
    }
    
    @Override
    public Object copyToNDJavaArray() {
    	DefaultArray array = (DefaultArray) getArray();

        return array.copyToNDJavaArray();
    }

    @Override
    public IArrayUtils flip(int dim) {
        IArray array = getArray().copy(false);
        IIndex index = array.getIndex();
        int rank = array.getRank();
        int[] shape = index.getShape();
        int[] origin = index.getOrigin();
        int[] position = index.getCurrentCounter();
        long[] stride = index.getStride();

        int[] newShape = new int[rank];
        int[] newOrigin = new int[rank];
        int[] newPosition = new int[rank];
        long[] newStride = new long[rank];

        for (int i = 0; i < rank; i++) {
            newShape[i] = shape[rank - 1 - i];
            newOrigin[i] = origin[rank - 1 - i];
            newStride[i] = stride[rank - 1 - i];
            newPosition[i] = position[rank - 1 - i];
        }

        index = new DefaultIndex(array.getFactoryName(), newShape, newOrigin, newShape);
        index.setStride(newStride);
        index.set(newPosition);
        array.setIndex(index);
        return array.getArrayUtils();
    }

    @Override
    public IArrayUtils permute(int[] dims) {
        IArray array = getArray().copy(false);
        IIndex index = array.getIndex();
        int rank = array.getRank();
        int[] shape = index.getShape();
        int[] origin = index.getOrigin();
        int[] position = index.getCurrentCounter();
        long[] stride = index.getStride();
        int[] newShape = new int[rank];
        int[] newOrigin = new int[rank];
        int[] newPosition = new int[rank];
        long[] newStride = new long[rank];
        for (int i = 0; i < rank; i++) {
            newShape[i] = shape[dims[i]];
            newOrigin[i] = origin[dims[i]];
            newStride[i] = stride[dims[i]];
            newPosition[i] = position[dims[i]];
        }

        index = new DefaultIndex(array.getFactoryName(), newShape, newOrigin, newShape);
        index.setStride(newStride);
        index.set(newPosition);
        array.setIndex(index);
        return array.getArrayUtils();
    }

    @Override
    public IArrayUtils transpose(int dim1, int dim2) {
        IArray array = getArray().copy(false);
        IIndex index = array.getIndex();
        int[] shape = index.getShape();
        int[] origin = index.getOrigin();
        int[] position = index.getCurrentCounter();
        long[] stride = index.getStride();

        int sha = shape[dim1];
        int ori = origin[dim1];
        int pos = position[dim1];
        long str = stride[dim1];

        shape[dim2] = shape[dim1];
        origin[dim2] = origin[dim1];
        stride[dim2] = stride[dim1];
        position[dim2] = position[dim1];

        shape[dim2] = sha;
        origin[dim2] = ori;
        stride[dim2] = str;
        position[dim2] = pos;

        index = new DefaultIndex(array.getFactoryName(), shape, origin, shape);
        index.setStride(stride);
        index.set(position);
        array.setIndex(index);
        return array.getArrayUtils();
    }

    @Override
    public IArrayUtils sectionNoReduce(List<IRange> ranges) throws InvalidRangeException {
        throw new NotImplementedException();
    }

    @Override
    public IArrayUtils createArrayUtils(IArray array) {
        IArrayUtils result;
        if (array instanceof DefaultArray) {
            result = new DefaultArrayUtils((DefaultArray) array);
        } else {
            result = array.getArrayUtils();
        }
        return result;
    }

    @Override
    public Object get1DJavaArray(Class<?> wantType) {
        throw new NotImplementedException();
    }

    @Override
    public boolean isConformable(IArray array) {
        boolean result = false;
        if (array.getRank() == getArray().getRank()) {
            IArray copy1 = this.reduce().getArray();
            IArray copy2 = array.getArrayUtils().reduce().getArray();

            int[] shape1 = copy1.getShape();
            int[] shape2 = copy2.getShape();

            for (int i = 0; i < shape1.length; i++) {
                if (shape1[i] != shape2[i]) {
                    result = false;
                    break;
                }
            }

        }
        return result;
    }
}

