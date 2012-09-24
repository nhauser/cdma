package org.cdma.engine.netcdf.utils;

import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.netcdf.array.NcArray;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IRange;

import org.cdma.utils.ArrayUtils;
import org.cdma.utils.IArrayUtils;


import ucar.ma2.MAMath;

public class NcArrayUtils extends ArrayUtils {

    public NcArrayUtils(NcArray array) {
        super(array);
    }

    protected NcArray getNcArray() {
    	return (NcArray) getArray();
    }

    @Override
    public boolean isConformable(IArray array) {
        return MAMath.conformable(getNcArray().getArray(), ((NcArray) array).getArray());
    }

    /**
     * @return IArray object
     * @see org.gumtree.data.interfaces.IArray#reduce()
     */
    @Override
    public IArrayUtils reduce() {
        return new NcArrayUtils(new NcArray(getNcArray().getArray().reduce(), getArray().getFactoryName()));
    }

    /**
     * @param dim
     *            integer value
     * @return IArray object
     * @see org.gumtree.data.interfaces.IArray#reduce(int)
     */
    @Override
    public IArrayUtils reduce(final int dim) {
        return new NcArrayUtils(new NcArray(getNcArray().getArray().reduce(dim), getArray().getFactoryName()));
    }


    /**
     * @param rank
     *            integer value
     * @return new Array object
     * @see org.gumtree.data.interfaces.IArray#reduceTo(int)
     */
    @Override
    public IArrayUtils reduceTo(final int rank) {
        NcArray result = getNcArray();
        int oldRank = getArray().getRank();
        if (oldRank <= rank) {
            return this;
        } else {
            int[] shape = getArray().getShape();
            for (int i = 0; i < shape.length; i++) {
                if (shape[i] == 1) {
                    NcArray reduced = (NcArray) reduce(i);
                    result = (NcArray) reduced.getArrayUtils().reduceTo(rank).getArray();
                }
            }
        }
        return new NcArrayUtils(result);
    }

    /**
     * @param shape
     *            array of integers
     * @return IArray object
     * @see org.gumtree.data.interfaces.IArray#reshape(int[])
     */
    @Override
    public IArrayUtils reshape(final int[] shape) {
        return new NcArrayUtils(new NcArray(getNcArray().getArray().reshape(shape), getArray().getFactoryName()));
    }
    
    /**
     * @param origin
     *            array of integers
     * @param shape
     *            array of integers
     * @return IArray object
     * @throws InvalidRangeException
     *             invalid range
     * @see org.gumtree.data.interfaces.IArray#section(int[], int[])
     */
    @Override
    public IArrayUtils section(final int[] origin, final int[] shape)
            throws InvalidRangeException {
        try {
            return new NcArrayUtils(new NcArray(getNcArray().getArray().section(origin, shape), getArray().getFactoryName()));
        } catch (ucar.ma2.InvalidRangeException e) {
            throw new InvalidRangeException(e);
        }
    }

    /**
     * @param origin
     *            array of integers
     * @param shape
     *            array of integers
     * @param stride
     *            array of integers
     * @return IArray object
     * @throws InvalidRangeException
     *             invalid range
     * @see org.gumtree.data.interfaces.IArray#section(int[], int[], int[])
     */
    @Override
    public IArrayUtils section(final int[] origin, final int[] shape,
            final long[] stride) throws InvalidRangeException {
        try {
        	int[] intStride = null;
        	if (stride != null) {
        		intStride = new int[stride.length];
            	for (int i = 0; i < stride.length; i++) {
            		intStride[i] = (int) stride[i];
            	}
        	}
            return new NcArrayUtils(new NcArray(getNcArray().getArray().section(origin, shape, intStride), getArray().getFactoryName()));
        } catch (ucar.ma2.InvalidRangeException e) {
            throw new InvalidRangeException(e);
        }
    }

    /**
     * @param origin
     *            array of integers
     * @param shape
     *            array of integers
     * @param stride
     *            array of integers
     * @return IArray object
     * @throws InvalidRangeException
     *             invalid range
     * @see org.gumtree.data.interfaces.IArray#sectionNoReduce(int[], int[], int[])
     */
    @Override
    public IArrayUtils sectionNoReduce(final int[] origin, final int[] shape,
            final long[] stride) throws InvalidRangeException {
        try {
        	int[] intStride = null;
        	if (stride != null) {
        		intStride = new int[stride.length];
            	for (int i = 0; i < stride.length; i++) {
            		intStride[i] = (int) stride[i];
            	}
        	}
            return (new NcArray(getNcArray().getArray()
                    .sectionNoReduce(origin, shape, intStride), getArray().getFactoryName())).getArrayUtils();
        } catch (ucar.ma2.InvalidRangeException e) {
            Factory.getLogger().log( Level.SEVERE, e.getMessage() );
            throw new InvalidRangeException(e);
        }
    }

    @Override
    public IArrayUtils sectionNoReduce(List<IRange> ranges) throws InvalidRangeException {
        throw new NotImplementedException();
    }

    /**
     * @param dim1
     *            integer value
     * @param dim2
     *            integer value
     * @return IArray object
     * @see org.gumtree.data.interfaces.IArray#transpose(int, int)
     */
    @Override
    public IArrayUtils transpose(final int dim1, final int dim2) {
        return new NcArrayUtils(new NcArray(getNcArray().getArray().transpose(dim1, dim2), getArray().getFactoryName()));
    }
    
    @Override
    public Object copyTo1DJavaArray() {
        return getNcArray().getArray().copyTo1DJavaArray();
    }

    @Override
	public Object get1DJavaArray(final Class<?> wantType) {
		return getNcArray().getArray().get1DJavaArray(wantType);
	}
	
    @Override
    public Object copyToNDJavaArray() {
        return getNcArray().getArray().copyToNDJavaArray();
    }

    @Override
    public IArrayUtils slice(int dim, int value) {
    	return new NcArrayUtils(new NcArray(getNcArray().getArray().slice(dim, value), getArray().getFactoryName()));
    }

	@Override
	public IArrayUtils createArrayUtils(IArray array) {
		return new NcArrayUtils((NcArray) array);
	}
	
	@Override
	public IArrayUtils flip(final int dim) {
		return new NcArrayUtils(new NcArray(getNcArray().getArray().flip(dim), getArray().getFactoryName()));
	}
    
	@Override
	public IArrayUtils permute(final int[] dims) {
		return new NcArrayUtils(new NcArray(getNcArray().getArray().permute(dims), getArray().getFactoryName()));
	}

}
