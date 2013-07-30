package org.cdma.engine.hdf.array;

import ncsa.hdf.object.ScalarDS;

import org.cdma.arrays.DefaultIndex;
import org.cdma.engine.hdf.utils.HdfObjectUtils;
import org.cdma.interfaces.IIndex;


public final class HdfIndex extends DefaultIndex implements Cloneable {
    private long mLast = -1;

    // Constructors
    public HdfIndex(String factoryName, ScalarDS ds) {
        this(factoryName, HdfObjectUtils.convertLongToInt(ds.getDims()));
    }

    public HdfIndex(String factoryName, int[] shape) {
        super(factoryName, shape.clone(), new int[shape.length], shape.clone());
    }

    public HdfIndex(HdfIndex index) {
        super(index);
    }

    public HdfIndex(String factoryName, int[] shape, int[] start, int[] length) {
        super( factoryName, shape, start, length );
    }

    @Override
    public IIndex clone() {
        return new HdfIndex(this);
    }

    @Override
    public long lastElement() {
        if( mLast < 0 ) {
            int[] position = getShape();
            for( int dim = 0; dim < position.length; dim++ ) {
                position[dim]--;
            }
            mLast = elementOffset( position );
        }
        return mLast;
    }

    @Override
    public void setOrigin(int[] origins) {
        mLast = -1;
        super.setOrigin(origins);
    }

    @Override
    public void setShape(int[] value) {
        mLast = -1;
        super.setShape(value);
    }

    @Override
    public void setStride(long[] stride) {
        mLast = -1;
        super.setStride(stride);
    }

    @Override
    public long firstElement() {
        return 0;
    }

    @Override
    public long elementOffset(int[] position) {
        long value = super.elementOffset(position) - super.elementOffset( new int[position.length] );
        if( value < 0 ) {
            value = -1;
        }
        return value;
    }



}
