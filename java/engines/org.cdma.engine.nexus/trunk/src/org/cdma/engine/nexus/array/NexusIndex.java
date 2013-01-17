//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.engine.nexus.array;

import org.cdma.arrays.DefaultIndex;
import org.cdma.interfaces.IIndex;

public final class NexusIndex extends DefaultIndex implements Cloneable {
	private long mLast = -1;
	
    /// Constructors
    public NexusIndex(String factoryName, fr.soleil.nexus.DataItem ds) {
        super(factoryName, ds.getSize(), new int[ds.getSize().length], ds.getSize());
    }

    public NexusIndex(String factoryName, int[] shape) {
        super(factoryName, shape.clone(), new int[shape.length], shape.clone());
    }

    public NexusIndex(NexusIndex index) {
        super(index);
    }

    public NexusIndex(String factoryName, int[] shape, int[] start, int[] length) {
        super( factoryName, shape, start, length );
    }

    @Override
    public IIndex clone() {
        return new NexusIndex(this);
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
