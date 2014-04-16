/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Cl�ment Rodriguez (clement.rodriguez@synchrotron-soleil.fr)
// ****************************************************************************
package org.cdma.arrays;

import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IRange;

public final class DefaultRange implements IRange, Cloneable {
    public static final DefaultRange EMPTY = new DefaultRange();
    public static final DefaultRange VLEN  = new DefaultRange(-1);

    /// Members
    private long      mLast;     // number of elements
    private long      mFirst;    // first value in range
    private long      mStride;   // stride, must be >= 1
    private String     mName;     // optional name
    private boolean   mReduced;  // was this ranged reduced or not

    /// Constructors
    /**
     * Used for EMPTY
     */
    private DefaultRange() {
	this.mLast    = 0;
	this.mFirst   = 0;
	this.mStride  = 1;
	this.mName    = null;
	this.mReduced = false;
    }

    /**
     * Create a range starting at zero, with an unit stride of length "length".
     * @param length number of elements in the NxsRange
     */
    public DefaultRange(int length) {
	this.mName    = null;
	this.mFirst   = 0;
	this.mStride  = 1;
	this.mLast    = length - 1;
	this.mReduced = false;
    }

    /**
     * Create a range with a specified stride.
     *
     * @param name   name of the range 
     * @param first  first index in range
     * @param last   last index in range, inclusive
     * @param stride stride between consecutive elements, must be > 0
     * @throws InvalidRangeException elements must be nonnegative: 0 <= first <= last, stride > 0
     */
    public DefaultRange(String name, long first, long last, long stride) throws InvalidRangeException {
	this.mLast    = last;
	this.mFirst   = first;
	this.mStride  = stride;
	this.mName    = name;
	this.mReduced = false;
    }

    public DefaultRange(String name, long first, long last, long stride, boolean reduced) throws InvalidRangeException {
	this(name, first, last, stride);
	mReduced = reduced;
    }

    public DefaultRange( IRange range ) {
	this.mLast    = range.last();
	this.mFirst   = range.first();
	this.mStride  = range.stride();
	this.mName    = range.getName();
	this.mReduced = false;
    }

    /// Accessors
    @Override
    public String getName() {
	return mName;
    }

    @Override
    public long first() {
	return mFirst;
    }

    @Override
    public long last() {
	return mLast;
    }

    @Override
    public int length() {
	return (int) ((mLast - mFirst) / mStride) + 1;
    }

    @Override
    public long stride() {
	return mStride;
    }

    public void stride(long value) {
	mStride = value;
    }

    public void last(long value) {
	mLast = value;
    }

    public void first(long value) {
	mFirst = value;
    }

    /// Methods
    @Override
    public IRange clone() {
	DefaultRange range = DefaultRange.EMPTY;
	try {
	    range = new DefaultRange(mName, mFirst, mLast, mStride);
	    range.mReduced = mReduced;
	} catch( InvalidRangeException e ) {
	}
	return range;
    }

    @Override
    public IRange compact() throws InvalidRangeException {
	long first, last, stride;
	String name;

	stride = 1;
	first  = mFirst / mStride;
	last   = mLast / mStride;
	name   = mName; 

	return new DefaultRange(name, first, last, stride);
    }

    @Override
    public IRange compose(IRange r) throws InvalidRangeException {
	if ((length() == 0) || (r.length() == 0)) {
	    return EMPTY;
	}
	if ( this.equals(VLEN) || r.equals(VLEN) ) {
	    return VLEN;
	}

	long first  = element(r.first());
	long stride = stride() * r.stride();
	long last   = element(r.last());
	return new DefaultRange(mName, first, last, stride);
    }

    @Override
    public boolean contains(int i) {
	if( i < first() ) {
	    return false;
	}
	if( i > last()) {
	    return false;
	}
	if( mStride == 1) {
	    return true;
	}
	return (i - mFirst) % mStride == 0;
    }

    @Override
    public int element(long i) throws InvalidRangeException {
	if (i < 0) {
	    throw new InvalidRangeException("i must be >= 0");
	}
	if (i > mLast) {
	    throw new InvalidRangeException("i must be < length");
	}

	return (int) (mFirst + i * mStride);
    }

    @Override
    public int getFirstInInterval(int start) {
	if (start > last()) { 
	    return -1;
	}
	if (start <= mFirst) { 
	    return (int) mFirst;
	}
	if (mStride == 1) { 
	    return start;
	}
	long offset = start - mFirst;
	long incr = offset % mStride;
	long result = start + incr;
	return (int) ((result > last()) ? -1 : result);
    }

    @Override
    public long index(int elem) throws InvalidRangeException {
	if (elem < mFirst) {
	    throw new InvalidRangeException("elem must be >= first");
	}
	long result = (elem - mFirst) / mStride;
	if (result > mLast) {
	    throw new InvalidRangeException("elem must be <= last = n * stride");
	}
	return (int) result;
    }

    @Override
    public IRange intersect(IRange r) throws InvalidRangeException {
	if ((length() == 0) || (r.length() == 0)) {
	    return EMPTY;
	}
	if ( this.equals(VLEN) || r.equals(VLEN) ) {
	    return VLEN;
	}

	long last = Math.min(this.last(), r.last());
	long stride = stride() * r.stride();

	long useFirst;
	if (stride == 1) {
	    useFirst = Math.max(this.first(), r.first());
	}
	else if (stride() == 1) { // then r has a stride
	    if (r.first() >= first()) {
		useFirst = r.first();
	    }
	    else {
		long incr = (first() - r.first()) / stride;
		useFirst = r.first() + incr * stride;
		if (useFirst < first()) {
		    useFirst += stride;
		}
	    }
	}
	else if (r.stride() == 1) { // then this has a stride
	    if (first() >= r.first()) {
		useFirst = first();
	    }
	    else {
		long incr = (r.first() - first()) / stride;
		useFirst = first() + incr * stride;
		if (useFirst < r.first()) {
		    useFirst += stride;
		}
	    }
	}
	else {
	    throw new UnsupportedOperationException("Intersection when both ranges have a stride");
	}
	if (useFirst > last) {
	    return EMPTY;
	}
	return new DefaultRange(mName, useFirst, last, stride);
    }

    @Override
    public boolean intersects(IRange r) {
	if ((length() == 0) || (r.length() == 0)) {
	    return false;
	}
	if ( this.equals(VLEN) || r.equals(VLEN) ) {
	    return true;
	}

	long last = Math.min(this.last(), r.last());
	long stride = stride() * r.stride();

	long useFirst;
	if (stride == 1) {
	    useFirst = Math.max(this.first(), r.first());
	}
	else if (stride() == 1) { // then r has a stride
	    if (r.first() >= first()) {
		useFirst = r.first();
	    }
	    else {
		long incr = (first() - r.first()) / stride;
		useFirst = r.first() + incr * stride;
		if (useFirst < first()) {
		    useFirst += stride;
		}
	    }
	}
	else if (r.stride() == 1) { // then this has a stride
	    if (first() >= r.first()) {
		useFirst = first();
	    }
	    else {
		long incr = (r.first() - first()) / stride;
		useFirst = first() + incr * stride;
		if (useFirst < r.first()) {
		    useFirst += stride;
		}
	    }
	}
	else {
	    throw new UnsupportedOperationException("Intersection when both ranges have a stride");
	}
	return (useFirst <= last);
    }

    @Override
    public IRange shiftOrigin(int origin) throws InvalidRangeException {
	return new DefaultRange( mName, mFirst + origin, mLast + origin, mStride, mReduced );
    }

    @Override
    public IRange union(IRange r) throws InvalidRangeException {
	if( r.stride() != mStride ) {
	    throw new InvalidRangeException("Stride must identical to make a IRange union!");
	}

	if (length() == 0) {
	    return r;
	}
	if ( this.equals(VLEN) || r.equals(VLEN) ) {
	    return VLEN;
	}
	if (r.length() == 0) {
	    return this;
	}

	long first, last;
	String name = mName;

	// Seek the smallest value
	first = Math.min( mFirst, r.first() );
	last  = Math.max( mLast , r.last()  );

	return new DefaultRange(name, first, last, mStride);
    }

    @Override
    public String toString() {
	StringBuffer str = new StringBuffer();
	str.append("name: '"   ).append(getName());
	str.append("', first: ").append(first());
	str.append(", last: "  ).append(last());
	str.append(", stride: ").append(stride());
	str.append(", length: ").append(length());
	str.append(", reduce: ").append(mReduced);
	return str.toString();
    }

    public boolean reduced() {
	return mReduced;
    }

    public void reduced(boolean reduce) {
	mReduced = reduce;
    }
}
