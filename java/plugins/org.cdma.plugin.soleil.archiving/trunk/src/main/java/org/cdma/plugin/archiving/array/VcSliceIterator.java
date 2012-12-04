package org.cdma.plugin.archiving.array;

import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.ISliceIterator;

public final class VcSliceIterator implements ISliceIterator {

	// Members
	private final VcArrayIterator mIterator; // iterator of the whole_array
	private final IArray mArray; // array of the original shape containing all slices
	private final int[] mDimension; // shape of the slice
	private IArray mSlice; // array that will be returned when asking getArrayNext

	// Constructor
	/**
	 * Create a new NxsSliceIterator that permits to iterate over slices that
	 * are the last dim dimensions of this array.
	 * 
	 * @param array
	 *            source of the slice
	 * @param dim
	 *            returned dimensions
	 */
	public VcSliceIterator(final IArray array, final int dim) throws InvalidRangeException {
		// If ranks are equal, make sure at least one iteration is performed.
		// We cannot use 'reshape' (which would be ideal) as that creates a
		// new copy of the array storage and so is unusable
		mArray = array;
		int[] shape = mArray.getShape();
		int[] rangeList = shape.clone();
		int[] origin = mArray.getIndex().getOrigin().clone();
		long[] stride = mArray.getIndex().getStride().clone();
		int rank = mArray.getRank();

		// shape to make a 2D array from multiple-dim array
		mDimension = new int[dim];
		System.arraycopy(shape, shape.length - dim, mDimension, 0, dim);
		for (int i = 0; i <  dim; i++) {
			rangeList[rank - i - 1] = 1;
		}
		// Create an iterator over the higher dimensions. We leave in the
		// final dimensions so that we can use the getCurrentCounter method
		// to create an origin.

		IIndex index = null;
		try {
			index = mArray.getIndex().clone();
			
	        // As we get a reference on array's IIndex we directly modify it
	        index.setOrigin(origin);
	        index.setStride(stride);
	        index.setShape(rangeList);
		} catch (CloneNotSupportedException e) {
			Factory.getLogger().log(Level.SEVERE, "Unable to initialize slice iterator!", e);
		}
        mIterator = new VcArrayIterator(mArray, index, false);
	}

	// / Public methods
	@Override
	public IArray getArrayNext() throws InvalidRangeException {
		next();
		if (mSlice == null) {
			createSlice();
		} else {
			updateSlice();
		}

		return mSlice;
	}

	@Override
	public int[] getSliceShape() throws InvalidRangeException {
		return mDimension.clone();
	}

	@Override
	public int[] getSlicePosition() {
		return mIterator.getCounter();
	}

	@Override
	public boolean hasNext() {
		return mIterator.hasNext();
	}

	@Override
	public void next() {
		mIterator.incrementIndex();
	}

	@Override
	public String getFactoryName() {
		return mArray.getFactoryName();
	}

	// ---------------------------------------------------------
	// / Private methods
	// ---------------------------------------------------------
	private void createSlice() throws InvalidRangeException {
		mSlice = mArray.copy(false);
		updateSlice();
	}

	private void updateSlice() throws InvalidRangeException {
		int i = 0;
		int[] iShape = mArray.getShape();
		int[] iOrigin = mArray.getIndex().getOrigin();
		int[] iCurPos = mIterator.getCounter();
		for (int pos : iCurPos) {
			if (pos >= iShape[i]) {
				mSlice = null;
				return;
			}
			iOrigin[i] += pos;
			i++;
		}

		java.util.Arrays.fill(iShape, 1);
		System.arraycopy(mDimension, 0, iShape, iShape.length - mDimension.length, mDimension.length);
		IIndex index;
		try {
			index = mArray.getIndex().clone();
			index.setShape(iShape);
			index.setOrigin(iOrigin);
			index.reduce();
			mSlice.setIndex(index);
		} catch (CloneNotSupportedException e) {
			mSlice = null;
		}
	}
}
