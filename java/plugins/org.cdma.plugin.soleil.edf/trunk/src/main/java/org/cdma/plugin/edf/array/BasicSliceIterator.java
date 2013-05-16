//package org.cdma.plugin.edf.array;
//
//import org.cdma.exception.InvalidRangeException;
//import org.cdma.interfaces.IArray;
//import org.cdma.interfaces.IArrayIterator;
//import org.cdma.interfaces.IIndex;
//import org.cdma.interfaces.ISliceIterator;
//
//public class BasicSliceIterator implements ISliceIterator {
//
//    // Members
//    private final IArrayIterator m_iterator; // iterator of the whole_array
//    private final IArray m_array; // array of the original shape containing all slices
//    private final int[] m_dimension; // shape of the slice
//
//    // Constructor
//    /**
//     * Create a new BasicSliceIterator that permits to iterate over slices that are the last dim
//     * dimensions of this array.
//     *
//     * @param array source of the slice
//     * @param dim returned dimensions
//     */
//    public BasicSliceIterator(final IArray array, final int dim) throws InvalidRangeException {
//        // If ranks are equal, make sure at least one iteration is performed.
//        // We cannot use 'reshape' (which would be ideal) as that creates a
//        // new copy of the array storage and so is unusable
//        m_array = array;
//        int[] shape = m_array.getShape();
//        int[] rangeList = shape.clone();
//        long[] stride = m_array.getIndex().getStride();
//        int rank = m_array.getRank();
//
//        // shape to make a 2D array from multiple-dim array
//        m_dimension = shape.clone();
//        for (int i = 0; i < rank - dim; i++) {
//            m_dimension[i] = 1;
//        }
//        for (int i = 0; i < dim; i++) {
//            rangeList[rank - i - 1] = 1;
//        }
//        // Create an iterator over the higher dimensions. We leave in the
//        // final dimensions so that we can use the getCurrentCounter method
//        // to create an origin.
//        IArray loopArray = new BasicArray(m_array.getStorage(), m_array.getShape());
//        IIndex index = loopArray.getIndex();
//        index.set(new int[rank]);
//        index.setShape(rangeList);
//        index.setStride(stride);
//
//        m_iterator = loopArray.getIterator();
//
//    }
//
//
//
//    @Override
//    public IArray getArrayNext() throws InvalidRangeException {
//        next();
//        return createSlice();
//    }
//
//    @Override
//    public int[] getSliceShape() throws InvalidRangeException {
//        return m_dimension;
//    }
//
//    @Override
//    public boolean hasNext() {
//        return m_iterator.hasNext();
//    }
//
//    @Override
//    public void next() {
//        m_iterator.next();
//    }
//
//    private IArray createSlice() throws InvalidRangeException {
//        IArray slice = new BasicArray(m_array.getStorage(), m_array.getShape());
//        IIndex index = slice.getIndex();
//        index.setShape(m_dimension);
//        index.setOrigin(m_iterator.getCurrentCounter());
//
//        return slice;
//    }
//
// }
