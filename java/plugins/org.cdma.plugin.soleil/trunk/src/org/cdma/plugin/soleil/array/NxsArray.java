package org.cdma.plugin.soleil.array;


import org.cdma.engine.nexus.array.NexusArray;
import org.cdma.engine.nexus.array.NexusArrayIterator;
import org.cdma.engine.nexus.array.NexusIndex;
import org.cdma.engine.nexus.array.NexusSliceIterator;
import org.cdma.exception.BackupException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.ISliceIterator;
import org.cdma.math.IArrayMath;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.utils.IArrayUtils;

import fr.soleil.nexus.DataItem;

public final class NxsArray implements IArray {
    private Object     mData;      // It's an array of values
    private NxsIndex   mIndex;     // IIndex corresponding to mArray shape
    private IArray[]   mArrays;    // IArray of IArray
    private boolean    mFastMode;  // flag toggling in fast mode: i.e when true the IArray will be shared among references

    public NxsArray( IArray[] arrays ) {
        mArrays   = arrays.clone();
        mData     = null;
        initDimSize();    

        // Define the same viewable part for all sub-IArray
        NexusIndex index = mIndex.getIndexStorage();
        for( IArray array : mArrays ) {
            array.setIndex(index.clone());
        }
        mFastMode = false;
    }

    public NxsArray( NxsArray array ) {
        mIndex = (NxsIndex) array.mIndex.clone();
        mData   = array.mData;

        IIndex index = mIndex.getIndexStorage();
        mArrays      = new IArray[array.mArrays.length];
        for( int i = 0; i < array.mArrays.length; i++ ) {
            mArrays[i] = array.mArrays[i].copy(false);
            mArrays[i].setIndex(index);

        }
        mFastMode = array.mFastMode;
    }

    public NxsArray(DataItem item) {
        this( new IArray[] { new NexusArray(NxsFactory.NAME, item) });
    }

    public NxsArray(Object oArray, int[] iShape) {
        this( new IArray[] { new NexusArray(NxsFactory.NAME, oArray, iShape) });
    }

    @Override
    public Object getObject(IIndex index) {
        return this.get(index);
    }

    @Override
    public IArray copy() {
        return copy(true);
    }

    @Override
    public IArray copy(boolean data) {
        NxsArray result = new NxsArray(this);

        if( data ) {
            result.mData = NexusArray.copyJavaArray(mData);
        }

        return result;
    }

    @Override
    public IArrayMath getArrayMath() {
        return NxsFactory.createArrayMath(this);
    }

    @Override
    public IArrayUtils getArrayUtils() {
        return NxsFactory.createArrayUtils(this);
    }

    @Override
    public boolean getBoolean(IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[(int) nodeIndex].getBoolean(itemIdx);
    }

    @Override
    public byte getByte(IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[(int) nodeIndex].getByte(itemIdx);
    }

    @Override
    public char getChar(IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[(int) nodeIndex].getChar(itemIdx);
    }

    @Override
    public double getDouble(IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[(int) nodeIndex].getDouble(itemIdx);
    }

    @Override
    public float getFloat(IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[(int) nodeIndex].getFloat(itemIdx);
    }

    @Override
    public int getInt(IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[(int) nodeIndex].getInt(itemIdx);
    }

    @Override
    public long getLong(IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[(int) nodeIndex].getLong(itemIdx);
    }

    @Override
    public short getShort(IIndex ima) {
        IndexNode ind = getIndexNode(ima);
        IIndex itemIdx = ind.getIndex();
        int nodeIndex = ind.getNode();
        return mArrays[(int) nodeIndex].getShort(itemIdx);
    }

    @Override
    public Class<?> getElementType() {
        Class<?> result = null;
        if( mArrays != null )
        {
            result = mArrays[0].getElementType();
        }
        return result;
    }

    @Override
    public IIndex getIndex() {
        return mIndex.clone();
    }

    @Override
    public IArrayIterator getIterator() {
        NxsIndex index = (NxsIndex) mIndex.clone();
        NexusIndex storage = index.getIndexStorage();
        storage.unReduce();
        storage.setOrigin( new int[storage.getRank()] );
        storage.reduce();
        return new NexusArrayIterator(this, index );
    }

    @Override
    public int getRank() {
        return mIndex.getRank();
    }

    @Override
    public IArrayIterator getRegionIterator(int[] reference, int[] range)
            throws InvalidRangeException {
        int[] shape = mIndex.getShape();
        IIndex index = new NexusIndex( NxsFactory.NAME, shape, reference, range );
        return new NexusArrayIterator(this, index);
    }

    @Override
    public int[] getShape() {
        return mIndex.getShape();
    }

    @Override
    public long getSize() {
        return mIndex.getSize();
    }

    @Override
    public Object getStorage() {
        Object result = mData;
        if( mData == null && mArrays != null ) {
            NexusIndex matrixIndex = (NexusIndex) mIndex.getIndexMatrix().clone();
            //matrixIndex.set(new int[matrixIndex.getRank()]);

            Long nbMatrixCells  = matrixIndex.getSize() == 0 ? 1 : matrixIndex.getSize();
            Long nbStorageCells = mIndex.getIndexStorage().getSize();

            int[] shape = new int[] { nbMatrixCells.intValue(), nbStorageCells.intValue() };
            result = java.lang.reflect.Array.newInstance(getElementType(), shape);
            
            if( mArrays[0].getElementType().equals(String.class) ) {
                for( int i = 0; i < nbMatrixCells; i++ ) {
                    java.lang.reflect.Array.set(result, i, new String[] { (String) mArrays[(int) matrixIndex.currentElement()].getStorage() } );
                    NexusArrayIterator.incrementIndex(matrixIndex);
                }
            }
            else {
                for( int i = 0; i < nbMatrixCells; i++ ) {
                    java.lang.reflect.Array.set(result, i, mArrays[(int) matrixIndex.currentElement()].getStorage() );
                    NexusArrayIterator.incrementIndex(matrixIndex);
                }
            }
        }

        return result;
    }

    @Override
    public void setBoolean(IIndex ima, boolean value) {
        set(ima, value);
    }

    @Override
    public void setByte(IIndex ima, byte value) {
        set(ima, value);
    }

    @Override
    public void setChar(IIndex ima, char value) {
        set(ima, value);
    }

    @Override
    public void setDouble(IIndex ima, double value) {
        set(ima, value);
    }

    @Override
    public void setFloat(IIndex ima, float value) {
        set(ima, value);
    }

    @Override
    public void setInt(IIndex ima, int value) {
        set(ima, value);
    }

    @Override
    public void setLong(IIndex ima, long value) {
        set(ima, value);
    }

    @Override
    public void setObject(IIndex ima, Object value) {
        set(ima, value);
    }

    @Override
    public void setShort(IIndex ima, short value) {
        set(ima, value);
    }

    @Override
    public String shapeToString() {
        int[] shape = getShape();
        StringBuilder sb = new StringBuilder();
        if (shape.length != 0) {
            sb.append('(');
            for (int i = 0; i < shape.length; i++) {
                int s = shape[i];
                if (i > 0) {
                    sb.append(",");
                }
                sb.append(s);
            }
            sb.append(')');
        }
        return sb.toString();
    }

    @Override
    public void setIndex(IIndex index) {
        if( index instanceof NxsIndex ) {
            mIndex = (NxsIndex) index;
        }
        else {
            mIndex = new NxsIndex(mIndex.getIndexMatrix().getRank(), index.getShape(), index.getOrigin(), index.getShape() );
            mIndex.set(index.getCurrentCounter());
        }

        for( IArray array : mArrays ) {
            array.setIndex(mIndex.getIndexStorage());
        }
    }

    @Override
    public ISliceIterator getSliceIterator(int rank)
            throws ShapeNotMatchException, InvalidRangeException {
        NexusSliceIterator result = new NexusSliceIterator(this, rank);
        if( mFastMode ) {
            result.setFastMode(mFastMode);
        }
        return result;
    }

    @Override
    public void releaseStorage() throws BackupException {
        throw new NotImplementedException();
    }

    @Override
    public long getRegisterId() {
        throw new NotImplementedException();
    }

    @Override
    public void lock() {
        throw new NotImplementedException();
    }

    @Override
    public void unlock() {
        throw new NotImplementedException();
    }

    @Override
    public boolean isDirty() {
        throw new NotImplementedException();
    }

    @Override
    public IArray setDouble(double value) {
        throw new NotImplementedException();
    }

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public String toString() {
        return mIndex.toString();
    }

    @Override
    public void setDirty(boolean dirty) {
        throw new NotImplementedException();
    }

    public boolean getFastMode() {
        return mFastMode;
    }
    
    public void setFastMode( boolean activate ) {
        mFastMode = activate;
    }
    
    // ---------------------------------------------------------
    /// Private methods
    // ---------------------------------------------------------
    /**
     * InitDimSize
     * Initialize member dimension sizes 'm_iDimS' according to defined member data 'm_oData'
     */
    private void initDimSize()
    {
        // Check data existence
        if( mArrays != null ) {
            // Set dimension rank
            int matrixRank = mArrays.length > 1 ? 1 : 0;
            int[] shape    = new int[ matrixRank + mArrays[0].getRank() ];


            // Fill dimension size array
            if( matrixRank > 0 ) {
                shape[0] = mArrays.length;
            }

            int i = 0;
            for( int size : mArrays[0].getShape() ) {
                shape[i + matrixRank] = size;
                i++;
            }

            mIndex = new NxsIndex( matrixRank, shape, new int[shape.length], shape );
        }
    }

    /**
     * Get the object targeted by given index and return it (eventually using auto-boxing).
     * It's the central data access method that all other methods rely on.
     * 
     * @param index targeting a cell 
     * @return the content of cell designed by the index
     * @throws InvalidRangeException if one of the index is bigger than the corresponding dimension shape
     */
    private Object get(IIndex index) {
        Object result = null;
        IndexNode ind = getIndexNode(index);
        IIndex itemIdx = ind.getIndex();

        int nodeIndex = ind.getNode();
        IArray slab = mArrays[(int) nodeIndex];
        if( slab != null ) {
            result = slab.getObject(itemIdx);
        }
        return result;
    }

    private IndexNode getIndexNode(IIndex index) {
        int nodeIndex;
        IIndex itemIdx;
        if( mArrays.length > 1 ) {
            NxsIndex idx;
            if( ! (index instanceof NxsIndex) ) {
                int rank = mIndex.getIndexMatrix().getRank();
                idx = new NxsIndex(rank, mIndex.getShape(), index.getOrigin(), index.getShape());
                idx.set(index.getCurrentCounter());
            }
            else {
                idx = (NxsIndex) index;
            }
            nodeIndex = (int) idx.currentElementMatrix();
            itemIdx = idx.getIndexStorage();
        }
        else {
            nodeIndex = 0;
            itemIdx = index;
        }
        return new IndexNode(itemIdx, nodeIndex);
    }

    private void set(IIndex index, Object value) {
        NxsIndex idx = null;
        if( ! (index instanceof NxsIndex) ) {
            idx = new NxsIndex(mIndex.getIndexMatrix().getRank(), index);
        }
        else {
            idx = (NxsIndex) index;
        }

        NexusIndex itemIdx = idx.getIndexStorage();
        long nodeIndex = idx.currentElementMatrix();
        IArray slab = mArrays[(int) nodeIndex];
        if( slab != null ) {
            slab.setObject(itemIdx, value);
        }
    }

    private class IndexNode {
        private IIndex mIndex;
        private int mNode;

        public IndexNode( IIndex index, int node ) {
            mIndex = index;
            mNode  = node;
        }

        public IIndex getIndex() {
            return mIndex;
        }

        public int getNode() {
            return mNode;
        }
    }

}
