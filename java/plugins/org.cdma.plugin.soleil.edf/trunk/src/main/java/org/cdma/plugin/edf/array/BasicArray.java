package org.cdma.plugin.edf.array;

import org.cdma.arrays.DefaultArrayInline;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.plugin.edf.EdfFactory;

public class BasicArray extends DefaultArrayInline {

    public BasicArray(Class<?> clazz, int[] iShape)
            throws InvalidArrayTypeException {
        super(EdfFactory.NAME, clazz, iShape);
        // TODO Auto-generated constructor stub
    }

    public BasicArray(Object o, int[] iShape) throws InvalidArrayTypeException {
        super(EdfFactory.NAME, o, iShape);
        // TODO Auto-generated constructor stub
    }

}

//public class BasicArray implements IArray, Cloneable {
//    // IIndex corresponding to this Array (dimension sizes defining the viewable part of the array)
//    private IIndex m_Index;
//    // It's an array of values
//    private Object m_oData;
//    // True if the stored array has a rank of 1 (independently of its shape)
//    private boolean m_bRawArray;
//    // Is the array synchronized with the handled file
//    private boolean m_isDirty;
//    // Shape of the array (dimension sizes of the storage backing)
//    private int[] m_iShape;
//
//    // Constructors
//    public BasicArray(Object oArray, int[] iShape) {
//        m_Index = new BasicIndex(iShape);
//        m_oData = oArray;
//        m_iShape = iShape;
//        if (iShape.length == 1) {
//            m_bRawArray = true;
//        }
//        else {
//            m_bRawArray = (oArray.getClass().isArray() && !(java.lang.reflect.Array.get(oArray, 0)
//                    .getClass().isArray()));
//        }
//
//    }
//
//    // ---------------------------------------------------------
//    // / @Override
//    // / public methods
//    // ---------------------------------------------------------
//
//    @Override
//    public IArrayMath getArrayMath() {
//        return new ArrayMath(this);
//    }
//
//    @Override
//    public IArrayUtils getArrayUtils() {
//        return new DefaultArrayUtils(this);
//    }
//
//    // / Specific method to match NetCDF plug-in behavior
//    @Override
//    public String toString() {
//        if (m_oData instanceof String) {
//            return (String) m_oData;
//        }
//        else {
//            StringBuilder sbuff = new StringBuilder();
//            IArrayIterator ii = getIterator();
//            int i = 1;
//
//            while (ii.hasNext()) {
//                Object data = ii.getObjectNext();
//                sbuff.append(data);
//                sbuff.append(data);
//                if (i % m_iShape[m_iShape.length - 1] == 0) {
//                    sbuff.append("\n");
//                }
//                else {
//                    sbuff.append(" ");
//                }
//
//                i++;
//            }
//            return sbuff.toString();
//        }
//
//    }
//
//    // / IArray underlying data access
//    @Override
//    public Object getStorage() {
//        return m_oData;
//    }
//
//    @Override
//    public int[] getShape() {
//        return m_iShape;
//    }
//
//    @Override
//    public Class<?> getElementType() {
//        if (m_oData != null) {
//            if (m_oData.getClass().isArray()) {
//                return m_oData.getClass().getComponentType();
//            }
//            else {
//                return m_oData.getClass();
//            }
//        }
//        return null;
//    }
//
//    @Override
//    public void lock() {
//        // TODO Auto-generated method stub
//    }
//
//    @Override
//    public boolean isDirty() {
//        return m_isDirty;
//    }
//
//    @Override
//    public void releaseStorage() throws BackupException {
//        // TODO Auto-generated method stub
//    }
//
//    @Override
//    public String shapeToString() {
//        int[] shape = getShape();
//        StringBuilder sb = new StringBuilder();
//        if (shape.length > 0) {
//            sb.append('(');
//            for (int i = 0; i < shape.length; i++) {
//                int s = shape[i];
//                if (i > 0) {
//                    sb.append(",");
//                }
//                sb.append(s);
//            }
//            sb.append(')');
//        }
//        return sb.toString();
//    }
//
//    @Override
//    public void unlock() {
//        // TODO Auto-generated method stub
//
//    }
//
//    // / IArray data manipulation
//    @Override
//    public IIndex getIndex() {
//        return m_Index;
//    }
//
//    @Override
//    public IArrayIterator getIterator() {
//        return new BasicArrayIterator(this);
//    }
//
//    @Override
//    public int getRank() {
//        int rank = 0;
//
//        if (m_iShape.length == 1 && m_iShape[0] == 1) {
//            return 0;
//        }
//
//        for (int i : m_iShape) {
//            if (i > 1)
//                rank++;
//        }
//
//        return rank;
//    }
//
//    @Override
//    public IArrayIterator getRegionIterator(int[] reference, int[] range)
//            throws InvalidRangeException {
//        BasicIndex index = new BasicIndex(m_iShape, reference, range);
//        return new BasicArrayIterator(this, index);
//    }
//
//    @Override
//    public long getRegisterId() {
//        // TODO Auto-generated method stub
//        return 0;
//    }
//
//    @Override
//    public long getSize() {
//        if (m_iShape.length == 0) {
//            return 0;
//        }
//
//        long size = 1;
//
//        for (int i = 0; i < m_iShape.length; i++) {
//            size *= m_iShape[i];
//        }
//
//        return size;
//    }
//
//    @Override
//    public ISliceIterator getSliceIterator(int rank) throws ShapeNotMatchException,
//    InvalidRangeException {
//        return new BasicSliceIterator(this, rank);
//    }
//
//    // IArray data getters and setters
//    @Override
//    public boolean getBoolean(IIndex ima) {
//        Object value = get(ima);
//        if (value instanceof Boolean) {
//            return (Boolean) value;
//        }
//        else if (value instanceof Number) {
//            return (((Number) value).doubleValue() != 0);
//        }
//        return false;
//    }
//
//    @Override
//    public char getChar(IIndex ima) {
//        return ((Character) get(ima)).charValue();
//    }
//
//    @Override
//    public byte getByte(IIndex ima) {
//        return ((Number) get(ima)).byteValue();
//    }
//
//    @Override
//    public short getShort(IIndex ima) {
//        return ((Short) get(ima)).shortValue();
//    }
//
//    @Override
//    public int getInt(IIndex ima) {
//        return ((Number) get(ima)).intValue();
//    }
//
//    @Override
//    public long getLong(IIndex ima) {
//        return ((Number) get(ima)).longValue();
//    }
//
//    @Override
//    public float getFloat(IIndex ima) {
//        return ((Number) get(ima)).floatValue();
//    }
//
//    @Override
//    public double getDouble(IIndex ima) {
//        return ((Number) get(ima)).doubleValue();
//    }
//
//    @Override
//    public Object getObject(IIndex index) {
//        return get(index);
//    }
//
//    @Override
//    public void setBoolean(IIndex ima, boolean value) {
//        set(ima, value);
//    }
//
//    @Override
//    public void setChar(IIndex ima, char value) {
//        set(ima, value);
//    }
//
//    @Override
//    public void setByte(IIndex ima, byte value) {
//        set(ima, value);
//    }
//
//    @Override
//    public void setShort(IIndex ima, short value) {
//        set(ima, value);
//    }
//
//    @Override
//    public void setInt(IIndex ima, int value) {
//        set(ima, value);
//    }
//
//    @Override
//    public void setLong(IIndex ima, long value) {
//        set(ima, value);
//    }
//
//    @Override
//    public void setFloat(IIndex ima, float value) {
//        set(ima, value);
//    }
//
//    @Override
//    public void setDouble(IIndex ima, double value) {
//        set(ima, value);
//    }
//
//    @Override
//    public void setObject(IIndex ima, Object value) {
//        set(ima, value);
//    }
//
//    @Override
//    public IArray setDouble(double value) {
//        if (m_bRawArray) {
//            java.util.Arrays.fill((double[]) m_oData, value);
//        }
//        else {
//            setDouble(m_oData, value);
//        }
//        return this;
//    }
//
//    @Override
//    public IArray copy() {
//        return (IArray) clone();
//    }
//
//    @Override
//    public void setIndex(IIndex index) {
//        m_Index = index;
//    }
//
//    // ---------------------------------------------------------
//    // / Protected methods
//    // ---------------------------------------------------------
//    protected void setStorage(Object obj) {
//        m_oData = obj;
//    }
//
//    protected boolean isSingleRawArray() {
//        return m_bRawArray;
//    }
//
//    protected IArray sectionNoReduce(int[] origin, int[] shape, long[] stride)
//            throws ShapeNotMatchException {
//        BasicArray array = new BasicArray(m_oData, m_iShape);
//        array.m_Index.setShape(shape);
//        array.m_Index.setStride(stride);
//        ((BasicIndex) array.m_Index).setOrigin(origin);
//        return array;
//    }
//
//    // ---------------------------------------------------------
//    // / Private methods
//    // ---------------------------------------------------------
//    /**
//     * Translate the given IIndex into the corresponding cell index. This method is used to access a
//     * multidimensional array's cell when the memory storage is a single raw array.
//     *
//     * @param index cibling a cell in a multidimensional array
//     * @return the cell number in a single raw array (that carry the same logical shape)
//     */
//    private int translateIndex(IIndex index) {
//        int[] indexes = index.getCurrentCounter();
//
//        int lPos = 0, lStartRaw;
//        for (int k = 1; k < m_iShape.length; k++) {
//
//            lStartRaw = 1;
//            for (int j = 0; j < k; j++) {
//                lStartRaw *= m_iShape[j];
//            }
//            lStartRaw *= indexes[k - 1];
//            lPos += lStartRaw;
//        }
//        lPos += indexes[indexes.length - 1];
//        return lPos;
//    }
//
//    /**
//     * Get the object targeted by given index and return it (eventually using outboxing). It's the
//     * central data access method that all other methods rely on.
//     *
//     * @param index targeting a cell
//     * @return the content of cell designed by the index
//     * @throws InvalidRangeException if one of the index is bigger than the corresponding dimension
//     *             shape
//     */
//    private Object get(IIndex index) {
//        Object oCurObj = null;
//
//        // If it's a string then no array
//        if (m_oData.getClass().equals(String.class)) {
//            return m_oData;
//        }
//        // If it's a scalar value then we return it
//        else if (!m_oData.getClass().isArray()) {
//            return m_oData;
//        }
//        // If it's a single raw array, then we compute indexes to have the corresponding cell number
//        else if (m_bRawArray) {
//            int lPos = (int) index.currentElement();
//            return java.lang.reflect.Array.get(m_oData, lPos);
//        }
//        // If it's a multidimensionnal array, then we get sub-part until to have the single cell
//        // we're interested in
//        else {
//            int[] indexes = index.getCurrentCounter();
//            oCurObj = m_oData;
//            for (int i = 0; i < indexes.length; i++) {
//                oCurObj = java.lang.reflect.Array.get(oCurObj, indexes[i]);
//            }
//        }
//
//        return oCurObj;
//    }
//
//    /**
//     * Set the given object into the targeted cell by given index (eventually using autoboxing).
//     * It's the central data access method that all other methods rely on.
//     *
//     * @param index targeting a cell
//     * @param value new value to set in the array
//     * @throws InvalidRangeException if one of the index is bigger than the corresponding dimension
//     *             shape
//     */
//    private void set(IIndex index, Object value) {
//        // If array has string class: then it's a scalar string
//        if (m_oData.getClass().equals(String.class)) {
//            m_oData = value;
//        }
//        // If array isn't an array we set the scalar value
//        else if (!m_oData.getClass().isArray()) {
//            m_oData = value;
//        }
//        // If it's a single raw array, then we compute indexes to have the corresponding cell number
//        else if (m_bRawArray) {
//            int lPos = translateIndex(index);
//            java.lang.reflect.Array.set(m_oData, lPos, value);
//        }
//        // Else it's a multidimensional array, so we will take slices from each dimension until we
//        // can reach requested cell
//        else {
//            int[] indexes = index.getCurrentCounter();
//            Object oCurObj = m_oData;
//            for (int i = 0; i < indexes.length - 1; i++) {
//                oCurObj = java.lang.reflect.Array.get(oCurObj, indexes[i]);
//            }
//            java.lang.reflect.Array.set(oCurObj, indexes[indexes.length - 1], value);
//        }
//
//    }
//
//    /**
//     * Recursive method that sets all values of the given array (whatever it's form is) to the same
//     * given double value
//     *
//     * @param array object array to fill
//     * @param value double value to be set in the array
//     * @return the array filled properly
//     * @note ensure the given array is a double[](...[]) or a Double[](...[])
//     */
//    private Object setDouble(Object array, double value) {
//        if (array.getClass().isArray()) {
//            int iLength = java.lang.reflect.Array.getLength(array);
//            for (int j = 0; j < iLength; j++) {
//                Object o = java.lang.reflect.Array.get(array, j);
//                if (o.getClass().isArray()) {
//                    setDouble(o, value);
//                }
//                else {
//                    java.util.Arrays.fill((double[]) array, value);
//                    return array;
//                }
//            }
//        }
//        else
//            java.lang.reflect.Array.set(array, 0, value);
//
//        return array;
//    }
//
//    @Override
//    public Object clone() {
//        try {
//            BasicArray clone = (BasicArray) super.clone();
//            if (m_Index instanceof BasicIndex) {
//                clone.m_Index = (IIndex) ((BasicIndex) m_Index).clone();
//            }
//            if (m_iShape != null) {
//                clone.m_iShape = Arrays.copyOf(m_iShape, m_iShape.length);
//            }
//            return clone;
//        }
//        catch (CloneNotSupportedException e) {
//            // Should not happen because this class is Cloneable
//            e.printStackTrace();
//            return null;
//        }
//    }
//
//    @Override
//    public String getFactoryName() {
//        // TODO Auto-generated method stub
//        return null;
//    }
//
//    @Override
//    public IArray copy(boolean data) {
//        // TODO Auto-generated method stub
//        return null;
//    }
//
//    @Override
//    public void setDirty(boolean dirty) {
//        // TODO Auto-generated method stub
//
//    }
//
// }
