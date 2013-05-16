//package org.cdma.plugin.edf.array;
//
//import java.util.ArrayList;
//import java.util.List;
//
//import org.cdma.exception.InvalidRangeException;
//import org.cdma.interfaces.IIndex;
//import org.cdma.interfaces.IRange;
//
//public class BasicIndex implements IIndex, Cloneable {
//    private int[] m_iCurPos; // Current position pointed by this index
//    private BasicRange[] m_ranges; // Ranges that constitute the index global view in each dimension
//    private boolean m_upToDate = false; // Does the overall shape has changed
//    private int m_lastIndex;
//
//    public BasicIndex(int[] shape) {
//        this(shape, new int[shape.length], shape);
//    }
//
//    public BasicIndex(int[] shape, int[] start, int length[]) {
//        super();
//        int stride = 1;
//        m_iCurPos = new int[shape.length];
//        m_ranges = new BasicRange[shape.length];
//        for (int i = shape.length - 1; i >= 0; i--) {
//            m_ranges[i] = new BasicRange("", start[i] * stride,
//                    (start[i] + length[i] - 1) * stride, stride);
//            stride *= shape[i];
//        }
//    }
//
//    // ---------------------------------------------------------
//    // Public methods
//    // ---------------------------------------------------------
//    @Override
//    public long currentElement() {
//        int value = 0;
//        try {
//            for (int i = 0; i < m_iCurPos.length; i++) {
//                value += (m_ranges[i]).element(m_iCurPos[i]);
//            }
//        }
//        catch (InvalidRangeException e) {
//            value = -1;
//        }
//
//        return value;
//    }
//
//    @Override
//    public int[] getCurrentCounter() {
//        return m_iCurPos;
//    }
//
//    @Override
//    public String getIndexName(int dim) {
//        return (m_ranges[dim]).getName();
//    }
//
//    @Override
//    public int getRank() {
//        return m_ranges.length;
//    }
//
//    @Override
//    public int[] getShape() {
//        int[] shape = new int[m_ranges.length];
//        int i = 0;
//        for (IRange range : m_ranges) {
//            shape[i] = range.length();
//            i++;
//        }
//        return shape;
//    }
//
//    @Override
//    public long[] getStride() {
//        long[] stride = new long[m_ranges.length];
//        int i = 0;
//        for (IRange range : m_ranges) {
//            stride[i] = range.stride();
//            i++;
//        }
//        return stride;
//    }
//
//    @Override
//    public int[] getOrigin() {
//        int[] stride = new int[m_ranges.length];
//        int i = 0;
//        for (IRange range : m_ranges) {
//            stride[i] = (int) range.first();
//            i++;
//        }
//        return stride;
//    }
//
//    @Override
//    public long getSize() {
//        if (m_ranges.length == 0) {
//            return 0;
//        }
//
//        long size = 1;
//        for (IRange range : m_ranges) {
//            size *= range.length();
//        }
//
//        return size;
//    }
//
//    @Override
//    public void setOrigin(int[] origins) {
//        int i = 0;
//        try {
//            for (BasicRange range : m_ranges) {
//                m_ranges[i] = (BasicRange) range.shiftOrigin(origins[i] * range.stride());
//                i++;
//            }
//        }
//        catch (InvalidRangeException e) {
//            e.printStackTrace();
//        }
//    }
//
//    @Override
//    public void setShape(int[] value) {
//        IRange range;
//        m_upToDate = false;
//        for (int i = 0; i < value.length; i++) {
//            range = m_ranges[i];
//            m_ranges[i] = new BasicRange(range.getName(), range.first(), range.first()
//                    + (value[i] - 1) * range.stride(), range.stride());
//        }
//    }
//
//    @Override
//    public void setStride(long[] stride) {
//        IRange range;
//        if (stride == null) {
//            return;
//        }
//        m_upToDate = false;
//        for (int i = 0; i < stride.length; i++) {
//            range = m_ranges[i];
//            m_ranges[i] = new BasicRange(range.getName(), range.first(), range.last(), stride[i]);
//        }
//    }
//
//    @Override
//    public IIndex set(int[] index) {
//        if (index.length != m_iCurPos.length)
//            throw new IllegalArgumentException();
//
//        m_iCurPos = index;
//        return this;
//    }
//
//    @Override
//    public IIndex set(int v0) {
//        m_iCurPos[0] = v0;
//        return this;
//    }
//
//    @Override
//    public IIndex set(int v0, int v1) {
//        m_iCurPos[0] = v0;
//        m_iCurPos[1] = v1;
//        return this;
//    }
//
//    @Override
//    public IIndex set(int v0, int v1, int v2) {
//        m_iCurPos[0] = v0;
//        m_iCurPos[1] = v1;
//        m_iCurPos[2] = v2;
//        return this;
//    }
//
//    @Override
//    public IIndex set(int v0, int v1, int v2, int v3) {
//        m_iCurPos[0] = v0;
//        m_iCurPos[1] = v1;
//        m_iCurPos[2] = v2;
//        m_iCurPos[3] = v3;
//        return this;
//    }
//
//    @Override
//    public IIndex set(int v0, int v1, int v2, int v3, int v4) {
//        m_iCurPos[0] = v0;
//        m_iCurPos[1] = v1;
//        m_iCurPos[2] = v2;
//        m_iCurPos[3] = v4;
//        return this;
//    }
//
//    @Override
//    public IIndex set(int v0, int v1, int v2, int v3, int v4, int v5) {
//        m_iCurPos[0] = v0;
//        m_iCurPos[1] = v1;
//        m_iCurPos[2] = v2;
//        m_iCurPos[3] = v4;
//        m_iCurPos[5] = v5;
//        return this;
//    }
//
//    @Override
//    public IIndex set(int v0, int v1, int v2, int v3, int v4, int v5, int v6) {
//        m_iCurPos[0] = v0;
//        m_iCurPos[1] = v1;
//        m_iCurPos[2] = v2;
//        m_iCurPos[3] = v4;
//        m_iCurPos[5] = v5;
//        m_iCurPos[6] = v6;
//        return this;
//    }
//
//    @Override
//    public IIndex set0(int v) {
//        m_iCurPos[0] = v;
//        return this;
//    }
//
//    @Override
//    public IIndex set1(int v) {
//        m_iCurPos[1] = v;
//        return this;
//    }
//
//    @Override
//    public IIndex set2(int v) {
//        m_iCurPos[2] = v;
//        return this;
//    }
//
//    @Override
//    public IIndex set3(int v) {
//        m_iCurPos[3] = v;
//        return this;
//    }
//
//    @Override
//    public IIndex set4(int v) {
//        m_iCurPos[4] = v;
//        return this;
//    }
//
//    @Override
//    public IIndex set5(int v) {
//        m_iCurPos[5] = v;
//        return this;
//    }
//
//    @Override
//    public IIndex set6(int v) {
//        m_iCurPos[6] = v;
//        return this;
//    }
//
//    @Override
//    public void setDim(int dim, int value) {
//        if (dim >= m_iCurPos.length || dim < 0)
//            throw new IllegalArgumentException();
//
//        m_iCurPos[dim] = value;
//    }
//
//    @Override
//    public void setIndexName(int dim, String indexName) {
//        m_upToDate = false;
//        IRange range = m_ranges[dim];
//        m_ranges[dim] = new BasicRange(indexName, range.first(), range.last(), range.stride());
//    }
//
//    @Override
//    public String toStringDebug() {
//        StringBuilder sbuff = new StringBuilder(100);
//        sbuff.setLength(0);
//        int rank = m_ranges.length;
//
//        sbuff.append(" shape= ");
//        for (int ii = 0; ii < rank; ii++) {
//            sbuff.append((m_ranges[ii]).length());
//            sbuff.append(" ");
//        }
//
//        sbuff.append(" stride= ");
//        for (int ii = 0; ii < rank; ii++) {
//            sbuff.append((m_ranges[ii]).stride());
//            sbuff.append(" ");
//        }
//
//        long size = 1;
//        for (int ii = 0; ii < rank; ii++) {
//            size *= (m_ranges[ii]).length();
//        }
//        sbuff.append(" size= ").append(size);
//        sbuff.append(" rank= ").append(rank);
//
//        sbuff.append(" current= ");
//        for (int ii = 0; ii < rank; ii++) {
//            sbuff.append(m_iCurPos[ii]);
//            sbuff.append(" ");
//        }
//
//        return sbuff.toString();
//    }
//
//    @Override
//    public IIndex reduce() {
//        IIndex c = (IIndex) clone();
//        for (int ii = 0; ii < m_iCurPos.length; ii++)
//            // is there a dimension with length = 1
//            if ((m_ranges[ii]).length() == 1) {
//                // remove it
//                IIndex newc = c.reduce(ii);
//
//                // ensure there is not any more to do
//                return newc.reduce();
//            }
//        m_upToDate = false;
//        return c;
//    }
//
//    /**
//     * Create a new Index based on current one by eliminating the specified dimension;
//     *
//     * @param dim: dimension to eliminate: must be of length one, else IllegalArgumentException
//     * @return the new Index
//     */
//    @Override
//    public IIndex reduce(int dim) {
//        if ((dim < 0) || (dim >= m_ranges.length))
//            throw new IllegalArgumentException("illegal reduce dim " + dim);
//        if ((m_ranges[dim]).length() != 1)
//            throw new IllegalArgumentException("illegal reduce dim " + dim + " : length != 1");
//
//        // Calculate new shape
//        int[] shape;
//        if (m_ranges.length > 1) {
//            shape = new int[m_ranges.length - 1];
//        }
//        else {
//            shape = new int[] { 1 };
//        }
//
//        // Create new index corresponding to shape
//        BasicIndex newindex = new BasicIndex(shape);
//
//        // Update index with all old ranges but the one corresponding to dim
//        int count = 0;
//        for (int ii = 0; ii <= (m_ranges.length - 1); ii++) {
//            if (ii != dim) {
//                newindex.m_ranges[count] = m_ranges[ii];
//                newindex.m_iCurPos[count] = m_iCurPos[ii];
//                count++;
//            }
//        }
//
//        try {
//            // Shifting index origin according to the removed dimension
//            if (dim > 0) {
//                newindex.m_ranges[dim - 1] = (BasicRange) newindex.m_ranges[dim - 1]
//                        .shiftOrigin(m_ranges[dim].first()/* + newindex.m_ranges[dim - 1].first() */);
//            }
//            else {
//                newindex.m_ranges[0] = (BasicRange) newindex.m_ranges[0].shiftOrigin(m_ranges[0]
//                        .first());
//            }
//        }
//        catch (InvalidRangeException e) {
//            e.printStackTrace();
//        }
//
//        return newindex;
//    }
//
//    @Override
//    public String toString() {
//        StringBuffer str = new StringBuffer(getClass().getSimpleName());
//        str.append(" - Position: [");
//        int i = 0;
//        for (int pos : m_iCurPos) {
//            if (i++ != 0) {
//                str.append(", ");
//            }
//            str.append(pos);
//        }
//        i = 0;
//        str.append("]  <=> Index: ").append(currentElement()).append("\nRanges:\n");
//        for (IRange r : m_ranges) {
//            str.append("\t- ").append(r);
//            if (++i < m_ranges.length) {
//                str.append("\n");
//            }
//        }
//        return str.toString();
//    }
//
//    @Override
//    public Object clone() {
//        try {
//            BasicIndex clone = (BasicIndex) super.clone();
//            clone.m_ranges = new BasicRange[m_ranges.length];
//            for (int i = 0; i < m_ranges.length; i++) {
//                clone.m_ranges[i] = (BasicRange) m_ranges[i].clone();
//            }
//            clone.m_iCurPos = m_iCurPos.clone();
//            return clone;
//        }
//        catch (CloneNotSupportedException e) {
//            // Should not happen because this class is Cloneable
//            e.printStackTrace();
//            return null;
//        }
//    }
//
//    // ---------------------------------------------------------
//    // Protected methods
//    // ---------------------------------------------------------
//    @Override
//    public long lastElement() {
//        int last = 0;
//        if (!m_upToDate) {
//            for (IRange range : m_ranges) {
//                last += range.last();
//            }
//            m_lastIndex = last;
//            m_upToDate = true;
//        }
//        return m_lastIndex;
//    }
//
//    protected List<IRange> getRangeList() {
//        ArrayList<IRange> list = new ArrayList<IRange>();
//
//        for (IRange range : m_ranges) {
//            list.add(range);
//        }
//        return list;
//    }
//
// }
