package org.cdma.plugin.edf.array;

import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.IIndex;

public class BasicArrayIterator implements IArrayIterator {

    private final IArray array;
    private final IIndex index;

    public BasicArrayIterator(IArray array) {
        this(array, array == null ? null : array.getIndex());
    }

    public BasicArrayIterator(IArray array, IIndex index) {
        super();
        this.array = array;
        this.index = index;
    }

    @Override
    public boolean getBooleanCurrent() {
        return getBoolean(getObjectCurrent());
    }

    @Override
    public boolean getBooleanNext() {
        return getBoolean(getObjectNext());
    }

    @Override
    public char getCharCurrent() {
        return getChar(getObjectCurrent());
    }

    @Override
    public char getCharNext() {
        return getChar(getObjectNext());
    }

    @Override
    public byte getByteCurrent() {
        return getByte(getObjectCurrent());
    }

    @Override
    public byte getByteNext() {
        return getByte(getObjectNext());
    }

    @Override
    public short getShortCurrent() {
        return getShort(getObjectCurrent());
    }

    @Override
    public short getShortNext() {
        return getShort(getObjectNext());
    }

    @Override
    public int getIntCurrent() {
        return getInt(getObjectCurrent());
    }

    @Override
    public int getIntNext() {
        return getInt(getObjectNext());
    }

    @Override
    public long getLongCurrent() {
        return getLong(getObjectCurrent());
    }

    @Override
    public long getLongNext() {
        return getLong(getObjectNext());
    }

    @Override
    public float getFloatCurrent() {
        return getFloat(getObjectCurrent());
    }

    @Override
    public float getFloatNext() {
        return getFloat(getObjectNext());
    }

    @Override
    public double getDoubleCurrent() {
        return getDouble(getObjectCurrent());
    }

    @Override
    public double getDoubleNext() {
        return getDouble(getObjectNext());
    }

    @Override
    public int[] getCurrentCounter() {
        if (index == null) {
            return null;
        }
        else {
            return index.getCurrentCounter();
        }
    }

    @Override
    public Object getObjectCurrent() {
        if (array == null) {
            return null;
        }
        else {
            return array.getObject(index);
        }
    }

    @Override
    public Object getObjectNext() {
        incrementIndex();
        Object oOutput = this.getObjectCurrent();
        return oOutput;
    }

    @Override
    public boolean hasNext() {
        if (index != null) {
            int[] dataIndex = index.getCurrentCounter();
            for (int i = dataIndex.length - 1; i >= 0; i--) {
                if (dataIndex[i] + 1 < index.getShape()[i]) {
                    return true;
                }
            }
        }
        return false;
    }

    @Override
    public Object next() {
        Object oOutput = this.getObjectCurrent();
        this.incrementIndex();
        return oOutput;
    }

    @Override
    public void setBooleanCurrent(boolean val) {
        setObjectCurrent(val);
    }

    @Override
    public void setBooleanNext(boolean val) {
        setObjectNext(val);
    }

    @Override
    public void setByteCurrent(byte val) {
        setObjectCurrent(val);
    }

    @Override
    public void setByteNext(byte val) {
        setObjectNext(val);
    }

    @Override
    public void setCharCurrent(char val) {
        setObjectCurrent(val);
    }

    @Override
    public void setCharNext(char val) {
        setObjectNext(val);
    }

    @Override
    public void setDoubleCurrent(double val) {
        setObjectCurrent(val);
    }

    @Override
    public void setDoubleNext(double val) {
        setObjectNext(val);
    }

    @Override
    public void setFloatCurrent(float val) {
        setObjectCurrent(val);
    }

    @Override
    public void setFloatNext(float val) {
        setObjectNext(val);
    }

    @Override
    public void setIntCurrent(int val) {
        setObjectCurrent(val);
    }

    @Override
    public void setIntNext(int val) {
        setObjectNext(val);
    }

    @Override
    public void setLongCurrent(long val) {
        setObjectCurrent(val);
    }

    @Override
    public void setLongNext(long val) {
        setObjectNext(val);
    }

    @Override
    public void setObjectCurrent(Object val) {
        if (array != null) {
            array.setObject(index, val);
        }
    }

    @Override
    public void setObjectNext(Object val) {
        incrementIndex();
        setObjectCurrent(val);
    }

    @Override
    public void setShortCurrent(short val) {
        setObjectCurrent(val);
    }

    @Override
    public void setShortNext(short val) {
        setObjectNext(val);
    }

    private void incrementIndex() {
        if (index != null) {
            int[] dataIndex = index.getCurrentCounter();
            for (int i = dataIndex.length - 1; i >= 0; i--) {
                if (dataIndex[i] + 1 >= index.getShape()[i] && i > 0) {
                    dataIndex[i] = 0;
                }
                else {
                    dataIndex[i]++;
                    index.set(dataIndex);
                    break;
                }
            }
        }
    }

    private boolean getBoolean(Object oOutput) {
        if (oOutput instanceof Boolean) {
            return ((Boolean) oOutput).booleanValue();
        }
        else if (oOutput instanceof Number) {
            return ((Number) oOutput).intValue() != 0;
        }
        return false;
    }

    private char getChar(Object oOutput) {
        if (oOutput instanceof Character) {
            return ((Character) oOutput).charValue();
        }
        else {
            return '\u0000';
        }
    }

    private byte getByte(Object oOutput) {
        if (oOutput instanceof Number) {
            return ((Number) oOutput).byteValue();
        }
        return (byte) 0;
    }

    private Short getShort(Object oOutput) {
        if (oOutput instanceof Number) {
            return ((Number) oOutput).shortValue();
        }
        return (short) 0;
    }

    private int getInt(Object oOutput) {
        if (oOutput instanceof Number) {
            return ((Number) oOutput).intValue();
        }
        return 0;
    }

    private long getLong(Object oOutput) {
        if (oOutput instanceof Number) {
            return ((Number) oOutput).longValue();
        }
        return 0;
    }

    private float getFloat(Object oOutput) {
        if (oOutput instanceof Number) {
            return ((Number) oOutput).floatValue();
        }
        return Float.NaN;
    }

    private double getDouble(Object oOutput) {
        if (oOutput instanceof Number) {
            return ((Number) oOutput).doubleValue();
        }
        return Double.NaN;
    }

}
