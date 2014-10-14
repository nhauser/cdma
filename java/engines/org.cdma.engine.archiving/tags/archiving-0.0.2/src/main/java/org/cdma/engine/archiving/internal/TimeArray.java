package org.cdma.engine.archiving.internal;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.cdma.exception.BackupException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.ISliceIterator;
import org.cdma.math.IArrayMath;
import org.cdma.utils.IArrayUtils;

/**
 * 
 * @author SAINTIN
 *         This class convert a String Array type in long time stamp array
 */

public class TimeArray implements IArray {
    private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
    private final IArray array;
    private final IArrayUtils arrayUtils;

    public TimeArray(final IArray array) {
        this.array = array;
        arrayUtils = new TimeArraysUtils(array.getArrayUtils());
    }

    @Override
    public String getFactoryName() {
        return array.getFactoryName();
    }

    @Override
    public IArray copy() {
        return array.copy();
    }

    @Override
    public IArray copy(boolean data) {
        return array.copy(data);
    }

    @Override
    public IArrayUtils getArrayUtils() {
        return arrayUtils;
    }


    @Override
    public IArrayMath getArrayMath() {
        return array.getArrayMath();
    }

    @Override
    public boolean getBoolean(IIndex index) {
        return array.getBoolean(index);
    }

    @Override
    public byte getByte(IIndex index) {
        return array.getByte(index);
    }

    @Override
    public char getChar(IIndex index) {
        return array.getChar(index);
    }

    @Override
    public double getDouble(IIndex index) {
        return getLong(index);
    }

    @Override
    public Class<?> getElementType() {
        // return long type instead of String
        return Long.class;
    }

    @Override
    public float getFloat(IIndex index) {
        return array.getFloat(index);
    }

    @Override
    public IIndex getIndex() {
        return array.getIndex();
    }

    @Override
    public int getInt(IIndex index) {
        return array.getInt(index);
    }

    @Override
    public IArrayIterator getIterator() {
        return array.getIterator();
    }

    @Override
    public long getLong(IIndex index) {
        long longValue =  array.getLong(index);
        String dateValue = getObject(index).toString();
        Long convertToMs = convertStringDateToMs(dateValue);
        if (convertToMs != null) {
            longValue = convertToMs.longValue();
        }
        return longValue;
    }

    public final static Long convertStringDateToMs(final String dateString) {
        Long longValue = null;
        if (dateString != null) {
            try {
                Date date = dateFormat.parse(dateString);
                longValue = date.getTime();

            } catch (ParseException e) {
                longValue = null;
            }
        }
        return longValue;
    }

    @Override
    public Object getObject(IIndex index) {
        return array.getObject(index);
    }

    @Override
    public int getRank() {
        return array.getRank();
    }

    @Override
    public IArrayIterator getRegionIterator(int[] reference, int[] range) throws InvalidRangeException {
        return array.getRegionIterator(reference, range);
    }

    @Override
    public int[] getShape() {
        return array.getShape();
    }

    @Override
    public short getShort(IIndex index) {
        return array.getShort(index);
    }

    @Override
    public long getSize() {
        return array.getSize();
    }

    @Override
    public Object getStorage() {
        return array.getStorage();
    }

    @Override
    public void setBoolean(IIndex index, boolean value) {
        array.setBoolean(index, value);
    }

    @Override
    public void setByte(IIndex index, byte value) {
        array.setByte(index, value);

    }

    @Override
    public void setChar(IIndex index, char value) {
        array.setChar(index, value);
    }

    @Override
    public void setDouble(IIndex index, double value) {
        array.setDouble(index, value);
    }

    @Override
    public void setFloat(IIndex index, float value) {
        array.setFloat(index, value);

    }

    @Override
    public void setInt(IIndex index, int value) {
        array.setInt(index, value);
    }

    @Override
    public void setLong(IIndex index, long value) {
        String dateValue = dateFormat.format(new Date(value));
        array.setObject(index, dateValue);
    }

    @Override
    public void setObject(IIndex index, Object value) {
        array.setObject(index, value);
    }

    @Override
    public void setShort(IIndex index, short value) {
        array.setShort(index, value);
    }

    @Override
    public String shapeToString() {
        return array.shapeToString();
    }

    @Override
    public void setIndex(IIndex index) {
        array.setIndex(index);
    }

    @Override
    public ISliceIterator getSliceIterator(int rank) throws ShapeNotMatchException, InvalidRangeException {
        return array.getSliceIterator(rank);
    }

    @Override
    public void releaseStorage() throws BackupException {
        array.releaseStorage();
    }

    @Override
    public long getRegisterId() {
        return array.getRegisterId();
    }

    @Override
    public void lock() {
        array.lock();
    }

    @Override
    public void unlock() {
        array.unlock();
    }

    @Override
    public boolean isDirty() {
        return array.isDirty();
    }

    @Override
    public void setDirty(boolean dirty) {
        array.setDirty(dirty);
    }

    @Override
    public IArray setDouble(double value) {
        return array.setDouble(value);
    }

}
