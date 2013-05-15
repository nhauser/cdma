package org.cdma.plugin.edf.array;

import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IRange;

public class BasicRange implements IRange, Cloneable {

    public final static BasicRange EMPTY = new BasicRange();

    // Members //
    private final String name; // optional name
    private final long first; // first value in range
    private final long last; // last value in range
    private final long stride; // stride, must be >= 1

    // Constructors //
    /**
     * Used for EMPTY
     */
    public BasicRange() {
        this(0);
    }

    /**
     * Create a range starting at zero, with unit stride.
     * 
     * @param length number of elements in the BasicRange
     */
    public BasicRange(int length) {
        this(null, 0, length - 1, 1);
    }

    /**
     * Create a range with a specified stride.
     * 
     * @param name name of the range
     * @param first first value in range
     * @param last last value in range, inclusive
     * @param stride stride between consecutive elements, must be > 0
     */
    public BasicRange(String name, long first, long last, long stride) {
        this.last = last;
        this.first = first;
        if (stride < 1) {
            stride = 1;
        }
        this.stride = stride;
        this.name = name;
    }

    // Getters //
    @Override
    public String getName() {
        return name;
    }

    @Override
    public int element(long i) throws InvalidRangeException {
        if (i < 0) {
            throw new InvalidRangeException(i + " is < 0");
        }
        else if (i > last) {
            throw new InvalidRangeException(i + "is > " + last);
        }
        else {
            return (int) (first + i * stride);
        }
    }

    @Override
    public long first() {
        return first;
    }

    @Override
    public long last() {
        return last;
    }

    @Override
    public int length() {
        if (last < first) {
            return 0;
        }
        return (int) (((last - first) / stride) + 1);
    }

    @Override
    public long stride() {
        return stride;
    }

    // Methods //
    @Override
    public IRange clone() {
        try {
            return (IRange) super.clone();
        }
        catch (CloneNotSupportedException e) {
            // Should never happen because this class is Cloneable
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public IRange compact() {
        return new BasicRange(name, first, last, 1);
    }

    @Override
    public IRange compose(IRange r) throws InvalidRangeException {
        if ((length() < 0) || (r.length() < 0)) {
            return null;
        }
        else if ((length() == 0) || (r.length() == 0)) {
            return EMPTY;
        }
        else {
            return new BasicRange(name, element(r.first()), element(r.last()), stride()
                    * r.stride());
        }

    }

    @Override
    public boolean contains(int i) {
        if ((i < first()) || (i > last())) {
            return false;
        }
        else if (stride == 1) {
            return true;
        }
        else {
            return (i - first) % stride == 0;
        }
    }

    @Override
    public int getFirstInInterval(int start) {
        if (start > last()) {
            return -1;
        }
        else if (start <= first) {
            return (int) first;
        }
        else if (stride == 1) {
            return start;
        }
        else {
            long offset = start - first;
            long incr = offset % stride;
            long result = start + incr;
            return (int) ((result > last()) ? -1 : result);
        }
    }

    @Override
    public long index(int elem) throws InvalidRangeException {
        if (elem < first) {
            throw new InvalidRangeException("elem must be >= first");
        }
        long result = (elem - first) / stride;
        if (result > last) {
            throw new InvalidRangeException("elem must be <= last = n * stride");
        }
        return result;
    }

    @Override
    public IRange intersect(IRange r) {
        if ((length() < 0) || (r.length() < 0)) {
            return null;
        }
        else if ((length() == 0) || (r.length() == 0)) {
            return EMPTY;
        }
        else {
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
                throw new UnsupportedOperationException(
                        "Intersection when both ranges have a stride");
            }
            if (useFirst > last) {
                return EMPTY;
            }
            else {
                return new BasicRange(name, useFirst, last, stride);
            }
        }
    }

    @Override
    public boolean intersects(IRange r) {
        return (intersect(r) != EMPTY);
    }

    @Override
    public IRange shiftOrigin(int origin) throws InvalidRangeException {
        return new BasicRange(name, first + origin, last + origin, stride);
    }

    @Override
    public IRange union(IRange r) throws InvalidRangeException {
        if (r.stride() != stride) {
            throw new InvalidRangeException("Stride must identical to make a IRange union!");
        }
        else if (length() == 0) {
            return r;
        }
        else if ((r.length() < 0) || (length() < 0)) {
            return null;
        }
        else if (r.length() == 0) {
            return this;
        }
        else {
            long first, last;
            // Seek the smallest value
            first = Math.min(this.first, r.first());
            last = Math.max(this.last, r.last());
            return new BasicRange(name, first, last, stride);
        }
    }

    @Override
    public String toString() {
        StringBuffer str = new StringBuffer(getClass().getSimpleName());
        str.append(" - name: '").append(getName());
        str.append("', first: ").append(first());
        str.append(", last: ").append(last());
        str.append(", stride: ").append(stride());
        return str.toString();
    }

}
