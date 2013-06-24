package org.cdma.arrays;

import java.util.List;

import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.IRange;
import org.cdma.interfaces.ISliceIterator;
import org.cdma.utils.ArrayUtils;
import org.cdma.utils.IArrayUtils;

public class DefaultCompositeArrayUtils extends ArrayUtils {

    public DefaultCompositeArrayUtils(DefaultCompositeArray array) {
        super(array);
    }

    @Override
    public Object copyTo1DJavaArray() {
        // Instantiate a new convenient array for storage
        int length = ((Long) getArray().getSize()).intValue();
        Class<?> type = getArray().getElementType();
        Object array = java.lang.reflect.Array.newInstance(type, length);

        DefaultIndex storageIndex = ((DefaultCompositeIndex) getArray().getIndex())
                .getIndexStorage();
        DefaultIndex matrixIndex = ((DefaultCompositeIndex) getArray().getIndex()).getIndexMatrix();

        // If the storing array is a stack of DataItem
        Long size = matrixIndex.getSize();
        Long nbMatrixCells = size == 0 ? 1 : size;
        Long nbStorageCells = storageIndex.getSize();

        Long startPos = storageIndex.elementOffset(new int[storageIndex.getRank()]);
        Object fullArray = getArray().getStorage();
        Object partArray = null;
        int[] posMatrix = new int[1];

        Object inlineArray;
        for( int cell = 0; cell < nbMatrixCells; cell++ ) {
            posMatrix[0] = cell;
            partArray = java.lang.reflect.Array.get(fullArray, cell );
            IArray iPartArray;
            try {
                iPartArray = new DefaultArrayMatrix("", partArray);
                inlineArray = iPartArray.getArrayUtils().copyTo1DJavaArray();
                System.arraycopy(inlineArray, startPos.intValue(), array,
                        cell * nbStorageCells.intValue(), nbStorageCells.intValue());
            }
            catch (InvalidArrayTypeException e) {

                e.printStackTrace();
            }
        }

        return array;
    }



    /**
     * This methods recovers the type of data present in a {@link Class} that represents N dimension
     * arrays.
     * 
     * @param arrayClass The {@link Class}
     * @return The {@link Class} that represents the data type in the given array. (Example: if
     *         <code>arrayClass</code> is <code>boolean[][]</code>, the result will be
     *         {@link Boolean#TYPE}). This method returns <code>null</code> if
     *         <code>arrayClass</code> is <code>null</code>.
     */
    public static Class<?> recoverDeepComponentType(Class<?> arrayClass) {
        Class<?> result = arrayClass;
        if (arrayClass != null) {
            while (result.isArray()) {
                result = result.getComponentType();
            }
        }
        return result;
    }

    @Override
    public Object copyToNDJavaArray() {
        return copyMatrixItemsToMultiDim();
    }

    // --------------------------------------------------
    // tools methods
    // --------------------------------------------------
    public static Object copyJavaArray(Object array) {
        Object result = array;
        if (result == null) {
            return null;
        }

        // Determine rank of array (by parsing data array class name)
        String sClassName = array.getClass().getName();
        int iRank = 0;
        int iIndex = 0;
        char cChar;
        while (iIndex < sClassName.length()) {
            cChar = sClassName.charAt(iIndex);
            iIndex++;
            if (cChar == '[') {
                iRank++;
            }
        }

        // Set dimension rank
        int[] shape = new int[iRank];

        // Fill dimension size array
        for (int i = 0; i < iRank; i++) {
            shape[i] = java.lang.reflect.Array.getLength(result);
            result = java.lang.reflect.Array.get(result, 0);
        }

        // Define a convenient array (shape and type)
        result = java.lang.reflect.Array.newInstance(array.getClass().getComponentType(), shape);

        return copyJavaArray(array, result);
    }

    public static Object copyJavaArray(Object source, Object target) {
        Object item = java.lang.reflect.Array.get(source, 0);
        int length = java.lang.reflect.Array.getLength(source);

        if (item.getClass().isArray()) {
            Object tmpSrc;
            Object tmpTar;
            for (int i = 0; i < length; i++) {
                tmpSrc = java.lang.reflect.Array.get(source, i);
                tmpTar = java.lang.reflect.Array.get(target, i);
                copyJavaArray(tmpSrc, tmpTar);
            }
        }
        else {
            System.arraycopy(source, 0, target, 0, length);
        }

        return target;
    }



    @Override
    public Object get1DJavaArray(Class<?> wantType) {
        throw new NotImplementedException();
    }


    @Override
    public IArrayUtils sectionNoReduce(List<IRange> ranges) throws InvalidRangeException {
        throw new NotImplementedException();
    }


    @Override
    public IArrayUtils transpose(int dim1, int dim2) {
        IArray array = getArray().copy(false);
        IIndex index = array.getIndex();
        int[] shape = index.getShape();
        int[] origin = index.getOrigin();
        int[] position = index.getCurrentCounter();
        long[] stride = index.getStride();

        int sha = shape[dim1];
        int ori = origin[dim1];
        int pos = position[dim1];
        long str = stride[dim1];

        shape[dim2] = shape[dim1];
        origin[dim2] = origin[dim1];
        stride[dim2] = stride[dim1];
        position[dim2] = position[dim1];

        shape[dim2] = sha;
        origin[dim2] = ori;
        stride[dim2] = str;
        position[dim2] = pos;

        index = new DefaultIndex(array.getFactoryName(), shape, origin, shape);
        index.setStride(stride);
        index.set(position);
        array.setIndex(index);
        return array.getArrayUtils();
    }

    @Override
    public boolean isConformable(IArray array) {
        boolean result = false;
        if (array.getRank() == getArray().getRank()) {
            IArray copy1 = this.reduce().getArray();
            IArray copy2 = array.getArrayUtils().reduce().getArray();

            int[] shape1 = copy1.getShape();
            int[] shape2 = copy2.getShape();

            for (int i = 0; i < shape1.length; i++) {
                if (shape1[i] != shape2[i]) {
                    result = false;
                    break;
                }
            }

        }
        return result;
    }


    @Override
    public IArrayUtils flip(int dim) {
        IArray array = getArray().copy(false);
        IIndex index = array.getIndex();
        int rank = array.getRank();
        int[] shape = index.getShape();
        int[] origin = index.getOrigin();
        int[] position = index.getCurrentCounter();
        long[] stride = index.getStride();

        int[] newShape = new int[rank];
        int[] newOrigin = new int[rank];
        int[] newPosition = new int[rank];
        long[] newStride = new long[rank];

        for (int i = 0; i < rank; i++) {
            newShape[i] = shape[rank - 1 - i];
            newOrigin[i] = origin[rank - 1 - i];
            newStride[i] = stride[rank - 1 - i];
            newPosition[i] = position[rank - 1 - i];
        }

        index = new DefaultIndex(array.getFactoryName(), newShape, newOrigin, newShape);
        index.setStride(newStride);
        index.set(newPosition);
        array.setIndex(index);
        return array.getArrayUtils();
    }

    @Override
    public IArrayUtils permute(int[] dims) {
        IArray array = getArray().copy(false);
        IIndex index = array.getIndex();
        int rank = array.getRank();
        int[] shape = index.getShape();
        int[] origin = index.getOrigin();
        int[] position = index.getCurrentCounter();
        long[] stride = index.getStride();
        int[] newShape = new int[rank];
        int[] newOrigin = new int[rank];
        int[] newPosition = new int[rank];
        long[] newStride = new long[rank];
        for (int i = 0; i < rank; i++) {
            newShape[i] = shape[dims[i]];
            newOrigin[i] = origin[dims[i]];
            newStride[i] = stride[dims[i]];
            newPosition[i] = position[dims[i]];
        }

        index = new DefaultIndex(array.getFactoryName(), newShape, newOrigin, newShape);
        index.setStride(newStride);
        index.set(newPosition);
        array.setIndex(index);
        return array.getArrayUtils();
    }

    // --------------------------------------------------
    // private methods
    // --------------------------------------------------
    /**
     * Copy the backing storage of this NxsArray into multidimensional corresponding Java primitive
     * array
     */
    private Object copyMatrixItemsToMultiDim() {
        // Create an array corresponding to the shape
        DefaultCompositeArray array = (DefaultCompositeArray) getArray();
        int[] shape = array.getShape();
        Object result = java.lang.reflect.Array.newInstance(array.getElementType(), shape);

        // Get the array's storage
        Object values = array.getStorage();

        int[] current;
        int length;
        Long startCell;

        try {

            ISliceIterator iter = array.getSliceIterator(1);
            DefaultCompositeIndex startIdx = (DefaultCompositeIndex) array.getIndex();
            DefaultIndex storage = startIdx.getIndexStorage();
            DefaultIndex items = startIdx.getIndexMatrix();
            startIdx.setOrigin(new int[startIdx.getRank()]);

            length = startIdx.getShape()[startIdx.getRank() - 1];

            storage.set(new int[storage.getRank()]);

            // Turning buffers
            Object slab = null;
            Object dataset = null;

            // Copy each slice
            int last = -1;
            int cell = (int) (items.currentElement() - items.firstElement());
            while (iter.hasNext()) {
                iter.next();
                slab = result;

                // Getting the right slab in the multidimensional resulting array
                current = iter.getSlicePosition();
                startIdx.set(current);
                for (int pos = 0; pos < current.length - 1; pos++) {
                    slab = java.lang.reflect.Array.get(slab, current[pos]);
                }

                cell = (int) (items.currentElement() - items.firstElement());
                if (last != cell) {
                    dataset = java.lang.reflect.Array.get(values, cell);
                    last = cell;
                }
                startCell = storage.currentElement();
                System.arraycopy(dataset, startCell.intValue(), slab, 0, length);
            }
        }
        catch (ShapeNotMatchException e) {
        }
        catch (InvalidRangeException e) {
        }
        return result;
    }

    @Override
    public IArrayUtils createArrayUtils(IArray array) {
        IArrayUtils result;
        if (array instanceof DefaultCompositeArray) {
            result = new DefaultCompositeArrayUtils((DefaultCompositeArray) array);
        }
        else {
            result = array.getArrayUtils();
        }
        return result;
    }
}