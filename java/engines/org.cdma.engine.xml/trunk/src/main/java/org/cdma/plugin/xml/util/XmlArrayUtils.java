package org.cdma.plugin.xml.util;

import java.util.List;

import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.IRange;
import org.cdma.plugin.xml.array.XmlArray;
import org.cdma.utilities.memory.DefaultIndex;
import org.cdma.utils.ArrayUtils;
import org.cdma.utils.IArrayUtils;

public class XmlArrayUtils extends ArrayUtils {

	public XmlArrayUtils(XmlArray array) {
		super(array);
	}

	@Override
	public Object get1DJavaArray(Class<?> wantType) {
		// Instantiate a new convenient array for storage
		int length = ((Long) getArray().getSize()).intValue();
		Class<?> type = getArray().getElementType();
		Object array = java.lang.reflect.Array.newInstance(type, length);

		int start = 0;

		start = ((DefaultIndex) getArray().getIndex()).currentProjectionElement();
		System.arraycopy(getArray().getStorage(), start, array, 0, length);

		return array;
	}

	@Override
	public IArrayUtils sectionNoReduce(List<IRange> ranges)
			throws InvalidRangeException {
		// TODO Auto-generated method stub
		return null;
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

	@Override
	public IArrayUtils createArrayUtils(IArray array) {
		IArrayUtils result;
		if (array instanceof XmlArray) {
			result = new XmlArrayUtils( (XmlArray) array );
		} else {
			result = array.getArrayUtils();
		}
		return result;
	}

}
