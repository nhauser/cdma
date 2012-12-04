package org.cdma.engine.sql.utils;

import java.util.List;

import org.cdma.engine.sql.array.SqlArray;
import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IRange;
import org.cdma.utils.ArrayUtils;
import org.cdma.utils.IArrayUtils;

public class SqlArrayUtils extends ArrayUtils {

	public SqlArrayUtils(SqlArray array) {
		super(array);
	}

	@Override
	public Object get1DJavaArray(Class<?> wantType) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArrayUtils sectionNoReduce(List<IRange> ranges)
			throws InvalidRangeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArrayUtils transpose(int dim1, int dim2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isConformable(IArray array) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public IArrayUtils flip(int dim) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArrayUtils permute(int[] dims) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArrayUtils createArrayUtils(IArray array) {
		// TODO Auto-generated method stub
		return null;
	}

}
