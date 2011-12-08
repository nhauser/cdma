package org.gumtree.data.soleil.utils;

import org.gumtree.data.engine.jnexus.array.NexusIndex;
import org.gumtree.data.engine.jnexus.utils.NexusArrayUtils;
import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.interfaces.ISliceIterator;
import org.gumtree.data.soleil.array.NxsArray;
import org.gumtree.data.soleil.array.NxsIndex;

public class NxsArrayUtils extends NexusArrayUtils {
	
	public NxsArrayUtils( NxsArray array) {
		super(array);
	}

	@Override
	public Object copyTo1DJavaArray() {
		// Instantiate a new convenient array for storage
		int length   = ((Long) getArray().getSize()).intValue();
		Class<?> type = getArray().getElementType();
		Object array = java.lang.reflect.Array.newInstance(type, length);
		
		// If the storing array is a stack of DataItem
		Long nbMatrixCells  = ((NxsIndex) getArray().getIndex()).getIndexMatrix().getSize();
		Long nbStorageCells = ((NxsIndex) getArray().getIndex()).getIndexStorage().getSize();

		Object fullArray = getArray().getStorage();
		Object partArray = null;
		for( int i = 0; i < nbMatrixCells; i++ ) {
			partArray = java.lang.reflect.Array.get(fullArray, i);
			System.arraycopy(partArray, 0, array, i * nbStorageCells.intValue(), nbStorageCells.intValue());
		}
			
		return array;
	}

	@Override
	public Object copyToNDJavaArray() {
		return copyMatrixItemsToMultiDim();
	}
	
	// --------------------------------------------------
	// tools methods
	// --------------------------------------------------
	static public Object copyJavaArray(Object array) {
		Object result = array;
		if( result == null )
			return null;
		
		// Determine rank of array (by parsing data array class name)
		String sClassName = array.getClass().getName();
		int iRank  = 0;
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
		int[] shape    = new int[iRank];

		// Fill dimension size array
		for ( int i = 0; i < iRank; i++) {
			shape[i] = java.lang.reflect.Array.getLength(result);
			result = java.lang.reflect.Array.get(result,0);
		}
		
		// Define a convenient array (shape and type)
		result = java.lang.reflect.Array.newInstance(array.getClass().getComponentType(), shape);
		
		return copyJavaArray(array, result);
	}
	
	static public Object copyJavaArray(Object source, Object target) {
		Object item = java.lang.reflect.Array.get(source, 0);
		int length = java.lang.reflect.Array.getLength(source);

		if( item.getClass().isArray() ) {
			Object tmpSrc;
			Object tmpTar;
			for( int i = 0; i < length; i++ ) {
				tmpSrc = java.lang.reflect.Array.get(source, i);
				tmpTar = java.lang.reflect.Array.get(target, i);
				copyJavaArray( tmpSrc, tmpTar);
			}
		}
		else {
			System.arraycopy(source, 0, target, 0, length);
		}
		
		return target;
	}
	
	
	// --------------------------------------------------
	// private methods
	// --------------------------------------------------
	/**
	 * Copy the backing storage of this NxsArray into multidimensional 
	 * corresponding Java primitive array
	 */
	private Object copyMatrixItemsToMultiDim() {
		NxsArray array = (NxsArray) getArray();
		int[] shape  = array.getShape();
		int[] current;
		int   length;
		int   startCell;
		Object result = java.lang.reflect.Array.newInstance(array.getElementType(), shape);
		Object slab;
		Object dataset;
		
		ISliceIterator iter;
		try {
			iter = array.getSliceIterator(1);
			NxsIndex startIdx = (NxsIndex) array.getIndex().clone();
			NexusIndex storage = startIdx.getIndexStorage();
			NexusIndex items   = startIdx.getIndexMatrix();
			startIdx.setOrigin(new int[startIdx.getRank()]);
			int i = 0;
			while( iter.hasNext() ) {
				length = ((Long) iter.getArrayNext().getSize()).intValue();
				slab = result;
				
				// Getting the right slab in the multidim result array
				current = iter.getSlicePosition();
				startIdx.set(current);
				for( int pos = 0;  pos < current.length - 1; pos++ ) {
					slab = java.lang.reflect.Array.get(slab, current[pos]);
				}
				int tamere = items.currentProjectionElement();
				dataset = java.lang.reflect.Array.get(array.getStorage(), tamere);
				
				startCell = storage.currentProjectionElement();
				System.arraycopy(dataset, startCell, slab, 0, length);
			}
		} catch (ShapeNotMatchException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidRangeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return result;
	}
}
