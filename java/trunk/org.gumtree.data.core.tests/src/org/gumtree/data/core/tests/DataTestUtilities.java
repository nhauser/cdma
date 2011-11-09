package org.gumtree.data.core.tests;

import static org.junit.Assert.*;

import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IArrayIterator;

public class DataTestUtilities {

	public static void compareArrays(IArray original, IArray copy) {
		// Test factory name
		assertEquals(original.getFactoryName(), copy.getFactoryName());
		// Test rank
		assertEquals(original.getRank(), copy.getRank());
		// Test shape
		assertArrayEquals(original.getShape(), copy.getShape());
		// Test type
		assertEquals(original.getElementType(), copy.getElementType());
		// Test content
		IArrayIterator iterArray1 = original.getIterator();
		IArrayIterator iterArray2 = copy.getIterator();
		long i = 0;
		while( iterArray1.hasNext() || iterArray2.hasNext() ) {
			Object obj1 = iterArray1.next();
			Object obj2 = iterArray2.next();
			assertEquals(
					"Invalid operation on compared content: original_cell[" + i
							+ "] = " + obj1 + " compared to copy_cell[" + i
							+ "] =  " + obj2, obj1, obj2);
			i++;
		}
	}
	
	private DataTestUtilities() {
		super();
	}
	
}
