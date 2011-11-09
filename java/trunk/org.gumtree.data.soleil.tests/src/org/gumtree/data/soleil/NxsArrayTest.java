package org.gumtree.data.soleil;

import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.interfaces.ArrayTest;
import org.junit.Ignore;
import org.junit.Test;

public class NxsArrayTest extends ArrayTest {

	public NxsArrayTest() {
		setFactory(new NxsFactory());
	}
	
	@Test
	@Ignore("Not implemented")
	public void testTest1DIntArrayMathAddArray() throws ShapeNotMatchException {
		super.testTest1DIntArrayMathAddArray();
	}
	
}
