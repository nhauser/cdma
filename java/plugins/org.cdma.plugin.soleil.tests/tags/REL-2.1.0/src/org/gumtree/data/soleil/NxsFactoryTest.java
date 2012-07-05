package org.gumtree.data.soleil;

import java.io.IOException;

import org.gumtree.data.FactoryTest;
import org.junit.Ignore;
import org.junit.Test;

public class NxsFactoryTest extends FactoryTest {

	public NxsFactoryTest() {
		setFactory(new NxsFactory());
	}

	@Test
	@Ignore("Not implmeneted")
	public void testOpenDataset() throws Exception {
		super.testOpenDataset();
	}
	
	@Test
	@Ignore("Not implmeneted")
	public void testCreateEmptyDatasetInstance() throws IOException {
		super.testCreateEmptyDatasetInstance();
	}
	
	@Test
	@Ignore("Not implmeneted")
	public void testCreateSingleGroup() throws IOException {
		super.testCreateSingleGroup();
	}
	
	@Test
	@Ignore("Not implmeneted")
	public void testCreateAndAttachGroup() throws IOException {
		super.testCreateAndAttachGroup();
	}
	
}
