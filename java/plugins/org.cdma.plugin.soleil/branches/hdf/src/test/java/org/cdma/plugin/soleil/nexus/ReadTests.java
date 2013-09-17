package org.cdma.plugin.soleil.nexus;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.List;

import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataset;
import org.junit.Test;

public class ReadTests {

    @Test
    public void aReadFile() throws IOException, NoResultException {
        IDataset dataSet = NxsDataset.instanciate(WriteTests.FIRST_FILE_TO_WRITE.toURI());
        IGroup root = dataSet.getRootGroup();
        if (root != null) {

            // Test ROOT
            List<IGroup> groupList = root.getGroupList();
            assertEquals(2, groupList.size());
            assertNull(root.getParentGroup());

            // Test IGroup
            IGroup group1 = root.getGroup("group1");
            assertNotNull(group1);
            assertEquals(2, group1.getAttributeList().size());
            assertEquals("Attribute List size", 2, group1.getAttributeList().size());
            assertEquals("group1", group1.getShortName());
            assertEquals("/group1", group1.getName());
            assertEquals(root, group1.getRootGroup());
            assertEquals(dataSet, group1.getDataset());
            assertTrue(group1.isEntry());
            assertFalse(group1.isRoot());

            // Test IDataItem
            IDataItem data1 = group1.getDataItem("data1");
            assertNotNull(data1);
            IArray iArray = data1.getData();
            Object storage = iArray.getStorage();

            Object[] arrays = (Object[]) storage;
            int[] array = (int[]) arrays[0];
            int[] expected = { 0, 1, 2, 3 };
            assertArrayEquals(expected, array);
            assertEquals(int.class, iArray.getElementType());
            assertEquals(root, data1.getRootGroup());
            assertEquals(group1, data1.getParentGroup());
            assertEquals(dataSet, data1.getDataset());
            assertEquals("/group1/data1", data1.getName());
            assertEquals("data1", data1.getShortName());

            // Test Navigation
            IContainer container = root.findContainerByPath("/group1/data1");
            assertNotNull(container);

            List<IContainer> containers = root.findAllContainerByPath("/group1/data1");
            assertNotNull(containers);
            assertTrue(containers.size() == 1);

            containers = root.findAllContainerByPath("/group*Modified/data*");
            assertNotNull(containers);
            assertTrue(containers.size() == 1);

            containers = root.findAllContainerByPath("/group*/data*");
            assertNotNull(containers);
            assertTrue(containers.size() == 2);
        }
    }

}
