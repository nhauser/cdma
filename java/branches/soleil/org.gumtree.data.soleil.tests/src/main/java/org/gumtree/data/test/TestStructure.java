package org.gumtree.data.test;

import java.util.List;

import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IGroup;

public class TestStructure {

    public static void displayStructure( IGroup group ) {
        System.out.println("Display physical structure: ");
        displayStructure(group, "");
    }

    public static void displayStructure( IGroup group, String indent ) {
        final long time = System.currentTimeMillis();
        System.out.println( indent + "--> Group: " + group.getShortName() );
        List<IGroup> groups = group.getGroupList();
        for( IGroup item : groups ) {
            displayStructure( item, indent + " |" );
        }

        List<IDataItem> items = group.getDataItemList();
        for( IDataItem item : items ) {
            System.out.println( indent + " |--> Item: " + item.getShortName() );
        }
        System.out.println(indent + " |--> time >>>>>>> " + ((System.currentTimeMillis() - time ) / 1000.) + " s");
    }

}
