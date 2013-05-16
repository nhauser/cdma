package org.gumtree.data.impl.edf;

import java.io.File;

import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.edf.navigation.EdfDataset;
import org.junit.Ignore;

@Ignore
public class EdfDatasetTest {

    public static void printGroup(IGroup parent, int tabs) throws Exception {
        if (parent != null) {
            System.out.println();
            printTabs(tabs);
            System.out.println("Treating group: " + (parent.isRoot() ? "ROOT" : parent.getName()));
            for (IDataItem item : parent.getDataItemList()) {
                printTabs(tabs);
                System.out.println("\tTreating item: " + item.getName());
                if (item.getData() != null) {
                    printTabs(tabs);
                    System.out.print("\t\titem data shape: [");
                    for (int i = 0; i < item.getData().getShape().length; i++) {
                        System.out.print(item.getData().getShape()[i]);
                        if (i < item.getData().getShape().length - 1) {
                            System.out.print(", ");
                        }
                    }
                    System.out.println("]");
                    printTabs(tabs);
                    System.out.println("\t\titem data value: " + item.getData().getStorage());
                }
                for (IAttribute attribute : item.getAttributeList()) {
                    printTabs(tabs);
                    System.out.println("\t\tTreating attribute: " + attribute.getName());
                    if (attribute.getValue() != null) {
                        printTabs(tabs);
                        System.out.print("\t\t\tattribute value shape: [");
                        for (int i = 0; i < attribute.getValue().getShape().length; i++) {
                            System.out.print(attribute.getValue().getShape()[i]);
                            if (i < attribute.getValue().getShape().length - 1) {
                                System.out.print(", ");
                            }
                            System.out.println("]");
                            printTabs(tabs);
                            System.out.println("\t\t\tattribute value value: "
                                    + attribute.getValue().getStorage());
                        }
                    }
                }
            }
            for (IGroup group : parent.getGroupList()) {
                printGroup(group, tabs + 1);
            }
        }
    }

    public static void printTabs(int tabs) {
        for (int i = 0; i < tabs; i++) {
            System.out.print("\t");
        }
    }

    public static long getSize(File file) {
        long size = 0;
        if (file.isFile()) {
            size = file.length();
        }
        else {
            for (File subFile : file.listFiles()) {
                size += getSize(subFile);
            }
        }
        return size;
    }

    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        long time = System.currentTimeMillis();
        long size = 0;
        if (args.length > 0) {
            size = getSize(new File(args[0]));
            EdfDataset dataset = new EdfDataset(args[0]);
            time = System.currentTimeMillis();
            System.out.println("EdfDatasetTest: treating directory " + dataset.getLocation());
            dataset.open();
            printGroup(dataset.getRootGroup(), 0);
        }
        System.out.println("Test finished");
        time = System.currentTimeMillis() - time;
        long hours, minutes, seconds, milli;
        long timeInSec = time / 1000;
        hours = timeInSec / 3600;
        minutes = (timeInSec % 3600) / 60;
        seconds = timeInSec % 60;
        milli = time % 1000;
        System.out
        .println(("Ellapsed time: " + (hours > 0 ? hours + "h " : "")
                + (minutes > 0 ? minutes + "mn " : "")
                + (seconds > 0 ? seconds + "s " : "") + (milli > 0 ? milli + "ms " : ""))
                .trim());
        String unit;
        double readableSize;
        if (size > 1024) {
            readableSize = size / 1024d;
            if (readableSize > 1024) {
                readableSize /= 1024d;
                if (readableSize > 1024) {
                    readableSize /= 1024d;
                    unit = " TB";
                }
                else {
                    unit = " MB";
                }
            }
            else {
                unit = " KB";
            }
        }
        else {
            readableSize = size;
            unit = " B";
        }
        System.out.print("Total size: " + readableSize + unit);
    }
}
