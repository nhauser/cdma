package org.cdma.plugin.edf.utils;

import java.io.File;
import java.text.Collator;
import java.util.Comparator;

public class FileComparator implements Comparator<File> {

    public FileComparator() {
        super();
    }

    @Override
    public int compare(File o1, File o2) {
        String name1, name2;
        if (o1 == null) {
            name1 = null;
        }
        else {
            name1 = o1.getName();
        }
        if (o2 == null) {
            name2 = null;
        }
        else {
            name2 = o2.getName();
        }
        return Collator.getInstance().compare(name1, name2);
    }

}
