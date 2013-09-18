package org.cdma.utilities.comparator;

import org.cdma.utilities.LabelledURI;

public class LabelledURINameComparator extends LabelledURIComparator {

    @Override
    public int compare(LabelledURI o1, LabelledURI o2) {
        // Deal with null objects
        int result = super.compare(o1, o2);

        // Both object are != null
        if (result == Integer.MAX_VALUE) {
            String label1 = o1.getLabel();
            String label2 = o2.getLabel();

            if (label1 == null && label2 == null) {
                result = -1;
            } else if (label2 == null) {
                result = 1;
            } else if (label1 == null) {
                result = 0;
            } else {
                result = label1.compareTo(label2);
            }
        }
        return result;
    }

}
