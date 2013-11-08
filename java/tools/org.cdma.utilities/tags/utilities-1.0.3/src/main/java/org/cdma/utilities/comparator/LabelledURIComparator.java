package org.cdma.utilities.comparator;

import java.util.Comparator;

import org.cdma.utilities.LabelledURI;

public abstract class LabelledURIComparator implements Comparator<LabelledURI> {

    @Override
    public int compare(LabelledURI o1, LabelledURI o2) {
        int result = Integer.MAX_VALUE;

        if (o1 == null && o2 == null) {
            result = 0;
        } else if (o1 == null) {
            result = -1;
        } else if (o2 == null) {
            result = 1;
        }
        return result;
    }

}
