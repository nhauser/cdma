package org.cdma.utilities.comparator;

import org.cdma.utilities.LabelledURI;

public class LabelledURIDateComparator extends LabelledURIComparator {

    @Override
    public int compare(LabelledURI o1, LabelledURI o2) {
        // Deal with null objects
        int result = super.compare(o1, o2);

        // Both object are != null
        if (result == Integer.MAX_VALUE) {

            result = Long.valueOf(o2.getDatasource().getLastModificationDate(o2.getURI())).compareTo(
                    Long.valueOf(o1.getDatasource().getLastModificationDate(o1.getURI())));
        }
        return result;
    }
}
