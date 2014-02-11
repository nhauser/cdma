/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
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
