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
package org.cdma.utilities;

import java.net.URI;

import org.cdma.interfaces.IDatasource;

public class LabelledURI {
    private final String mLabel;
    private final URI mURI;
    private final IDatasource mSource;
    private final Long mLastModificationDate;

    public LabelledURI(String label, URI uri, IDatasource datasource) {
        mLabel = label;
        mURI = uri;
        mSource = datasource;
        mLastModificationDate = datasource.getLastModificationDate(mURI);
    }

    public LabelledURI(URI uri, IDatasource datasource) {
        this(extractLabel(uri, datasource), uri, datasource);
    }

    public URI getURI() {
        return mURI;
    }

    public String getLabel() {
        return mLabel;
    }

    public IDatasource getDatasource() {
        return mSource;
    }

    protected static String extractLabel(URI uri, IDatasource datasource) {
        String label;
        if (datasource != null) {
            String[] parts = datasource.getURIParts(uri);
            if (parts != null && parts.length > 0) {
                label = parts[parts.length - 1];
            } else {
                label = String.valueOf(uri);
            }
        } else {
            label = String.valueOf(uri);
        }
        return label;
    }

    @Override
    public boolean equals(Object obj) {
        boolean equals;
        if (obj == this) {
            equals = true;
        } else if (obj instanceof LabelledURI) {
            LabelledURI toCompare = (LabelledURI) obj;
            equals = sameObject(mLabel, toCompare.mLabel) && sameObject(mURI, toCompare.mURI)
                    && sameSource(mSource, toCompare.mSource)
                    && sameObject(mLastModificationDate, toCompare.mLastModificationDate);
        } else {
            equals = false;
        }
        return equals;
    }

    @Override
    public int hashCode() {
        int code = 0x7AB;
        int mult = 0xE77ED;
        code = code * mult + getCode(mLabel);
        code = code * mult + getCode(mURI);
        code = code * mult + getSourceCode(mSource);
        return code;
    }

    protected boolean sameSource(IDatasource d1, IDatasource d2) {
        boolean same;
        if (d1 == null) {
            same = (d2 == null);
        } else if (d2 == null) {
            same = false;
        } else {
            same = d1.getClass().equals(d2.getClass());
        }
        return same;
    }

    protected boolean sameObject(Object o1, Object o2) {
        boolean same;
        if (o1 == null) {
            same = (o2 == null);
        } else if (o2 == null) {
            same = false;
        } else {
            same = o1.equals(o2);
        }
        return same;
    }

    protected int getSourceCode(IDatasource d) {
        return (d == null ? 0 : d.getClass().hashCode());
    }

    protected int getCode(Object obj) {
        return (obj == null ? 0 : obj.hashCode());
    }

    @Override
    public String toString() {
        return mLabel;
    }
}
