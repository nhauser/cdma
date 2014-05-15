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
package org.cdma.plugin.soleil.edf.internal;

import java.io.File;
import java.net.URI;

import org.cdma.plugin.soleil.edf.EdfDatasource;

public class DetectedSource {
    private boolean mIsExperiment;
    private boolean mIsBrowsable;
    private boolean mIsProducer;
    private boolean mIsReadable;
    private final URI mURI;

    public DetectedSource(URI uri) {
        mURI = uri;
        init(uri);
    }

    public URI getURI() {
        return mURI;
    }

    public boolean isExperiment() {
        return mIsExperiment;
    }

    public boolean isBrowsable() {
        return mIsBrowsable;
    }

    public boolean isProducer() {
        return mIsProducer;
    }

    public boolean isReadable() {
        return mIsReadable;
    }

    // ---------------------------------------------------------
    // / private methods
    // ---------------------------------------------------------
    private void init(URI uri) {
        mIsExperiment = false;
        mIsBrowsable = false;
        mIsProducer = false;
        mIsReadable = false;

        if (uri != null) {
            String path = uri.getPath();
            if (path != null) {
                File file = new File(path);
                if (file.exists()) {
                    mIsBrowsable = file.isDirectory();
                    mIsReadable = true;
                    //                    if (mIsBrowsable){
                    //                        if (EdfDatasource.findEDFFiles(file)){
                    //                            mIsReadable = true;
                    //                        }
                    //                    }
                    // ValidURIFilter filter = new ValidURIFilter();
                    String fileName = file.getPath();
                    int length = fileName.length();
                    boolean accept = (length > EdfDatasource.EXTENSION.length() && fileName.substring(
                            length - EdfDatasource.EXTENSION.length()).equalsIgnoreCase(EdfDatasource.EXTENSION));
                    if (accept) {
                        mIsBrowsable = true;
                        mIsReadable = true;
                        mIsProducer = true;
                        mIsExperiment = true;
                    }
                }
            }
        }
    }
}
