/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.soleil.internal;

import java.io.File;
import java.io.FilenameFilter;
import java.net.URI;

import org.cdma.exception.FileAccessException;
import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.NxsDatasource;
import org.cdma.plugin.soleil.navigation.NxsDataset;
import org.cdma.utilities.configuration.ConfigDataset;

public class DetectedSource {
    private static final long MIN_LAST_MODIF_TIME = 5000;
    private static final int EXTENSION_LENGTH = 4;
    private static final String EXTENSION = ".nxs";
    private static final String CREATOR = "Synchrotron SOLEIL";
    private static final String[] BEAMLINES = new String[] { "CONTACQ", "AILES", "ANTARES", "CASSIOPEE", "CRISTAL",
        "DIFFABS", "DEIMOS", "DESIRS", "DISCO", "GALAXIES", "LUCIA", "MARS", "METROLOGIE", "NANOSCOPIUM", "ODE",
        "PLEIADES", "PROXIMA1", "PROXIMA2", "PSICHE", "SAMBA", "SEXTANTS", "SIRIUS", "SIXS", "SMIS", "TEMPO",
        "SWING", "ROCK", "PUMA", "ANATOMIX", "HERMES" };

    public static class NeXusFilter implements FilenameFilter {
        @Override
        public boolean accept(File dir, String name) {
            return DetectedSource.accept(name);
        }
    }

    private boolean mIsDataset;
    private boolean mIsExperiment;
    private boolean mIsBrowsable;
    private boolean mIsProducer;
    private boolean mIsReadable;
    private boolean mIsFolder;
    private boolean mInitialized;
    private final URI mURI;
    private long mTimestamp;

    public DetectedSource(URI uri, boolean browsable, boolean readable, boolean producer, boolean experiment,
            boolean datasetFolder) {
        mIsReadable = readable;
        mIsProducer = producer;
        mIsBrowsable = browsable;
        mIsExperiment = experiment;
        mIsDataset = datasetFolder;
        mURI = uri;
        mInitialized = true;
        mTimestamp = getTimestamp(uri);
    }

    public DetectedSource(URI uri) {
        mURI = uri;
        init(uri);
    }

    private long getTimestamp(URI uri) {
        long result = Integer.MIN_VALUE;
        String path = uri.getPath();
        if (path != null) {
            File file = new File(path);
            result = file.lastModified();
        }
        return result;
    }

    public URI getURI() {
        return mURI;
    }

    public boolean isDatasetFolder() {
        mIsDataset = isDatasetFolder(new File(mURI.getPath()));
        return mIsDataset;
    }

    public boolean isExperiment() {
        // For NeXus SOLEIL: experience implies a fragment in URI
        if (mURI.getFragment() != null) {
            fullInit();
        }
        return mIsExperiment;
    }

    public boolean isBrowsable() {
        fullInit();
        return mIsBrowsable;
    }

    public boolean isProducer() {
        return true;
        // fullInit();
        // return mIsProducer;
    }

    public boolean isReadable() {
        if (!mIsReadable) {
            mIsReadable = initReadable(mURI);
        }
        return mIsReadable;
    }

    public boolean isFolder() {
        return mIsFolder;
    }

    public void setInitialized(boolean valueToSet) {
        mInitialized = valueToSet;
    }

    /**
     * Return true if the source hasn't been modified since a while and is
     * considered as stable.
     */
    public boolean hasChanged(URI uri) {
        boolean result = false;
        long currentTimestamp = getTimestamp(uri);
        if (currentTimestamp != mTimestamp) {
            result = true;
        }
        return result;
    }

    // ---------------------------------------------------------
    // / private methods
    // ---------------------------------------------------------
    private void init(URI uri) {
        if (uri != null) {
            // Check if the uri is a folder
            String path = uri.getPath();
            if (path != null) {
                File file = new File(path);
                mTimestamp = file.lastModified();
                if (file.isDirectory()) {

                    mIsFolder = true;
                } else {
                    mIsFolder = false;
                }

                // Check it is a NeXus file
                mIsReadable = initReadable(uri);
            } else {
                mIsReadable = false;
                mIsProducer = false;
                mIsBrowsable = false;
                mIsExperiment = false;
                mInitialized = true;
            }
        }
    }

    private void fullInit() {
        synchronized (this) {
            if (!mInitialized || hasChanged(mURI)) {
                // Check if we are producer of the source
                mIsProducer = initProducer(mURI);

                // Check if the uri corresponds to dataset experiment
                mIsExperiment = initExperiment(mURI);

                // Check if the URI is considered as browsable
                mIsBrowsable = initBrowsable(mURI);

                mInitialized = true;
            }
        }
    }

    private boolean initReadable(URI uri) {
        boolean result = false;

        File file = new File(uri.getPath());
        String name = file.getName();

        if (file.exists()) {
            // Check if the URI is a NeXus file
            if (DetectedSource.accept(name)) {
                result = true;
            }
        }
        return result;
    }

    private boolean initProducer(URI uri) {

        boolean result = false;
        if (mIsReadable) {
            result = true;
        }

        //            File file = new File(uri.getPath());
        // IDataset dataset = null;
        //
        // try {
        // // instantiate
        // dataset = NxsDataset.instanciate(file.toURI());
        // // open file
        // dataset.open();
        //
        // // seek at root for 'creator' attribute
        //
        // IGroup group = dataset.getRootGroup();
        // if (group.hasAttribute("creator", CREATOR)) {
        // result = true;
        // } else {
        // group = group.getGroup("<NXentry>");
        // if (group != null) {
        // group = group.getGroup("<NXinstrument>");
        // }
        //
        // if (group != null) {
        // String node = group.getShortName();
        //
        // for (String name : BEAMLINES) {
        // if (node.equalsIgnoreCase(name)) {
        // result = true;
        // break;
        // }
        // }
        // }
        // }
        // // close file
        // dataset.close();
        //
        // } catch (IOException e) {
        // // close file
        // if (dataset != null) {
        // try {
        // dataset.close();
        // } catch (IOException e1) {
        // }
        // }
        // } catch (NoResultException e) {
        // Factory.getLogger().log(Level.WARNING, e.getMessage());
        // }
        // }
        return result;
    }

    private boolean initExperiment(URI uri) {
        boolean result = false;
        // Check if the URI is a NeXus file
        if (mIsProducer) {
            try {
                // Instantiate the dataset and detect its configuration
                NxsDataset dataset = NxsDataset.instanciate(uri);

                // Interrogate the config to know the experiment path
                ConfigDataset conf = dataset.getConfiguration();
                if (conf != null) {
                    result = true;
                }
            } catch (NoResultException e) {
            }
        }
        return result;
    }

    private boolean initBrowsable(URI uri) {
        boolean result = false;

        // If experiment not browsable
        if (!mIsExperiment) {
            // If it is a folder containing split NeXus file (quick_exaf)
            if (mIsFolder || mIsProducer) {
                result = true;
            }
        }
        return result;
    }

    /**
     * Return true if the given is a folder dataset.
     * 
     * @note the given file must be a folder
     */
    private boolean isDatasetFolder(File file) {
        boolean result = false;

        NeXusFilter filter = new NeXusFilter();
        File[] files = file.listFiles(filter);
        if (files != null && files.length > 0) {
            try {
                NxsDatasource source = NxsDatasource.getInstance();
                DetectedSource detect = source.getSource(files[0].toURI());

                if (detect.isProducer()) {
                    IDataset dataset = new NexusDatasetImpl(files[0], false);
                    IGroup group = dataset.getRootGroup();

                    IContainer groups = group.findContainerByPath("/<NXentry>/<NXdata>");
                    if (groups instanceof IGroup) {
                        for (IDataItem item : ((IGroup) groups).getDataItemList()) {
                            if (item.getAttribute("dataset_part") != null) {
                                result = true;
                                break;
                            }
                        }
                    }
                }
            } catch (NoResultException e) {
            } catch (FileAccessException e) {
            }
        }
        return result;
    }

    private static boolean accept(String filename) {
        int length = filename.length();
        return (length > EXTENSION_LENGTH && filename.substring(length - EXTENSION_LENGTH).equalsIgnoreCase(EXTENSION));
    }
}
