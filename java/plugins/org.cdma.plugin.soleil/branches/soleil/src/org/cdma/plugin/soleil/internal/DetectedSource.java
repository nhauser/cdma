package org.cdma.plugin.soleil.internal;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.net.URI;

import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.navigation.NxsDataset;
import org.cdma.utilities.configuration.ConfigDataset;

public class DetectedSource {
    private static final int EXTENSION_LENGTH = 4;
    private static final String EXTENSION = ".nxs";
    private static final String CREATOR = "Synchrotron SOLEIL";
    private static final String[] BEAMLINES = new String[] { "CONTACQ", "AILES", "ANTARES",
            "CASSIOPEE", "CRISTAL", "DIFFABS", "DEIMOS", "DESIRS", "DISCO", "GALAXIES", "LUCIA",
            "MARS", "METROLOGIE", "NANOSCOPIUM", "ODE", "PLEIADES", "PROXIMA1", "PROXIMA2",
            "PSICHE", "SAMBA", "SEXTANTS", "SIRIUS", "SIXS", "SMIS", "TEMPO", "SWING" };

    public static final class NeXusFilter implements FilenameFilter {

        public boolean accept(File dir, String name) {
            return (name.endsWith(EXTENSION));
        }
    }

    private boolean mIsDataset;
    private boolean mIsExperiment;
    private boolean mIsBrowsable;
    private boolean mIsProducer;
    private boolean mIsReadable;
    private boolean mIsFolder;
    private URI mURI;

    public DetectedSource(URI uri, boolean browsable, boolean readable, boolean producer, boolean experiment, boolean datasetFolder) {
        mIsReadable = readable;
        mIsProducer = producer;
        mIsBrowsable = browsable;
        mIsExperiment = experiment;
        mIsDataset = datasetFolder;
        mURI = uri;
    }

    public DetectedSource(URI uri) {
        mURI = uri;
        init(uri);
    }

    public URI getURI() {
        return mURI;
    }

    public boolean isDatasetFolder() {
        return mIsDataset;
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

    public boolean isFolder() {
        return mIsFolder;
    }

    // ---------------------------------------------------------
    // / private methods
    // ---------------------------------------------------------
    private void init(URI uri) {
        if (uri != null) {
            // Check if the uri is a folder
            File file = new File(uri.getPath());
            if (file.isDirectory()) {
                mIsDataset = isDatasetFolder(file);
                mIsFolder = true;
            }
            else {
                mIsDataset = false;
                mIsFolder = false;
            }

            // Check it is a NeXus file
            mIsReadable = initReadable(uri);

            // Check if we are producer of the source
            mIsProducer = initProducer(uri);

            // Check if the uri corresponds to dataset experiment
            mIsExperiment = initExperiment(uri);

            // Check if the URI is considered as browsable
            mIsBrowsable = initBrowsable(uri);
        }

    }

    private boolean initReadable(URI uri) {

        boolean result = false;
        if (mIsDataset) {
            result = true;
        }
        else {
            File file = new File(uri.getPath());
            String name = file.getName();
            int length = name.length();

            if( file.exists() && file.length() != 0L ) {
                // Check if the URI is a NeXus file
                if (length > EXTENSION_LENGTH && name.substring(length - EXTENSION_LENGTH).equals(EXTENSION)) {
                    result = true;
                }
                else {
                    result = false;
                }
            }
        }

        return result;
    }

    private boolean initProducer(URI uri) {
        boolean result = false;
        if (mIsReadable) {
            File file = new File(uri.getPath());
            IDataset dataset = null;
            try {
                // instantiate
                dataset = NxsDataset.instanciate(file.toURI());

                // open file
                dataset.open();

                // seek at root for 'creator' attribute
                IGroup group = dataset.getRootGroup();
                if (group.hasAttribute("creator", CREATOR)) {
                    result = true;
                }
                else {
                    group = group.getGroup("<NXentry>");
                    if (group != null) {
                        group = group.getGroup("<NXinstrument>");
                    }

                    if (group != null) {
                        String node = group.getShortName();

                        for (String name : BEAMLINES) {
                            if (node.equalsIgnoreCase(name)) {
                                result = true;
                                break;
                            }
                        }
                    }
                }

                // close file
                dataset.close();

            }
            catch (IOException e) {
                // close file
                if (dataset != null) {
                    try {
                        dataset.close();
                    }
                    catch (IOException e1) {
                    }
                }
            }
            catch (NoResultException e) {
                e.printStackTrace();
            }
        }
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
            }
            catch (NoResultException e) {
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
        if (files.length > 0) {
            try {
                IDataset dataset = NxsDataset.instanciate(file.toURI());
                IGroup group = dataset.getRootGroup();
    
                IContainer groups = group.findContainerByPath("/<NXentry>/<NXdata>");
                if( groups instanceof IGroup ) {
                    for( IDataItem item : ((IGroup) groups).getDataItemList() ) {
                        if( item.getAttribute( "dataset_part" ) != null ) {
                            result = true;
                            break;
                        }
                    }
                }
            } catch (NoResultException e) {
            }
        }
        
        return result;
    }
}
