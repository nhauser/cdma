package org.cdma.plugin.soleil.navigation;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.lang.ref.SoftReference;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.dictionary.ExtendedDictionary;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.nexus.navigation.NexusDataset;
import org.cdma.engine.nexus.navigation.NexusGroup;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.dictionary.NxsLogicalGroup;
import org.cdma.plugin.soleil.internal.DetectedSource.NeXusFilter;
import org.cdma.plugin.soleil.internal.NexusDatasetImpl;
import org.cdma.utilities.configuration.ConfigDataset;
import org.cdma.utilities.configuration.ConfigManager;
import org.cdma.utils.Utilities.ModelType;

public final class NxsDataset implements IDataset {
    private boolean            mOpen;         // is the dataset open
    private URI                mPath;         // URI of this dataset
    private ConfigDataset      mConfig;       // Configuration associated to this dataset
    private List<NexusDataset> mDatasets;     // NexusDataset compounding this NxsDataset
    private IGroup             mRootPhysical; // Physical root of the document
    private NxsLogicalGroup    mRootLogical;  // Logical root of the document
    
    // SoftReference of dataset associated to their URI
    private static Map<String, SoftReference<NxsDataset>> datasets;
    
    // datasets' URIs associated to the last modification
    private static Map<String, Long> lastModifications;

    public static NxsDataset instanciate(URI destination) throws NoResultException {
        NxsDataset dataset = null;
        if (datasets == null) {
            synchronized (NxsDataset.class) {
                if (datasets == null) {
                    datasets = new HashMap<String, SoftReference<NxsDataset>>();
                    lastModifications = new HashMap<String, Long>();
                }
            }
        }

        synchronized (datasets) {
            boolean resetBuffer = false;
            String uri = destination.toString();
            SoftReference<NxsDataset> ref = datasets.get( uri );
            if (ref != null) {
                dataset = ref.get();
                long last = lastModifications.get( uri );
                if( dataset != null ) {
                    if( last < dataset.getLastModificationDate() ) {
                        dataset = null;
                        resetBuffer = true;
                    }
                }
            }

            if (dataset == null) {
                try {
                    dataset = new NxsDataset(new File(destination.getPath()), resetBuffer);
                    String fragment = destination.getFragment();
    
                    if (fragment != null && !fragment.isEmpty()) {
                        IGroup group = dataset.getRootGroup();
                        try {
                            String path = URLDecoder.decode(fragment, "UTF-8");
                            for (IContainer container : group.findAllContainerByPath(path)) {
                                if (container.getModelType().equals(ModelType.Group)) {
                                    dataset.mRootPhysical = (IGroup) container;
                                    break;
                                }
                            }
                        }
                        catch (UnsupportedEncodingException e) {
                            Factory.getLogger().log( Level.WARNING, e.getMessage());
                        }
                    }
                    datasets.put( uri, new SoftReference<NxsDataset>(dataset));
                    lastModifications.put( uri, dataset.getLastModificationDate());
                }
                catch ( FileAccessException e ) {
                    throw new NoResultException( e );
                }
            }
        }
        return dataset;
    }

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public LogicalGroup getLogicalRoot() {
        if (mRootLogical == null) {
            String param;
            try {
                param = getConfiguration().getParameter(NxsFactory.DEBUG_INF);
                boolean debug = Boolean.parseBoolean(param);
                mRootLogical = new NxsLogicalGroup(null, null, this, debug);
            }
            catch (NoResultException e) {
                Factory.getLogger().log( Level.WARNING, e.getMessage());
            }
        }
        else {
            ExtendedDictionary dict = mRootLogical.getDictionary();
            if (dict != null && !dict.getView().equals(Factory.getActiveView())) {
                mRootLogical.setDictionary(mRootLogical.findAndReadDictionary());
            }
        }
        return mRootLogical;
    }

    @Override
    public IGroup getRootGroup() {
        if (mRootPhysical == null && mDatasets.size() > 0) {
            NexusGroup[] groups = new NexusGroup[mDatasets.size()];
            int i = 0;
            for (IDataset dataset : mDatasets) {
                groups[i++] = (NexusGroup) dataset.getRootGroup();
            }
            mRootPhysical = new NxsGroup(groups, null, this);
        }
        return mRootPhysical;
    }

    @Override
    public void saveTo(String location) throws WriterException {
        for (IDataset dataset : mDatasets) {
            dataset.saveTo(location);
        }
    }

    @Override
    public void save(IContainer container) throws WriterException {
        for (IDataset dataset : mDatasets) {
            dataset.save(container);
        }
    }

    @Override
    public void save(String parentPath, IAttribute attribute) throws WriterException {
        for (IDataset dataset : mDatasets) {
            dataset.save(parentPath, attribute);
        }
    }

    @Override
    public boolean sync() throws IOException {
        boolean result = true;
        for (IDataset dataset : mDatasets) {
            if (!dataset.sync()) {
                result = false;
            }
        }
        return result;
    }

    @Override
    public void close() throws IOException {
        mOpen = false;
    }

    @Override
    public String getLocation() {
        return mPath.toString();
    }

    @Override
    public String getTitle() {
        String title = getRootGroup().getShortName();
        if (title.isEmpty()) {
            try {
                title = mDatasets.get(0).getTitle();
            }
            catch (NoSuchElementException e) {
            }
        }

        return title;
    }

    @Override
    public void setLocation(String location) {
        if (location != null && !location.equals(mPath.toString())) {
            try {
                mPath = new URI(location);
            }
            catch (URISyntaxException e) {
                Factory.getLogger().log( Level.WARNING, e.getMessage());
            }
            mDatasets.clear();
        }
    }

    @Override
    public void setTitle(String title) {
        try {
            mDatasets.get(0).setTitle(title);
        }
        catch (NoSuchElementException e) {
        }
    }

    @Override
    public void open() throws IOException {
        mOpen = true;
    }

    @Override
    public void save() throws WriterException {
        for (IDataset dataset : mDatasets) {
            dataset.save();
        }
    }

    @Override
    public boolean isOpen() {
        return mOpen;
    }

    public ConfigDataset getConfiguration() throws NoResultException {
        if (mConfig == null) {
            if (mDatasets.size() > 0) {
                ConfigDataset conf;
                conf = ConfigManager.getInstance(NxsFactory.getInstance(), NxsFactory.CONFIG_FILE).getConfig(this);
                mConfig = conf;
            }
        }
        return mConfig;
    }

    @Override
    public long getLastModificationDate() {
        long last = 0;
        long temp = 0;
        
        File path = new File( mPath.getPath() );
        if( path.exists() && path.isDirectory() ) {
            last = path.lastModified();
        }

        for (NexusDataset dataset : mDatasets) {
            temp = dataset.getLastModificationDate();
            if (temp > last) {
                last = temp;
            }
        }

        return last;
    }

    // ---------------------------------------------------------
    // / Private methods
    // ---------------------------------------------------------
    private NxsDataset(File destination, boolean resetBuffer) throws FileAccessException {
        mPath = destination.toURI();
        mDatasets = new ArrayList<NexusDataset>();
        NexusDatasetImpl datafile;
        if (destination.exists() && destination.isDirectory()) {
            NeXusFilter filter = new NeXusFilter();
            for (File file : destination.listFiles(filter)) {
                datafile = new NexusDatasetImpl(file, resetBuffer);
                mDatasets.add((NexusDatasetImpl) datafile);
            }
        }
        else {
            datafile = new NexusDatasetImpl(destination, resetBuffer);
            mDatasets.add(datafile);
        }
        mOpen = false;
    }
    
    
}
