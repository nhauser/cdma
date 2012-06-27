package org.cdma.plugin.soleil.navigation;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import org.cdma.Factory;
import org.cdma.dictionary.ExtendedDictionary;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.nexus.navigation.NexusDataset;
import org.cdma.engine.nexus.navigation.NexusGroup;
import org.cdma.exception.NoResultException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.NxsDatasource.NeXusFilter;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.dictionary.NxsLogicalGroup;
import org.cdma.utilities.configuration.ConfigDataset;
import org.cdma.utilities.configuration.ConfigManager;

public final class NxsDataset implements IDataset {
    // ---------------------------------------------------------
    /// Inner class that concretes the abstract NexusDataset 
    // ---------------------------------------------------------
    private class NexusDatasetImpl extends NexusDataset {
        public NexusDatasetImpl(NexusDatasetImpl dataset) {
            super(dataset);
        }

        public NexusDatasetImpl( File nexusFile ) {
            super( NxsFactory.NAME, nexusFile );
        }

        @Override
        public LogicalGroup getLogicalRoot() {
            return new LogicalGroup(null, null, this, false);
        }
    }

    private ConfigDataset      mConfig;
    private List<NexusDataset> mDatasets;     // all found datasets in the folder
    private URI                mPath;         // folder containing all datasets
    private IGroup             mRootPhysical; // Physical root of the document 
    private NxsLogicalGroup    mRootLogical;  // Logical root of the document
    private boolean            mOpen;         // is the dataset open 

    public static NxsDataset instanciate(File destination) throws IOException, NoResultException {
        return new NxsDataset(destination);
    }

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public LogicalGroup getLogicalRoot() {
        if( mRootLogical == null ) {
            boolean debug = Boolean.parseBoolean( mConfig.getParameter(NxsFactory.DEBUG_INF, this) );
            mRootLogical = new NxsLogicalGroup(null, null, this, debug);

        }
        else {
            ExtendedDictionary dict = mRootLogical.getDictionary();
            if ( dict != null && ! dict.getView().equals( Factory.getActiveView() ) ) {
                mRootLogical.setDictionary(mRootLogical.findAndReadDictionary());
            }
        }
        return mRootLogical;
    }

    @Override
    public IGroup getRootGroup() {
        if( mRootPhysical == null && mDatasets.size() > 0 ) {
            NexusGroup[] groups = new NexusGroup[mDatasets.size()];
            int i = 0;
            for( IDataset dataset : mDatasets ) {
                groups[i++] = (NexusGroup) dataset.getRootGroup();
            }
            mRootPhysical = new NxsGroup(groups, null, this);
        }
        return mRootPhysical;
    }


    @Override
    public void saveTo(String location) throws WriterException {
        for( IDataset dataset : mDatasets ) {
            dataset.saveTo(location);
        }
    }

    @Override
    public void save(IContainer container) throws WriterException {
        for( IDataset dataset : mDatasets ) {
            dataset.save(container);
        }
    }

    @Override
    public void save(String parentPath, IAttribute attribute)
            throws WriterException {
        for( IDataset dataset : mDatasets ) {
            dataset.save(parentPath, attribute);
        }
    }

    @Override
    public boolean sync() throws IOException {
        boolean result = true;
        for( IDataset dataset : mDatasets ) {
            if( ! dataset.sync() ) {
                result = false;
            }
        }
        return result;
    }

    @Override
    public void writeNcML(OutputStream os, String uri) throws IOException {
        for( IDataset dataset : mDatasets ) {
            dataset.writeNcML(os, uri);
        }
    }

    @Override
    public void close() throws IOException {
        mOpen = false;
    }

    @Override
    public String getLocation() {
        return mPath.getPath();
    }

    @Override
    public String getTitle() {
        String title = "";
        try {
            title = mDatasets.get(0).getTitle();
        }
        catch( NoSuchElementException e ) {}

        return title;
    }

    @Override
    public void setLocation(String location) {
        File newLoc = new File( location );
        File oldLoc = new File( mPath );
        if( ! oldLoc.equals(newLoc) ) {
            try {
                mPath = new URI(newLoc.getAbsolutePath());
            } catch (URISyntaxException e) {
                e.printStackTrace();
            }
            mDatasets.clear();
        }
    }

    @Override
    public void setTitle(String title) {
        try
        {
            mDatasets.get(0).setTitle(title);
        }
        catch( NoSuchElementException e ) {}
    }

    @Override
    public void open() throws IOException {
    }

    @Override
    public void save() throws WriterException {
        for( IDataset dataset : mDatasets ) {
            dataset.save();
        }
    }

    @Override
    public boolean isOpen() {
        return mOpen;
    }

    public ConfigDataset getConfiguration() {
        if( mConfig == null ) {
            ConfigDataset conf;
            try {
                open();
                conf = ConfigManager.getInstance( NxsFactory.getInstance(), NxsFactory.CONFIG_FILE ).getConfig(this);
                mConfig = conf;
                close();
            } catch (NoResultException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return mConfig;
    }

    // ---------------------------------------------------------
    /// Private methods
    // ---------------------------------------------------------
    private NxsDataset(File destination) {
        mPath = destination.toURI();
        mDatasets = new ArrayList<NexusDataset>();
        if( destination.exists() && destination.isDirectory() ) {
            IDataset datafile;
            NeXusFilter filter = new NeXusFilter();
            for( File file : destination.listFiles(filter) ) {
                datafile = new NexusDatasetImpl(file);
                mDatasets.add( (NexusDatasetImpl) datafile );
            }
        }
        else {
            mDatasets.add( new NexusDatasetImpl(destination ) );
        }
        mOpen = false;
    }

    private void setConfig(ConfigDataset conf) {
        mConfig = conf;
    }
}
