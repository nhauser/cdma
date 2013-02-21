package org.cdma.plugin.ansto.internal;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.net.URI;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.netcdf.navigation.NcDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.ansto.AnstoFactory;

public class DetectedSource {
	private static final String EXTENSION[] = new String[] { "nxs", "hdf", "h4", "hdf4", "he4", "h5", "hdf5", "he5" };

    public static final class NetCDFFilter implements FileFilter {
    	private boolean mAllowFolder;
    	
    	public NetCDFFilter(boolean allowFolder) {
    		mAllowFolder = allowFolder;
    	}
    	
    	public NetCDFFilter() {
    		this(false);
    	}
    	
		public boolean accept(File file) {
			boolean result = false;

			if(  file.isDirectory() ) {
				result = mAllowFolder;
			}
			else {
				String name = file.getName();
				for (String ext : EXTENSION) {
					if (name.endsWith(ext)) {
						result = true;
						break;
					}
				}
			}

			return result;
		}
    }

    private boolean mIsExperiment;
    private boolean mIsBrowsable;
    private boolean mIsProducer;
    private boolean mIsReadable;
    private boolean mInitialized;
    private URI mURI;
    private boolean mIsFolder;

    public DetectedSource(URI uri, boolean browsable, boolean readable, boolean producer, boolean experiment, boolean datasetFolder) {
        mIsReadable = readable;
        mIsProducer = producer;
        mIsBrowsable = browsable;
        mIsExperiment = experiment;
        mURI = uri;
    }

    public DetectedSource(URI uri) {
        mURI = uri;
        init(uri);
    }

    public URI getURI() {
        return mURI;
    }

    public boolean isExperiment() {
    	if( ! mInitialized ) {
    		fullInit();
    	}
        return mIsExperiment;
    }

    public boolean isBrowsable() {
    	if( ! mInitialized ) {
    		fullInit();
    	}
        return mIsBrowsable;
    }

    public boolean isProducer() {
    	if( ! mInitialized ) {
    		fullInit();
    	}
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
            if ( file.isDirectory() ) {
                mIsFolder = true;
                mIsReadable = false;
                mIsProducer = false;
                mIsBrowsable = true;
                mIsExperiment = false;
            }
            else {
                mIsFolder = false;

                // Check it is a NeXus file
                mIsReadable = initReadable(uri);
            }
        }

    }
    
    private void fullInit() {
    	synchronized( this ) {
    		if( ! mInitialized ) {
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

        if( file.exists() && !file.isDirectory() && file.length() != 0L ) {
            // Check if the URI is a NetCDF file
            for( String ext : EXTENSION ) {
                if( name.endsWith(ext)) {
                    result = true;
                    break;
                }
            }
        }

        return result;
    }

    private boolean initProducer(URI uri) {
        boolean result = false;
        
        // Check if the URI targets an ANSTO file
        if ( mIsReadable ) {
        	// TODO [SOLEIL][clement] make a real test case
        	{
        		File file = new File(uri.getPath());
//	        	if ( accept(file.getName()) ) {
//	        		return true;
//	        	}
//	        	else {
	        		try {
						NcDataset dataset = new NcDataset(file.getAbsolutePath(), AnstoFactory.NAME);
						IGroup grp = dataset.getRootGroup().getGroup("entry1");
						if( grp != null ) {
							result = grp.getDataItem("program_name") != null;
						}
					} catch (IOException e) {
						Factory.getLogger().log(Level.WARNING, "Unable to test if a producer or not!", e);
					}
//	        	}
        	}
        }
        return result;
    }

    private boolean initExperiment(URI uri) {
        boolean result = false;

        // Check if the URI targets an ANSTO experiment dataset
        if ( mIsProducer ) {
        	// TODO [SOLEIL][clement] make a real test case
        	return true;
        }
        return result;
    }

    private boolean initBrowsable(URI uri) {
        boolean result = false;

        // If experiment not browsable
        if (!mIsExperiment) {
            // If it is a folder or if we are producer of the file
            if (mIsFolder || mIsProducer) {
                result = true;
            }
        }
        return result;
    }
    
    private static boolean accept( String filename ) {
    	boolean result = false;
    	if( filename != null ) {
    		int length = filename.length();
    		for( String extension : EXTENSION ) {
    			if( length >= extension.length() && filename.endsWith(extension) ) {
    				result = true;
    				break;
    			}
    		}
    	}
    	
    	return result;
    }
}
