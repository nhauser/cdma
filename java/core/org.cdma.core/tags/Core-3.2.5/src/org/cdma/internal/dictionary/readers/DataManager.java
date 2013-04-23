package org.cdma.internal.dictionary.readers;

import java.lang.ref.SoftReference;
import java.util.HashMap;
import java.util.Map;

public class DataManager {

    private Map<String, SoftReference<DataMapping> >  mDataMapping;   // All available mapping
    private Map<String, SoftReference<DataView> >     mDataView;      // All available keys
    private Map<String, SoftReference<DataConcepts> > mDataConcepts;  // All available concepts
	
    private static DataManager manager; // Singleton pattern
    
    public static DataManager instantiate() {
        if (manager == null) {
            synchronized (DataManager.class) {
                if (manager == null) {
                    manager = new DataManager();
                }
            }
        }
        return manager;
    }    
    
    private DataManager() {
    	mDataConcepts = new HashMap<String, SoftReference<DataConcepts>>();
    	mDataMapping  = new HashMap<String, SoftReference<DataMapping>>();
    	mDataView     = new HashMap<String, SoftReference<DataView>>();
    }
    
    public DataMapping getMapping( String mappingFile ) {
    	DataMapping result = null;
    	synchronized( mDataMapping ) {
    		SoftReference<DataMapping> ref = mDataMapping.get(mappingFile);
    		if( ref != null ) {
    			result = ref.get();
    		}
    	}
    	return result;
    }
    
    public DataView getView( String viewFile ) {
    	DataView result = null;
    	synchronized( mDataView ) {
    		SoftReference<DataView> ref = mDataView.get(viewFile);
    		if( ref != null ) {
    			result = ref.get();
    		}
    	}
    	return result;
    }
    
    public DataConcepts getConcept( String conceptFile ) {
    	DataConcepts result = null;
    	synchronized( mDataConcepts ) {
    		SoftReference<DataConcepts> ref = mDataConcepts.get(conceptFile);
    		if( ref != null ) {
    			result = ref.get();
    		}
    	}
    	return result;
    }
    
    public void registerMapping( String mappingFile, DataMapping mapping ) {
    	synchronized( mDataMapping ) {
    		SoftReference<DataMapping> ref = new SoftReference<DataMapping>(mapping);
    		mDataMapping.put(mappingFile, ref);
    	}
    }
    
    public void registerView( String viewFile, DataView view ) {
    	synchronized( mDataView ) {
    		SoftReference<DataView> ref = new SoftReference<DataView>(view);
    		mDataView.put(viewFile, ref);
    	}
    }
    
    public void registerConcept( String viewFile, DataConcepts concept ) {
    	synchronized( mDataConcepts ) {
    		SoftReference<DataConcepts> ref = new SoftReference<DataConcepts>(concept);
    		mDataConcepts.put(viewFile, ref);
    	}
    }
}
