package org.gumtree.data.soleil.dictionary;

import org.gumtree.data.Factory;
import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.IExtendedDictionary;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.dictionary.impl.ExtendedDictionary;
import org.gumtree.data.dictionary.impl.LogicalGroup;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IKey;
import org.gumtree.data.soleil.NxsFactory;
import org.gumtree.data.soleil.navigation.NxsDataset;
import org.gumtree.data.util.configuration.ConfigDataset;

public class NxsLogicalGroup extends LogicalGroup {
  
  public NxsLogicalGroup(IDataset dataset, IKey key) {
    super(key, dataset);
  }
  
  public NxsLogicalGroup(IDataset dataset, IKey key, boolean debug) {
    super(key, dataset, debug);
    }
    
  public NxsLogicalGroup(ILogicalGroup parent, IKey key, IDataset dataset) {
    super(parent, key, dataset, false);
  }

    public NxsLogicalGroup(ILogicalGroup parent, IKey key, IDataset dataset, boolean debug) {
      super(parent, key, dataset, debug);
    }
    
  public IExtendedDictionary findAndReadDictionary() {
    IFactory factory = NxsFactory.getInstance();
    IExtendedDictionary dictionary;
    // Detect the key dictionary file and mapping dictionary file
    String keyFile = "";
    String mapFile = "";
    try {

    keyFile = Factory.getKeyDictionaryPath();
    mapFile = Factory.getMappingDictionaryFolder( factory ) + detectDictionaryFile();
      dictionary = new ExtendedDictionary( NxsFactory.getInstance(), keyFile, mapFile );
          dictionary.readEntries();
        } catch (FileAccessException e) {
           System.out.println(">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<\n"+keyFile +"\n" + mapFile + ">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<\n");
            e.printStackTrace();
            dictionary = null;
    }
        
      return dictionary;
    }
  
  /**
   * According to the current corresponding dataset, this method will try
   * to guess which XML dictionary mapping file should be used
   * @return
   * @throws FileAccessException 
   */
  protected String detectDictionaryFile() throws FileAccessException {
    // Get the belonging Dataset and its configuration
    NxsDataset dataset = (NxsDataset) getDataset();
    ConfigDataset conf = dataset.getConfiguration();
    
    // Ask for beamline and datamodel parameters
    String beamline = conf.getParameter("BEAMLINE", dataset);
    String model = conf.getParameter("MODEL", dataset);
    
    // Construct the dictionary file name
    if( beamline != null ) {
      beamline = beamline.toLowerCase();
    }
    else {
      beamline = "UNKNOWN";
    }
    if( model != null ) {
      model = model.toLowerCase();
    }
    else {
      model = "UNKNOWN";
    }
    return beamline + "_" + model + ".xml";
  }
}
