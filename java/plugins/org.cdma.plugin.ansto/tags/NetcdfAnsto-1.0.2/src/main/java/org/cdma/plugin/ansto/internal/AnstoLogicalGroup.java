package org.cdma.plugin.ansto.internal;

import org.cdma.dictionary.ExtendedDictionary;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IKey;

public class AnstoLogicalGroup extends LogicalGroup {
    
    public AnstoLogicalGroup(IKey key, IDataset dataset) {
        super( key, dataset );
    }
    
    // TODO [SOLEIL][clement] this method returns an ExtendedDictionary class. The main goal is to determine 
    // which mapping file should be used for the corresponding dataset. If using the super implementation
    // the mapping will be the one defined by: dictionaries_folder/mappings/plug-in_ID/plug-in_ID_dictionary.xml
    @Override
    public ExtendedDictionary findAndReadDictionary() {
        return super.findAndReadDictionary();
    }
}
