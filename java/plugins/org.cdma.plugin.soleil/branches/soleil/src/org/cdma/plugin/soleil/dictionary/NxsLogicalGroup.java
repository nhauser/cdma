package org.cdma.plugin.soleil.dictionary;

import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.dictionary.ExtendedDictionary;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.FileAccessException;
import org.cdma.interfaces.IDataset;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.navigation.NxsDataset;

public class NxsLogicalGroup extends LogicalGroup {

    public NxsLogicalGroup(IDataset dataset, Key key) {
        super(key, dataset);
    }

    public NxsLogicalGroup(IDataset dataset, Key key, boolean debug) {
        super(key, dataset, debug);
    }

    public NxsLogicalGroup(LogicalGroup parent, Key key, IDataset dataset) {
        super(parent, key, dataset, false);
    }

    public NxsLogicalGroup(LogicalGroup parent, Key key, IDataset dataset, boolean debug) {
        super(parent, key, dataset, debug);
    }

    public ExtendedDictionary findAndReadDictionary() {
        IFactory factory = NxsFactory.getInstance();

        // Detect the key dictionary file and mapping dictionary file
        String keyFile = Factory.getPathKeyDictionary();
        String mapFile = Factory.getPathMappingDictionaryFolder( factory ) + NxsDictionary.detectDictionaryFile( (NxsDataset) getDataset() );
        ExtendedDictionary dictionary = new ExtendedDictionary( factory, keyFile, mapFile);
        try {
            dictionary.readEntries();
        } catch (FileAccessException e) {
            Factory.getLogger().log( Level.SEVERE, e.getMessage());
            dictionary = null;
        }

        return dictionary;
    }
}
