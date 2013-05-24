package org.cdma.plugin.edf.dictionary;

import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.dictionary.ExtendedDictionary;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.FileAccessException;
import org.cdma.interfaces.IDataset;
import org.cdma.plugin.edf.EdfFactory;
import org.cdma.plugin.edf.navigation.EdfDataset;

public class EdfLogicalGroup extends LogicalGroup {

    public EdfLogicalGroup(IDataset dataset, Key key) {
        super(key, dataset);
    }

    public EdfLogicalGroup(IDataset dataset, Key key, boolean debug) {
        super(key, dataset, debug);
    }

    public EdfLogicalGroup(LogicalGroup parent, Key key, IDataset dataset) {
        super(parent, key, dataset, false);
    }

    public EdfLogicalGroup(LogicalGroup parent, Key key, IDataset dataset, boolean debug) {
        super(parent, key, dataset, debug);
    }

    // @Override
    // public ExtendedDictionary findAndReadDictionary() {
    // IFactory factory = EdfFactory.getInstance();
    //
    // // Detect the key dictionary file and mapping dictionary file
    // String keyFile = Factory.getPathKeyDictionary();
    // String mapFile = Factory.getPathMappingDictionaryFolder(factory)
    // + EdfLogicalGroup.detectDictionaryFile((EdfDataset) getDataset());
    // ExtendedDictionary dictionary = new ExtendedDictionary(factory, keyFile, mapFile);
    // try {
    // dictionary.readEntries();
    // }
    // catch (FileAccessException e) {
    // Factory.getLogger().log(Level.SEVERE, e.getMessage(), e);
    // dictionary = null;
    // }
    //
    // return dictionary;
    // }

    /**
     * According to the current corresponding dataset, this method will try to guess which XML
     * dictionary mapping file should be used
     * 
     * @return
     * @throws FileAccessException
     */
    public static String detectDictionaryFile(EdfDataset dataset) {
        return "SoleilEDF_dictionary.xml";
    }
    @Override
    public ExtendedDictionary findAndReadDictionary() {
        IFactory factory = EdfFactory.getInstance();

        // Detect the key dictionary file and mapping dictionary file
        String keyFile = "/home/viguier/CDMADictionaryRoot/views/edf_view.xml"; // Factory.getPathKeyDictionary();
        String mapFile = Factory.getPathMappingDictionaryFolder(factory)
                + EdfLogicalGroup.detectDictionaryFile((EdfDataset) getDataset());
        ExtendedDictionary dictionary = new ExtendedDictionary(factory, keyFile, mapFile);
        try {
            dictionary.readEntries();
        }
        catch (FileAccessException e) {
            Factory.getLogger().log(Level.SEVERE, e.getMessage(), e);
            dictionary = null;
        }

        return dictionary;
    }
}
