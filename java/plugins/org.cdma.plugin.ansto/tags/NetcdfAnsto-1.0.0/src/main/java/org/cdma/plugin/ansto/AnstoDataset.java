package org.cdma.plugin.ansto;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.netcdf.navigation.NcDataset;

public class AnstoDataset extends NcDataset {

    public static AnstoDataset instantiate(URI uri) {
        AnstoDataset result = null;
        if( uri != null ) {
            String path = uri.getPath();
            File file = new File(path);
            if( file.exists() && ! file.isDirectory() ) {
                try {
                    result = new AnstoDataset(path);
                }
                catch( IOException e ) {
                    Logger.getLogger(Logger.GLOBAL_LOGGER_NAME).log(Level.SEVERE, "Unable to instantiate dataset:\n" + e.getMessage());
                }
            }
        }
        
        return result;
    }
    
    public AnstoDataset(String path) throws IOException {
        super(path, AnstoFactory.NAME);
    }

    @Override
    public LogicalGroup getLogicalRoot() {
        return new LogicalGroup( null, this );
    }

}
