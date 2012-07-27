package org.cdma.plugin.soleil;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.List;

import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDatasource;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.navigation.NxsDataset;
import org.cdma.plugin.soleil.utils.NxsConstant;
import org.nexusformat.NexusException;

import fr.soleil.nexus.DataItem;
import fr.soleil.nexus.NexusFileReader;
import fr.soleil.nexus.NexusNode;
import fr.soleil.nexus.PathGroup;
import fr.soleil.nexus.PathNexus;

public final class NxsDatasource implements IDatasource {
    private static final int    EXTENSION_LENGTH = 4;
    private static final String EXTENSION = ".nxs";
    private static final String CREATOR = "Synchrotron SOLEIL";
    private static final String[] BEAMLINES = new String[] {"CONTACQ", "AILES", "ANTARES", "CASSIOPEE", "CRISTAL", "DIFFABS", "DEIMOS", "DESIRS", "DISCO", "GALAXIES", "LUCIA", "MARS", "METROLOGIE", "NANOSCOPIUM", "ODE", "PLEIADES", "PROXIMA1", "PROXIMA2", "PSICHE", "SAMBA", "SEXTANTS", "SIRIUS", "SIXS", "SMIS", "TEMPO", "SWING"};

    public static final class NeXusFilter implements FilenameFilter {


        public boolean accept(File dir, String name) {
            return (name.endsWith(EXTENSION));
        }
    }

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public boolean isReadable(URI target) {
        File file      = new File(target.getPath());
        String name    = file.getName();
        int length     = name.length();
        boolean result = false;

        if( file.isDirectory() ) {
            // Check if the folder is an aggregated dataset  
            if( isDatasetFolder(file) ) {
                result = true;
            }
        }
        // Check if the URI is a NeXus file
        else if( length > EXTENSION_LENGTH && name.substring(length - EXTENSION_LENGTH).equals(EXTENSION) ) {
            result = true;
        }
        return result;
    }

    @Override
    public boolean isProducer(URI target) {
        boolean result = false;
        if( isReadable(target) ) {
            File file = new File(target.getPath());
            IDataset dataset = null;
            try {
                // instantiate
                dataset = NxsDataset.instanciate( file );

                // open file
                dataset.open();

                // seek at root for 'creator' attribute
                IGroup group = dataset.getRootGroup();
                if( group.hasAttribute("creator", CREATOR) ) {
                    result = true;
                }
                else {
                    group = group.getGroup("<NXentry>");
                    if( group != null ) {
                        group = group.getGroup("<NXinstrument>");
                    }

                    if( group != null ) {
                        String node = group.getShortName();

                        for( String name : BEAMLINES ) {
                            if( node.equalsIgnoreCase(name) ) {
                                result = true;
                                break;
                            }
                        }
                    }
                }

                // close file
                dataset.close();

            } catch (IOException e) {
                // close file
                if( dataset != null ) {
                    try {
                        dataset.close();
                    } catch (IOException e1) {
                    }
                }
            } catch (NoResultException e) {
                e.printStackTrace();
            }
        }
        return result;
    }

    @Override
    public boolean isExperiment(URI target) {
        boolean result = false;

        // Check if the URI is a NeXus file
        if( isProducer(target) ) {
            File file = new File(target.getPath());
            try {
                // Instantiate the dataset corresponding to file and detect its configuration
                NxsDataset dataset = NxsDataset.instanciate(file);
                
                // Interrogate the config to know the experiment path
                String experiment  = dataset.getConfiguration().getParameter(NxsConstant.EXPERIMENT_PATH, dataset);
                String uriFragment = target.getFragment();
                if( uriFragment == null ) {
                    uriFragment = "";
                }
                
                // Decode the fragment part
                uriFragment = URLDecoder.decode(uriFragment, "UTF-8");

                // construct path node to compare them
                NexusNode[] expNodes = PathNexus.splitStringToNode(experiment);
                NexusNode[] fraNodes = PathNexus.splitStringToNode(uriFragment);

                // compare both path
                if (expNodes.length == fraNodes.length) {
                    result = true;
                    // search for not similar nodes in path
                    for (int i = 0; i < expNodes.length; i++) {
                        if (expNodes[i].matchesPartNode(fraNodes[i])) {
                            result = false;
                            break;
                        }
                    }
                }
            } catch (NoResultException e) {
                e.printStackTrace();
            }catch (UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }

        return result;
    }
    
    /*
    @Override
    public boolean isExperiment(URI target) {
        File file      = new File(target.getPath());
        boolean result = false;

        if( file.isDirectory() ) {
            // Check if the folder is an aggregated dataset  
            if( isDatasetFolder(file) ) {
                result = true;
            }
        }
        // Check if the URI is a NeXus file
        else if( isProducer(target) ) {
            try {
                // TODO !!!!!!!!!!!!!!!!
                NxsDataset dataset = NxsDataset.instanciate(file);
                String experiment = dataset.getConfiguration().getParameter(NxsConstant.EXPERIMENT_PATH, dataset);
                String uriSubPath = target.getQuery();
                if( experiment.equals(uriSubPath) ) {
                    result = true;
                }
            } catch (NoResultException e) {
                e.printStackTrace();
            }
            
            result = true;
        }

        return result;
    }
     */

    @Override
    public boolean isBrowsable(URI target) {
        boolean result = false;
        
        // If experiment not browsable
        if( ! isExperiment(target) ) {
            File file = new File(target.getPath());
            
            // If it is a folder containing split NeXus file (quick_exaf)
            if( file.isDirectory() || isProducer(target) ) {
                result = true;
            }
        }
        
        return result;
    }

    @Override
    public List<URI> getValidURI(URI target) {
        List<URI> result = new ArrayList<URI>();
        
        File source = new File( target.getPath() );
        if( source.isDirectory() && ! isDatasetFolder(source)) {
            for( File file : source.listFiles() ) {
                if( file.isDirectory() ) {
                    result.add( file.toURI() );
                }
                else {
                    if( isProducer( file.toURI() ) ) {
                        result.add(file.toURI());
                    }
                }
                
            }
        }
        else {
            if( isReadable(target) && isBrowsable(target) ) {
                try {
                    String uri = target.toString();
                    String sep = target.getFragment() == null? "#" : ""; 
                    
                    NxsDataset dataset = NxsDataset.instanciate( target );
                    IGroup group = dataset.getRootGroup();
                    for( IGroup node : group.getGroupList() ) {
                        result.add( URI.create( uri + sep + URLEncoder.encode( "/" + node.getShortName(), "UTF-8" ) ) );
                    }
                    
                } catch (NoResultException e) {
                    e.printStackTrace();
                } catch (UnsupportedEncodingException e) {
                    e.printStackTrace();
                }
            }
        }
        
        return result;
    }
    
    // ---------------------------------------------------------
    /// private methods
    // ---------------------------------------------------------
    private static boolean isDatasetFolder(File file) {
        boolean result = false;
        if( file.isDirectory() ) {
            NeXusFilter filter = new NeXusFilter();
            for( File nxFile : file.listFiles(filter) ) {
                NexusFileReader reader = new NexusFileReader(nxFile.getAbsolutePath());
                PathNexus path = new PathGroup(new String[] {"<NXentry>", "<NXdata>"} );
                try {
                    reader.openFile();
                    reader.openPath(path);
                    ArrayList<NexusNode> list = reader.listChildren();
                    for( NexusNode node : list ) {
                        reader.openNode(node);
                        DataItem data = reader.readDataInfo();
                        if( data.getAttribute("dataset_part") != null ) {
                            result = true;
                            reader.closeFile();
                            break;
                        }
                        reader.closeData();

                    }
                    reader.closeFile();
                } catch (NexusException e1) {
                    try {
                        reader.closeFile();
                    }
                    catch (NexusException e2) {} 
                    finally {
                        result = false;
                    }
                }
                if( result ) {
                    break;
                }
            }
        }
        return result;
    }
}
