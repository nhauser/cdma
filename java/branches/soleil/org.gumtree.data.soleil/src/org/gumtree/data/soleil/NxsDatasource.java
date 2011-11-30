package org.gumtree.data.soleil;

import java.io.File;
import java.io.IOException;
import java.net.URI;

import org.gumtree.data.IDatasource;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.soleil.navigation.NxsDataset;
import org.gumtree.data.soleil.navigation.NxsDatasetFolder;

public class NxsDatasource implements IDatasource {
	
	private static final String CREATOR = "Synchrotron SOLEIL";
	private static final String[] BEAMLINES = new String[] {"AILES", "ANTARES", "CASSIOPEE", "CRISTAL", "DIFFABS", "DEIMOS", "DESIRS", "DISCO", "GALAXIES", "LUCIA", "MARS", "METROLOGIE", "NANOSCOPIUM", "ODE", "PLEIADES", "PROXIMA1", "PROXIMA2", "PSICHE", "SAMBA", "SEXTANTS", "SIRIUS", "SIXS", "SMIS", "TEMPO", "SWING"};
	 
	@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}

	@Override
	public boolean isReadable(URI target) {
		File file      = new File(target);
		String name    = file.getName();
		int length     = name.length();
		boolean result = false;
		
		if( file.isDirectory() ) {
			// Check if the folder is an aggregated dataset  
			if( NxsDatasetFolder.isDataset(file) ) {
				result = true;
			}
		}
		// Check if the URI is a NeXus file
		else if( name.substring(length - 4).equals(".nxs") ) {
			result = true;
		}
		return result;
	}

	@Override
	public boolean isProducer(URI target) {
		boolean result = false;

		
		if( isReadable(target) ) {
			File file = new File(target);
			IDataset dataset = null;
			try {
				// instantiate
				dataset = NxsDataset.instanciate( file.getAbsolutePath() );
				
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
							if( node.equals(name) ) {
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
			}
		}
		return result;
	}

	@Override
	public boolean isExperiment(URI target) {
		File file      = new File(target);
		String name    = file.getName();
		int length     = name.length();
		boolean result = false;
		
		if( file.isDirectory() ) {
			// Check if the folder is an aggregated dataset  
			if( NxsDatasetFolder.isDataset(file) ) {
				result = true;
			}
		}
		// Check if the URI is a NeXus file
		else if( name.substring(length - 4).equals(".nxs") ) {
			result = true;
		}
		
		return result;
	}

	@Override
	public boolean isBrowsable(URI target) {
		return isReadable(target);
	}
}
