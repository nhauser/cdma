package org.gumtree.data.soleil.navigation;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.Vector;

import org.gumtree.data.exception.GDMWriterException;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.soleil.NxsFactory;
import org.gumtree.data.soleil.dictionary.NxsLogicalGroup;
import org.nexusformat.NexusException;

import fr.soleil.nexus4tango.DataItem;
import fr.soleil.nexus4tango.NexusFileReader;
import fr.soleil.nexus4tango.NexusFileWriter;
import fr.soleil.nexus4tango.NexusNode;
import fr.soleil.nexus4tango.PathGroup;
import fr.soleil.nexus4tango.PathNexus;

public class NxsDatasetFolder extends  NxsDataset {
	private String                 m_path;         // folder containing all datasets
	private Vector<NxsDatasetFile> m_datasets;     // all found datasets in the folder
	private boolean                m_open;         // is the dataset open

	static class NeXusFilter implements FilenameFilter {
	    public boolean accept(File dir, String name) {
	        return (name.endsWith(".nxs"));
	    }
	}

	public NxsDatasetFolder(File destination) {
		m_path = destination.getAbsolutePath();
		m_open = false;
		m_datasets = new Vector<NxsDatasetFile>();
		NxsDatasetFile datafile;
		NeXusFilter filter = new NeXusFilter();
		for( File file : destination.listFiles(filter) ) {
			try {
				datafile = (NxsDatasetFile) NxsFactory.getInstance().createDatasetInstance( file.toURI() );
				m_datasets.add( datafile );
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	@SuppressWarnings("unchecked")
	public NxsDatasetFolder( NxsDatasetFolder original ) {
		m_path = original.m_path;
		m_datasets = (Vector<NxsDatasetFile>) original.m_datasets.clone();
		m_open = original.m_open;
		try {
			m_rootLogical = (NxsLogicalGroup) original.m_rootLogical.clone();
		} catch (CloneNotSupportedException e) { }
		m_rootPhysical = (NxsGroup) original.m_rootPhysical.clone();
	}

	@Override
	public String getLocation() {
		return m_path;
	}

	@Override
	public String getTitle() {
		String title = "";
		try
		{
			title = m_datasets.firstElement().getTitle();
		}
		catch( NoSuchElementException e ) {}
			
		return title;
	}

	@Override
	public void setLocation(String location) {
		File newLoc = new File( location );
		File oldLoc = new File( m_path );
		if( ! oldLoc.equals(newLoc) ) {
			m_path = newLoc.getAbsolutePath();
			m_datasets.clear();
		}
	}

	@Override
	public void setTitle(String title) {
		try
		{
			m_datasets.firstElement().setTitle(title);
		}
		catch( NoSuchElementException e ) {}
	}

	@Override
	public boolean sync() throws IOException {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void open() throws IOException {
		for( IDataset dataset : m_datasets ) {
			dataset.open();
		}
		m_open = true;
	}

	@Override
	public void close() throws IOException {
		for( IDataset dataset : m_datasets ) {
			dataset.close();
		}
		m_open = false;
	}

	@Override
	public boolean isOpen() {
		return m_open;
	}

	@Override
	IGroup getPhyRoot() {
		NxsGroup[] groups = new NxsGroup[m_datasets.size()];
		int i = 0;
		for( IDataset dataset : m_datasets ) {
			groups[i++] = (NxsGroup) dataset.getRootGroup();
		}
		return new NxsGroupFolder(groups, null, this);
	}

	@Override
	public NxsDatasetFolder clone()
	{
		return new NxsDatasetFolder(this);
	}

	@Override
	public void save() throws GDMWriterException {
		for( IDataset dataset : m_datasets ) {
			dataset.save();
		}		
	}
	
	public NexusFileWriter getHandler()
    {
        return m_datasets.firstElement().getHandler();
    }

	public static boolean isDataset(File file) {
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
					result = false;
				}
				if( result ) {
					break;
				}
			}
		}
		return result;
	}

}
