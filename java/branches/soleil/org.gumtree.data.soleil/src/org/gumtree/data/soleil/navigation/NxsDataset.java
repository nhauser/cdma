package org.gumtree.data.soleil.navigation;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.util.NoSuchElementException;
import java.util.Vector;

import org.gumtree.data.Factory;
import org.gumtree.data.dictionary.IExtendedDictionary;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.engine.jnexus.NexusDatasource.NeXusFilter;
import org.gumtree.data.engine.jnexus.NexusFactory;
import org.gumtree.data.engine.jnexus.navigation.NexusDataset;
import org.gumtree.data.engine.jnexus.navigation.NexusGroup;
import org.gumtree.data.exception.GDMWriterException;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.soleil.NxsFactory;
import org.gumtree.data.soleil.dictionary.NxsLogicalGroup;

import fr.soleil.nexus4tango.NexusFileWriter;

public class NxsDataset implements IDataset {
	private Vector<NexusDataset> m_datasets;     // all found datasets in the folder
	private String               m_path;         // folder containing all datasets
	private IGroup               m_rootPhysical; // Physical root of the document 
	private ILogicalGroup        m_rootLogical;  // Logical root of the document
	private boolean              m_open;         // is the dataset open 
	
	public static NxsDataset instanciate(String location) throws IOException
	{
		File destination = new File(location);
		return new NxsDataset(destination);
	}
	
	public NxsDataset(File destination) {
		m_path = destination.getAbsolutePath();
		m_datasets = new Vector<NexusDataset>();
		if( destination.exists() && destination.isDirectory() ) {
			IDataset datafile;
			NeXusFilter filter = new NeXusFilter();
			for( File file : destination.listFiles(filter) ) {
				try {
					datafile = NexusFactory.getInstance().createDatasetInstance( file.toURI() );
					m_datasets.add( (NexusDataset) datafile );
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
		else {
			m_datasets.add( new NexusDataset( destination ) );
		}
		m_open = false;
	}
	
		@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}

	@Override
	public ILogicalGroup getLogicalRoot() {
		if( m_rootLogical == null ) {
			boolean debug = false;
			if( null != System.getProperty(NxsFactory.DEBUG_INF, System.getenv(NxsFactory.DEBUG_INF)) )
			{
				debug = true;
			}
			m_rootLogical = new NxsLogicalGroup(null, null, this, debug);
			
        }
		else {
			IExtendedDictionary dict = m_rootLogical.getDictionary();
			if ( dict != null && ! dict.getView().equals( Factory.getActiveView() ) ) {
				m_rootLogical.setDictionary(m_rootLogical.findAndReadDictionary());
			}
		}
        return m_rootLogical;
	}
	
	@Override
    public IGroup getRootGroup() {
		if( m_rootPhysical == null ) {
			if( m_datasets.size() > 0 ) {
				NexusGroup[] groups = new NexusGroup[m_datasets.size()];
				int i = 0;
				for( IDataset dataset : m_datasets ) {
					groups[i++] = (NexusGroup) dataset.getRootGroup();
				}
				m_rootPhysical = new NxsGroup(groups, null, this);
			}
        }
        return m_rootPhysical;
    }
	
	
	@Override
	public void saveTo(String location) throws GDMWriterException {
		for( IDataset dataset : m_datasets ) {
			dataset.saveTo(location);
		}
	}

	@Override
	public void save(IContainer container) throws GDMWriterException {
		for( IDataset dataset : m_datasets ) {
			dataset.save(container);
		}
	}

	@Override
	public void save(String parentPath, IAttribute attribute)
			throws GDMWriterException {
		for( IDataset dataset : m_datasets ) {
			dataset.save(parentPath, attribute);
		}
	}

	@Override
	public boolean sync() throws IOException {
		boolean result = true;
		for( IDataset dataset : m_datasets ) {
			if( ! dataset.sync() ) {
				result = false;
			}
		}
		return result;
	}

	@Override
	public void writeNcML(OutputStream os, String uri) throws IOException {
		for( IDataset dataset : m_datasets ) {
			dataset.writeNcML(os, uri);
		}
	}

	@Override
	public void close() throws IOException {
		for( IDataset dataset : m_datasets ) {
			dataset.close();
		}
		m_open = false;
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
	public void open() throws IOException {
		for( IDataset dataset : m_datasets ) {
			dataset.open();
		}
		m_open = true;
	}

	@Override
	public void save() throws GDMWriterException {
		for( IDataset dataset : m_datasets ) {
			dataset.save();
		}
	}

	@Override
	public boolean isOpen() {
		return m_open;
	}
	
	public NexusFileWriter getHandler()
    {
       return m_datasets.firstElement().getHandler();
    }
}
