package org.gumtree.data.soleil.navigation;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;

import org.gumtree.data.Factory;
import org.gumtree.data.dictionary.IExtendedDictionary;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.exception.GDMWriterException;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.soleil.NxsFactory;
import org.gumtree.data.soleil.dictionary.NxsLogicalGroup;

import fr.soleil.nexus4tango.NexusFileWriter;

public abstract class NxsDataset implements IDataset {
	protected IGroup        m_rootPhysical; // Physical root of the document 
	protected ILogicalGroup m_rootLogical;  // Logical root of the document
	
	public static NxsDataset instanciate(String location) throws IOException
	{
		NxsDataset dataset;
		File destination = new File(location);
		if( destination.exists() && destination.isDirectory() ) {
			dataset = new NxsDatasetFolder( destination );
		}
		else {
			dataset = new NxsDatasetFile( destination );
		}
		return dataset;
	}
	
	abstract IGroup getPhyRoot();
	abstract public NxsDataset clone();
	abstract public NexusFileWriter getHandler();

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
            m_rootPhysical = getPhyRoot();
        }
        return m_rootPhysical;
    }
	
	
	@Override
	public void saveTo(String location) throws GDMWriterException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void save(IContainer container) throws GDMWriterException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void save(String parentPath, IAttribute attribute)
			throws GDMWriterException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean sync() throws IOException {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void writeNcML(OutputStream os, String uri) throws IOException {
		// TODO Auto-generated method stub

	}
}
