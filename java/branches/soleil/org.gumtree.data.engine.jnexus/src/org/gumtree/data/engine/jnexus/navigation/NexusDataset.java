package org.gumtree.data.engine.jnexus.navigation;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.engine.jnexus.NexusFactory;
import org.gumtree.data.exception.GDMWriterException;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IGroup;
import org.nexusformat.NexusException;
import org.nexusformat.NexusFile;

import fr.soleil.nexus4tango.DataItem;
import fr.soleil.nexus4tango.NexusFileWriter;
import fr.soleil.nexus4tango.NexusNode;
import fr.soleil.nexus4tango.PathData;
import fr.soleil.nexus4tango.PathGroup;
import fr.soleil.nexus4tango.PathNexus;

public class NexusDataset implements IDataset {
	protected IGroup        m_rootPhysical; // Physical root of the document 
	private NexusFileWriter m_n4tWriter; 	// Instance manipulating the NeXus file
	private String			m_sTitle;
	protected PathNexus		m_n4tCurPath;	// Instance of the current path

	public NexusDataset( File nexusFile )
	{
		m_n4tWriter    = null;
		m_rootPhysical = null;
		m_n4tWriter    = new NexusFileWriter(nexusFile.getAbsolutePath());
		m_n4tCurPath   = PathNexus.ROOT_PATH.clone();
		m_sTitle	   = nexusFile.getName();
		m_n4tCurPath.setFile(nexusFile.getAbsolutePath());
		m_n4tWriter.setBufferSize(300);
		m_n4tWriter.isSingleRawResult(true);
		m_n4tWriter.setCompressedData(true);
	}

	public NexusDataset( NexusDataset dataset )
	{
		m_rootPhysical = dataset.m_rootPhysical;
		m_n4tWriter    = dataset.m_n4tWriter;
		m_n4tCurPath   = dataset.m_n4tCurPath.clone();
		m_sTitle	   = dataset.m_sTitle;
		m_n4tWriter.isSingleRawResult(true);
	}

	@Override
	public NexusDataset clone()
	{
		return new NexusDataset(this);
	}

	@Override
	public void open() throws IOException
	{
		try
		{
			m_n4tWriter.closeFile();
			m_n4tWriter.openFile(m_n4tCurPath.getFilePath(), NexusFile.NXACC_READ);
			m_n4tWriter.openPath(m_n4tCurPath);
		}
		catch(NexusException ne)
		{
			throw new IOException(ne);
		}
	}

	/// Methods
	@Override
	public void close() throws IOException
	{
		try
		{
			m_n4tWriter.closeFile();
		}
		catch(NexusException ne)
		{
			throw new IOException(ne);
		}
	}

	
	@Override
	public String getLocation() {
		if( m_n4tCurPath != null )
		{
			return m_n4tCurPath.getFilePath();
		}
		return null;
	}
	
	@Override
	public String getTitle() {
		return m_sTitle;
	}

	@Override
	public void save() throws GDMWriterException {
        List<IDataItem> items = new ArrayList<IDataItem>(); 
        NexusGroup.getDescendentDataItem(items, m_rootPhysical);
        try {
            // Open the destination file
            m_n4tWriter.openFile(m_n4tWriter.getFilePath(), NexusFile.NXACC_RDWR);
            
            // Save each IDataItem
            DataItem data;
            for( IDataItem item : items ) {
            	data = ((NexusDataItem) item).getN4TDataItem();
            	m_n4tWriter.writeData(data, data.getPath());
            	
            }
            
            // Close the file
            m_n4tWriter.closeFile();
        } catch(NexusException e) {
            throw new GDMWriterException(e.getMessage(), e);
        }
	}

	@Override
	public void setLocation(String location)
	{
		String sCurFile = "";
		PathNexus path = new PathGroup(location.split("/"));

		if( ! m_rootPhysical.equals(PathNexus.ROOT_PATH) ) {
			sCurFile = m_rootPhysical.getLocation();
		}

		try
		{
			m_n4tWriter.openPath(path);
		}
		catch(NexusException e1)
		{
			NexusNode topNode = path.popNode();
			path = new PathData((PathGroup) path, topNode.getNodeName());
			try
			{
				m_n4tWriter.openPath(path);
			} catch(NexusException e2) {}
		}
		m_n4tCurPath = path;
		m_n4tCurPath.setFile(sCurFile);
	}

	@Override
	public void setTitle(String title) {
		m_sTitle = title;
	}

	@Override
	public boolean isOpen() {
		return m_n4tWriter.isFileOpened();
	}
    
	public NexusFileWriter getHandler()
    {
        return m_n4tWriter;
    }

	// -----------------------------------------------------------
	/// protected methods
	// -----------------------------------------------------------
	// Accessors
	protected PathNexus getCurrentPath()
	{
		return m_n4tCurPath;
	}

	protected PathNexus getRootPath()
	{
		return ((NexusGroup) m_rootPhysical).getPathNexus();
	}

	// Methods
	protected void setLocation(PathNexus location)
	{
		try
		{
			m_n4tWriter.openPath(location);
		}
		catch(NexusException e) {}
		m_n4tCurPath = location.clone();
	}

	protected void setRootGroup(PathNexus rootPath)
	{
		m_rootPhysical = new NexusGroup(rootPath, this);
	}

	@Override
	public String getFactoryName() {
		return NexusFactory.NAME;
	}

	@Override
	public IGroup getRootGroup() {
		if( m_rootPhysical == null ) {
            m_rootPhysical = new NexusGroup(null, PathNexus.ROOT_PATH.clone(), this);
        }
        return m_rootPhysical;
	}

	@Override
	public ILogicalGroup getLogicalRoot() {
		return null;
	}

	@Override
	public boolean sync() throws IOException {
		// TODO Auto-generated method stub
		return false;
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
	public void writeNcML(OutputStream os, String uri) throws IOException {
		// TODO Auto-generated method stub
		
	}

}
