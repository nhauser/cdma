package org.gumtree.data.engine.jnexus.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.gumtree.data.exception.BackupException;
import org.gumtree.data.exception.InvalidArrayTypeException;
import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.exception.NoResultException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IDimension;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.interfaces.IRange;
import org.gumtree.data.engine.jnexus.NexusFactory;
import org.gumtree.data.engine.jnexus.array.NexusArray;
import org.gumtree.data.engine.jnexus.array.NexusIndex;
import org.gumtree.data.utils.Utilities.ModelType;
import org.nexusformat.NexusFile;

import fr.soleil.nexus4tango.DataItem;
import fr.soleil.nexus4tango.DataItem.Data;
import fr.soleil.nexus4tango.PathGroup;

public class NexusDataItem implements IDataItem {

    // Inner class
    // Associate a IDimension to an order of the array 
    private class DimOrder {
        // Members
        public int          m_order;        // order of the corresponding dimension in the NxsDataItem
        public IDimension   m_dimension;    // dimension object

        public DimOrder(int order, IDimension dim) { 
            m_order     = order;
            m_dimension = dim;
        }
    }
    
    
	/// Members
    private NexusDataset         m_cdmDataset;       // CDM IDataset i.e. file handler
    private IGroup               m_parent = null;    // parent group
    private DataItem             m_n4tDataItem;     // NeXus dataitem support of the data
    private IArray               m_array = null;     // CDM IArray supporting a view of the data
    private ArrayList<DimOrder>  m_dimension;        // list of dimensions

    
	/// Constructors
	public NexusDataItem(final NexusDataItem dataItem)
	{
        m_cdmDataset   = dataItem.m_cdmDataset;
		m_n4tDataItem = dataItem.getN4TDataItem();
        m_dimension    = new ArrayList<DimOrder> (dataItem.m_dimension);
        m_parent       = dataItem.getParentGroup();
        m_array        = null;
        try {
    		m_array = new NexusArray((NexusArray) dataItem.getData());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public NexusDataItem(DataItem data, NexusDataset handler) {
        m_cdmDataset  = handler;
        m_n4tDataItem = data;
        m_dimension   = new ArrayList<DimOrder>();
        m_parent      = null;
        m_array       = null;
	}
	
	public NexusDataItem(DataItem data, IGroup parent, NexusDataset handler) {
        m_cdmDataset  = handler;
        m_n4tDataItem = data;
        m_dimension   = new ArrayList<DimOrder>();
        m_parent      = parent;
        m_array       = null;
	}

	/// Methods
	@Override
	public ModelType getModelType() {
		return ModelType.DataItem;
	}
	
	@Override
	public List<IAttribute> getAttributeList()
	{
		HashMap<String, DataItem.Data<?>> inList;
		List<IAttribute> outList = new ArrayList<IAttribute>();
		NexusAttribute tmpAttr;
		String sAttrName;

		inList = m_n4tDataItem.getAttributes();

		Iterator<String> iter = inList.keySet().iterator();
        while( iter.hasNext() )
        {
        	sAttrName = iter.next();
			tmpAttr   = new NexusAttribute(sAttrName, inList.get(sAttrName).getValue());
			outList.add(tmpAttr);
		}

		return outList;
	}

	@Override
	public IArray getData() throws IOException
	{
        if( m_array == null ) {
    		m_array = new NexusArray(m_n4tDataItem);
        }
		return m_array;
	}

	@Override
	public IArray getData(int[] origin, int[] shape) throws IOException, InvalidRangeException
	{
        IArray array = getData().copy(false);
        array.getIndex().setShape(shape);
        array.getIndex().setOrigin(origin);
		return array;
	}

	@Override
	public IDataItem clone()
	{
		return new NexusDataItem(this);
	}

	@Override
	public void addOneAttribute(IAttribute att) {
        m_n4tDataItem.setAttribute(att.getName(), att.getValue().getStorage());
	}

	@Override
	public void addStringAttribute(String name, String value) {
	    m_n4tDataItem.setAttribute(name, value);
    }

	@Override
	public IAttribute getAttribute(String name) {
		HashMap<String, Data<?> > inList;
		String sAttrName;

		inList = m_n4tDataItem.getAttributes();
		Iterator<String> iter = inList.keySet().iterator();
        while( iter.hasNext() )
        {
        	sAttrName = iter.next();
        	if( sAttrName.equals(name) )
        		return new NexusAttribute(sAttrName, inList.get(sAttrName).getValue());
		}

		return null;
	}

	@Override
	public IAttribute findAttributeIgnoreCase(String name) {
		HashMap<String, ?> inList;
		String sAttrName;

		inList = m_n4tDataItem.getAttributes();
		Iterator<String> iter = inList.keySet().iterator();
        while( iter.hasNext() )
        {
        	sAttrName = iter.next();
        	if( sAttrName.toUpperCase().equals(name.toUpperCase()) )
        		return new NexusAttribute(sAttrName, inList.get(sAttrName));
		}

		return null;
	}

	@Override
	public int findDimensionIndex(String name) {
	    for( DimOrder dimord : m_dimension ) {
	        if( dimord.m_dimension.getName().equals(name) )
                return dimord.m_order;
        }
            
        return -1;
	}

	@Override
	public String getDescription() {
		String sDesc = null;

		sDesc = m_n4tDataItem.getAttribute("long_name");
		if( sDesc == null )
			sDesc = m_n4tDataItem.getAttribute("description");
		if( sDesc == null )
			sDesc = m_n4tDataItem.getAttribute("title");
		if( sDesc == null )
			sDesc = m_n4tDataItem.getAttribute("standard_name");
        if( sDesc == null )
            sDesc = m_n4tDataItem.getAttribute("name");

		return sDesc;
	}

	@Override
	public List<IDimension> getDimensions(int i) {
        ArrayList<IDimension> list = new ArrayList<IDimension>();
        
        for( DimOrder dim : m_dimension ) {
            if( dim.m_order == i ) {
                list.add( m_dimension.get(i).m_dimension );
            }
        }
        
        if( list.size() > 0 )    
            return list;
        else
            return null;
	}

	@Override
	public List<IDimension> getDimensionList() {
        ArrayList<IDimension> list = new ArrayList<IDimension>();
        
        for( DimOrder dimOrder : m_dimension ) {
            list.add(dimOrder.m_dimension);
        }
        
		return list;
	}

	@Override
	public String getDimensionsString() {
	    String dimList = "";
        
        int i = 0;
        for( DimOrder dim : m_dimension ) {
            if( i++ != 0 ) {
                dimList += " ";
            }
            dimList += dim.m_dimension.getName();
        }
        
        return dimList;
	}

	@Override
	public int getElementSize() {
        switch( m_n4tDataItem.getType() ) {
            case NexusFile.NX_BINARY:
            case NexusFile.NX_BOOLEAN:
            case NexusFile.NX_CHAR:
                return 1;
            case NexusFile.NX_INT16:
                return 2;
            case NexusFile.NX_FLOAT32:
            case NexusFile.NX_INT32:
                return 4;
            case NexusFile.NX_FLOAT64:
            case NexusFile.NX_INT64:
                return 8;
            default:
                return 1;
        }
	}

	@Override
	public String getName() {
		/*
        String name = m_n4tDataSet.getAttribute("name");
        if( name == null )
            name = m_n4tDataSet.getAttribute("long_name");
        if( name == null )
        	name = m_n4tDataSet.getNodeName();
		return name;
		*/
		return m_n4tDataItem.getPath().toString(false);
	}

	@Override
	public String getNameAndDimensions() {
        StringBuffer buf = new StringBuffer();
        getNameAndDimensions(buf, true, false);
		return buf.toString();
	}

	@Override
	public void getNameAndDimensions(StringBuffer buf, boolean useFullName,
			boolean showDimLength) {
        useFullName = useFullName && !showDimLength;
        String name = useFullName ? getName() : getShortName();
        buf.append(name);

        if (getRank() > 0) buf.append("(");
        for (int i = 0; i < m_dimension.size(); i++) {
          DimOrder dim   = m_dimension.get(i);
          IDimension myd = dim.m_dimension;
          String dimName = myd.getName();
          if ((dimName == null) || !showDimLength)
            dimName = "";

          if (i != 0) buf.append(", ");

          if (myd.isVariableLength()) {
            buf.append("*");
          } else if (myd.isShared()) {
            if (!showDimLength)
              buf.append(dimName + "=" + myd.getLength());
            else
              buf.append(dimName);
          } else {
            if (dimName != null) {
              buf.append(dimName);
            }
            buf.append(myd.getLength());
          }
        }

        if (getRank() > 0) buf.append(")");
	}

	@Override
	public IGroup getParentGroup()
	{
        if( m_parent == null )
        {
            PathGroup path = m_n4tDataItem.getPath().getParentPath();
            try {
				m_parent = (IGroup) m_cdmDataset.getRootGroup().findContainerByPath(path.getValue());
			} catch (NoResultException e) {}
            //m_parent = new NxsGroup(path, m_cdmDataset);
            ((NexusGroup) m_parent).setChild(this);
        }
		return m_parent;
	}

	@Override
	public List<IRange> getRangeList() {
        List<IRange> list = null;
        try {
            list = new NexusIndex(getData().getShape()).getRangeList();
        } catch( IOException e ) {
            e.printStackTrace();
        }
		return list;
	}
    
    @Override
    public List<IRange> getSectionRanges() {
        List<IRange> list = null;
        try {
            list = ((NexusIndex) getData().getIndex()).getRangeList(); 
        } catch( IOException e ) {
            e.printStackTrace();
        }
        return list;
    }

	@Override
	public int getRank() {
		int[] shape = getShape();
		if( m_n4tDataItem.getType() == NexusFile.NX_CHAR )
			return 0;
		else if( shape.length == 1 && shape[0] == 1 )
			return 0;
		else
			return shape.length;
	}

	@Override
	public IDataItem getSection(List<IRange> section)
			throws InvalidRangeException {
        NexusDataItem item = null;
        try {
            item = new NexusDataItem(this);
            m_array = (NexusArray) item.getData().getArrayUtils().sectionNoReduce(section).getArray();
        } catch( IOException e ) {
            e.printStackTrace();
        }
		return item;
	}



	@Override
	public int[] getShape() {
		try {
			return getData().getShape();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return new int[] {-1};
		}
	}

	@Override
	public String getShortName() {
		return m_n4tDataItem.getNodeName();
	}

	@Override
	public long getSize() {
        int[] shape = m_n4tDataItem.getSize();
        int size    = shape[0];
        for( int i = 1; i < shape.length; i++ ) {
            size *= shape[i];
        }
        
		return size;
	}

	@Override
	public int getSizeToCache() {
		// TODO Auto-generated method stub
        new BackupException("Method not supported yet in this plug-in!").printStackTrace();
		return 0;
	}

	@Override
	public IDataItem getSlice(int dim, int value) throws InvalidRangeException {
	    NexusDataItem item = new NexusDataItem(this);
        try {
            item.m_array = item.getData().getArrayUtils().slice(dim, value).getArray();
        }
        catch (Exception e) {
        	item = null;
        }
        return item;
	}

    @Override
    public IDataItem getASlice(int dimension, int value) throws InvalidRangeException {
        return getSlice(dimension, value);
    }
    
	@Override
	public Class<?> getType() {
		return m_n4tDataItem.getDataClass();
	}

    @Override
	public String getUnitsString() {
        IAttribute attr = getAttribute("unit");
        if( attr != null )
            return attr.getStringValue();
        else
            return null;
	}

	@Override
    public boolean hasAttribute(String name, String value) {
    	IAttribute attr;
    	List<IAttribute> listAttr = getAttributeList();

        Iterator<IAttribute> iter = listAttr.iterator();
        while( iter.hasNext() )
        {
        	attr = iter.next();
        	if( attr.getStringValue().equals(value) )
        		return true;
		}
        return false;
    }

	@Override
	public boolean hasCachedData() {
		// TODO Auto-generated method stub
        new BackupException("Method not supported yet in this plug-in!").printStackTrace();
		return false;
	}

	@Override
	public void invalidateCache() {
		// TODO Auto-generated method stub
        new BackupException("Method not supported yet in this plug-in!").printStackTrace();
	}

	@Override
	public boolean isCaching() {
		// TODO Auto-generated method stub
        new BackupException("Method not supported yet in this plug-in!").printStackTrace();
		return false;
	}

	@Override
	public boolean isMemberOfStructure() {
		// TODO Auto-generated method stub
        new BackupException("Method not supported yet in this plug-in!").printStackTrace();
		return false;
	}

	@Override
	public boolean isMetadata() {
		return ( getAttribute("signal") == null );
	}

	@Override
	public boolean isScalar() {
        int rank = 0;
		try {
            rank = getData().getRank();
        } catch(IOException e) {}
        return (rank == 0);
	}

	@Override
	public boolean isUnlimited() {
		return false;
	}

	@Override
	public boolean isUnsigned() {
        int type = m_n4tDataItem.getType(); 
		if(
		     type == NexusFile.NX_UINT16 ||
             type == NexusFile.NX_UINT32 ||
             type == NexusFile.NX_UINT64 ||
             type == NexusFile.NX_UINT8
           ) {
		    return true;
        }
        else {
            return false;
        }
	}

	@Override
	public byte readScalarByte() throws IOException {
		return ((byte[]) m_n4tDataItem.getData())[0];
	}

	@Override
	public double readScalarDouble() throws IOException {
		return ((byte[]) m_n4tDataItem.getData())[0];
	}

	@Override
	public float readScalarFloat() throws IOException {
		return ((float[]) m_n4tDataItem.getData())[0];
	}

	@Override
	public int readScalarInt() throws IOException {
		return ((int[]) m_n4tDataItem.getData())[0];
	}

	@Override
	public long readScalarLong() throws IOException {
		return ((long[]) m_n4tDataItem.getData())[0];
	}

	@Override
	public short readScalarShort() throws IOException {
		return ((short[]) m_n4tDataItem.getData())[0];
	}

	@Override
	public String readScalarString() throws IOException {
		return (String) m_n4tDataItem.getData();
	}

	@Override
	public boolean removeAttribute(IAttribute a) {
        m_n4tDataItem.setAttribute(a.getName(), null);
		return false;
	}

	@Override
	public void setCachedData(IArray cacheData, boolean isMetadata)
			throws InvalidArrayTypeException {
		// TODO Auto-generated method stub
        new BackupException("Method not supported yet in this plug-in!").printStackTrace();

	}

	@Override
	public void setCaching(boolean caching) {
		// TODO Auto-generated method stub
        new BackupException("Method not supported yet in this plug-in!").printStackTrace();
	}

	@Override
	public void setDataType(Class<?> dataType) {
	    try {
            throw new BackupException("Method not support in plug-in: setDataType(Class<?> dataType)!");
        } catch(BackupException e) {
            e.printStackTrace();
        }
	}

	@Override
	public void setDimensions(String dimString) {
        m_parent = getParentGroup();

        List<String> dimNames = java.util.Arrays.asList(dimString.split(" "));
        List<IDataItem> items = m_parent.getDataItemList();
        
        for( IDataItem item : items ) {
            IAttribute attr = item.getAttribute("axis");
            if( attr != null ) {
                if( "*".equals(dimString) ) {
                    setDimension(new NexusDimension(item), attr.getNumericValue().intValue() );
                }
                else if( dimNames.contains(attr.getName()) ) {
                    setDimension(new NexusDimension(item), attr.getNumericValue().intValue() );
                }
            }
        }
	}
    
    @Override
    public void setDimension(IDimension dim, int ind) {
        m_dimension.add( new DimOrder(ind, dim) );
    }
    
	@Override
	public void setElementSize(int elementSize) {
        try {
            throw new BackupException("Method not support in plug-in: setElementSize(int elementSize)!");
        } catch(BackupException e) {
            e.printStackTrace();
        }
    }

	@Override
	public void setName(String name) {
        m_n4tDataItem.setAttribute("name", name);
	}

	@Override
	public void setParent(IGroup group) {
        if( m_parent == null || ! m_parent.equals(group) )
        {
            m_parent = group;
            group.addDataItem(this);
        }
	}

	@Override
	public void setSizeToCache(int sizeToCache) {
		// TODO Auto-generated method stub
        new BackupException("Method not supported yet in this plug-in!").printStackTrace();

	}

    @Override
	public String toStringDebug() {
        String strDebug = "" + getName();
        if( strDebug != null && !strDebug.isEmpty() )
            strDebug += "\n";
        try {
            strDebug += "shape: " + getData().shapeToString() + "\n";
        } catch( IOException e ) {
            e.printStackTrace();
        }
        List<IDimension> dimensions = getDimensionList();
        for( IDimension dim : dimensions ) {
            strDebug += dim.getCoordinateVariable().toString();
        }
        
        List<IAttribute> list = getAttributeList();
        if( list.size() > 0 ) {
        	strDebug += "\nAttributes:\n";
        }
        for( IAttribute a : list ) {
            strDebug += "- " + a.toString() + "\n";
        }
        
		return strDebug;
	}

	@Override
	public String writeCDL(String indent, boolean useFullName, boolean strict) {
		// TODO Auto-generated method stub
        new BackupException("Method not supported yet in this plug-in!").printStackTrace();
		return null;
	}

	@Override
	public void setUnitsString(String units) {
		// TODO Auto-generated method stub
        new BackupException("Method not supported yet in this plug-in!").printStackTrace();
	}

	@Override
	public IDataset getDataset() {
		return m_parent.getDataset();
	}

	@Override
	public String getLocation() {
		return m_parent.getLocation();
	}

	@Override
	public IGroup getRootGroup() {
		return m_parent.getRootGroup();
	}

	@Override
	public void setShortName(String name) {
		m_n4tDataItem.setNodeName(name);
		
	}
    
	@Override
	public String getFactoryName() {
		return NexusFactory.NAME;
	}

    public DataItem getN4TDataItem()
    {
        return m_n4tDataItem;
    }
    // ------------------------------------------------------------------------
    /// Protected methods
    // ------------------------------------------------------------------------
}
