package org.gumtree.data.soleil.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.gumtree.data.engine.jnexus.navigation.NexusDataItem;
import org.gumtree.data.engine.jnexus.navigation.NexusDimension;
import org.gumtree.data.exception.BackupException;
import org.gumtree.data.exception.InvalidArrayTypeException;
import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IDimension;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.interfaces.IRange;
import org.gumtree.data.soleil.NxsFactory;
import org.gumtree.data.soleil.array.NxsArray;
import org.gumtree.data.soleil.array.NxsArray;
import org.gumtree.data.soleil.array.NxsIndex;
import org.gumtree.data.utils.Utilities.ModelType;
import org.nexusformat.NexusFile;

import fr.soleil.nexus4tango.DataItem;

public class NxsDataItem implements IDataItem {

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
    private NxsDataset           m_dataset;       // CDM IDataset i.e. file handler
    private IGroup               m_parent = null;    // parent group
    private NexusDataItem[]      m_dataItems;     // NeXus dataitem support of the data
    private IArray               m_array = null;     // CDM IArray supporting a view of the data
    private ArrayList<DimOrder>  m_dimension;        // list of dimensions

    
	/// Constructors
	public NxsDataItem(final NxsDataItem dataItem)
	{
        m_dataset   = dataItem.m_dataset;
		m_dataItems = dataItem.m_dataItems.clone();
        m_dimension = new ArrayList<DimOrder> (dataItem.m_dimension);
        m_parent    = dataItem.getParentGroup();
        m_array     = null;
        try {
        	if( m_dataItems.length == 1 ) {
        		m_array = new NxsArray((NxsArray) dataItem.getData());
        	}
        	else {
        		m_array = new NxsArray((NxsArray) dataItem.getData());
        	}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public NxsDataItem(NexusDataItem[] data, NxsDataset handler) {
        m_dataset   = handler;
        m_dataItems = data;
        m_dimension = new ArrayList<DimOrder>();
        m_parent    = getParentGroup();
        m_array     = null;
	}
	
	public NxsDataItem(NexusDataItem[] data, IGroup parent, NxsDataset handler) {
        m_dataset   = handler;
        m_dataItems = data;
        m_dimension = new ArrayList<DimOrder>();
        m_parent    = parent;
        m_array     = null;
	}

	public NxsDataItem(NexusDataItem item, NxsDataset dataset) {
    	this(new NexusDataItem[] {item}, dataset);
    }

	public NxsDataItem(NxsDataItem[] items, NxsDataset dataset) {
        ArrayList<NexusDataItem> list = new ArrayList<NexusDataItem>();
		for( NxsDataItem cur : items ) {
			for( NexusDataItem item : cur.m_dataItems ) {
				list.add(item);
			}
        }
        m_dataItems = list.toArray( new NexusDataItem[list.size()] );
        
		m_dataset   = dataset;
        m_dimension = new ArrayList<DimOrder>();
        m_parent    = getParentGroup();
        m_array     = null;
	}

	/// Methods
	@Override
	public ModelType getModelType() {
		return ModelType.DataItem;
	}
	
	@Override
	public List<IAttribute> getAttributeList()
	{
		//[soleil][clement][12/02/2011] TODO ensure method is correct: shall not we take all attributes of al nodes ?
		List<IAttribute> outList = new ArrayList<IAttribute>();
		outList = m_dataItems[0].getAttributeList();
		return outList;
	}

	@Override
	public IArray getData() throws IOException
	{
        if( m_array == null ) {
        	if( m_dataItems.length > 0 ) {
        		IArray[] arrays = new IArray[m_dataItems.length];
        		for( int i = 0; i < m_dataItems.length; i++ ) {
        			arrays[i] = m_dataItems[i].getData();
        		}
        		m_array = new NxsArray(arrays);
        	}
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
		return new NxsDataItem(this);
	}

	@Override
	public void addOneAttribute(IAttribute att) {
        m_dataItems[0].addOneAttribute(att);
	}

	@Override
	public void addStringAttribute(String name, String value) {
	    m_dataItems[0].addStringAttribute(name, value);
    }

	@Override
	public IAttribute getAttribute(String name) {
		IAttribute result = null;

		for( IDataItem item : m_dataItems ) {
			result = item.getAttribute(name);
			if( result != null ) {
				break;
			}
		}
		
		return result;
	}

	@Override
	public IAttribute findAttributeIgnoreCase(String name) {
		IAttribute result = null;

		for( IDataItem item : m_dataItems ) {
			result = item.findAttributeIgnoreCase(name);
			if( result != null ) {
				break;
			}
		}
		
		return result;
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
		String result = null;

		for( IDataItem item : m_dataItems ) {
			result = item.getDescription();
			if( result != null ) {
				break;
			}
		}
		
		return result;
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
        return m_dataItems[0].getElementSize();
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
		return m_dataItems[0].getName();
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
        	// TODO do not reconstruct the physical hierarchy: keep what has been done in construct
//        	IGroup[] groups = new IGroup[m_dataItems.length];
//        	int i = 0;
//        	for( IDataItem item : m_dataItems ) {
//        		groups[i++] = item.getParentGroup();
//        	}
//        	
//        	m_parent = new NxsGroup(groups, null, m_dataset);
//        	((NxsGroup) m_parent).setChild(this);
        	
        }
		return m_parent;
	}

	@Override
	public List<IRange> getRangeList() {
        List<IRange> list = null;
        try {
            list = new NxsIndex(getData().getShape()).getRangeList();
        } catch( IOException e ) {
            e.printStackTrace();
        }
		return list;
	}
    
    @Override
    public List<IRange> getSectionRanges() {
        List<IRange> list = null;
        try {
            list = ((NxsIndex) getData().getIndex()).getRangeList(); 
        } catch( IOException e ) {
            e.printStackTrace();
        }
        return list;
    }

	@Override
	public int getRank() {
		int[] shape = getShape();
		if( m_dataItems[0].getN4TDataItem().getType() == NexusFile.NX_CHAR )
			return 0;
		else if( shape.length == 1 && shape[0] == 1 )
			return 0;
		else
			return shape.length;
	}

	@Override
	public IDataItem getSection(List<IRange> section)
			throws InvalidRangeException {
        NxsDataItem item = null;
        try {
            item = new NxsDataItem(this);
            m_array = (NxsArray) item.getData().getArrayUtils().sectionNoReduce(section).getArray();
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
			e.printStackTrace();
			return new int[] {-1};
		}
	}

	@Override
	public String getShortName() {
		return m_dataItems[0].getShortName();
	}

	@Override
	public long getSize() {
        int[] shape = getShape();
        long  total = 1;
        for( int size : shape ) {
        	total *= size;
        }
        
		return total;
	}

	@Override
	public int getSizeToCache() {
		// TODO Auto-generated method stub
        new BackupException("Method not supported yet in this plug-in!").printStackTrace();
		return 0;
	}

	@Override
	public IDataItem getSlice(int dim, int value) throws InvalidRangeException {
	    NxsDataItem item = new NxsDataItem(this);
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
		return m_dataItems[0].getType();
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
		return m_dataItems[0].hasCachedData();
	}

	@Override
	public void invalidateCache() {
		m_dataItems[0].invalidateCache();
	}

	@Override
	public boolean isCaching() {
		return m_dataItems[0].isCaching();
	}

	@Override
	public boolean isMemberOfStructure() {
		return m_dataItems[0].isMemberOfStructure();
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
        return m_dataItems[0].isUnsigned(); 
	}

	@Override
	public byte readScalarByte() throws IOException {
		return ((byte[]) m_dataItems[0].getData().getStorage())[0];
	}

	@Override
	public double readScalarDouble() throws IOException {
		return ((double[]) m_dataItems[0].getData().getStorage())[0];
	}

	@Override
	public float readScalarFloat() throws IOException {
		return ((float[]) m_dataItems[0].getData().getStorage())[0];
	}

	@Override
	public int readScalarInt() throws IOException {
		return ((int[]) m_dataItems[0].getData().getStorage())[0];
	}

	@Override
	public long readScalarLong() throws IOException {
		return ((long[]) m_dataItems[0].getData().getStorage())[0];
	}

	@Override
	public short readScalarShort() throws IOException {
		return ((short[]) m_dataItems[0].getData().getStorage())[0];
	}

	@Override
	public String readScalarString() throws IOException {
		return (String) m_dataItems[0].readScalarString();
	}

	@Override
	public boolean removeAttribute(IAttribute attr) {
		boolean result = false;
		for( IDataItem item : m_dataItems ) {
			item.removeAttribute(attr);
		}
		result = true;
		return result;
	}

	@Override
	public void setCachedData(IArray cacheData, boolean isMetadata)
			throws InvalidArrayTypeException {
		for( IDataItem item : m_dataItems ) {
			item.setCachedData(cacheData, isMetadata);
		}
	}

	@Override
	public void setCaching(boolean caching) {
		for( IDataItem item : m_dataItems ) {
			item.setCaching(caching);
		}
	}

	@Override
	public void setDataType(Class<?> dataType) {
		for( IDataItem item : m_dataItems ) {
			item.setDataType(dataType);
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
		for( IDataItem item : m_dataItems ) {
			item.setElementSize(elementSize);
		}
    }

	@Override
	public void setName(String name) {
		for( IDataItem item : m_dataItems ) {
			item.setName(name);
		}
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
		for( IDataItem item : m_dataItems ) {
			item.setSizeToCache(sizeToCache);
		}
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
		for( IDataItem item : m_dataItems ) {
			item.setShortName(name);
		}
	}
    
	@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}
	
	// specific methods
	public DataItem[] getNexusItems() {
		DataItem[] result = new DataItem[ m_dataItems.length ];
		int i = 0;
		for( NexusDataItem item : m_dataItems ) {
			result[i] = item.getN4TDataItem();
			i++;
		}
		return result;
	}
}
