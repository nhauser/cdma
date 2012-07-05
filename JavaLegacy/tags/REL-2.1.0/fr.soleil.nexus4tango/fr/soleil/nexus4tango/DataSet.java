package fr.soleil.nexus4tango;

// Nexus Lib
import java.lang.ref.SoftReference;
import java.util.HashMap;
import java.util.Iterator;

import org.nexusformat.NexusException;
import org.nexusformat.NexusFile;

public class DataSet implements Cloneable
{
	public class Data<T>
	{
		private T			m_tValue;
		private Class<T>	m_cType;

		@SuppressWarnings("unchecked")
		protected 		Data(T tValue)	{ m_tValue = tValue; m_cType = (Class<T>) tValue.getClass(); }
		public T 	 	getValue()		{ return m_tValue; }
		public Class<T> getType()		{ return m_cType; }
	}

	/// Members
	private int 						m_iType;		// Data type (see NexusFile class)
	private int[]						m_iDimS;		// Dataset (node) dimension's sizes
	private int[]						m_iDimData;		// Data slab dimension's sizes
	private int[]                       m_iStart;       // Data slab start position
	private SoftReference<Object>		m_oData;		// Data it's an array of values belonging to a dataset
	private String						m_sNodeName;	// DataSet's node name
	private boolean						m_bSingleRaw;	// Is the data stored in memory a single raw

	private HashMap<String, Data<?> >	m_mAttr;		// Map containing all node's attributes having name as key associated to value
	private PathData					m_pnPath;		// Path were the DataSet has been READ (not used when writing)

	// Constructors
	public DataSet() { 
		m_bSingleRaw = true;
		m_mAttr      = new HashMap<String, Data<?>>(); 
		m_pnPath     = null; 
		m_oData      = null; 
		m_iDimData   = null;
		m_iDimS      = null;
		m_iStart     = null;
	}

	public DataSet(Object oArray) throws NexusException	{ 
		m_mAttr = new HashMap<String, Data<?>>(); 
		initFromData(oArray); 
		m_pnPath = null;
	}

	@SuppressWarnings("unchecked")
	public DataSet clone() throws CloneNotSupportedException {
		DataSet result = new DataSet();

		if(m_iStart != null ) {
			result.m_iStart = this.m_iStart.clone();
		}
		result.m_bSingleRaw = this.m_bSingleRaw;
		result.m_iDimData   = this.m_iDimData.clone();
		result.m_iDimS      = this.m_iDimData.clone();
		result.m_iType      = this.m_iType;
		result.m_mAttr      = (HashMap<String, Data<?>>) this.m_mAttr.clone();
		result.m_pnPath     = this.m_pnPath;
		result.m_sNodeName  = this.m_sNodeName;
		result.m_oData      = result.m_oData;
		
		return result;
	}

	// Accessors
	public int 		getType() 					{ return m_iType; }
	public int[]	getSize() 					{ return m_iDimS; }
	public int[]    getSlabSize()				{ return m_iDimData; }
	public int[]    getStart()                  { return m_iStart; }
	public Object	getData()		 			{ if( m_oData == null || m_oData.get() == null ) loadData(); return m_oData.get(); }
	public PathData	getPath()					{ return m_pnPath; }

	public void		setType(int iType)			{ m_iType = iType; }
	public void		setSize(int[] iDimS)		{ m_iDimS = iDimS.clone(); }
	public void		setStart(int[] iStart)		{ m_iStart = iStart.clone(); }
	public void     setSlabSize(int[] iSlab)    { m_iDimData = iSlab.clone(); }
	public void		setData(Object oData)		{ m_oData = new SoftReference<Object>(oData); }
	public void		setPath(PathNexus pnPath)	{ if( pnPath instanceof PathData ) { m_pnPath = (PathData) pnPath; } else { m_pnPath = PathData.Convert(pnPath); } }

	// Attribute accessors
	public String	getUnit() 						{ Object attrValue = getAttribute("units");			return attrValue == null ? null : attrValue.toString(); }
	public String	getDesc() 						{ Object attrValue = getAttribute("description");	return attrValue == null ? null : attrValue.toString(); }
	public String	getTime() 						{ Object attrValue = getAttribute("timestamp");		return attrValue == null ? null : attrValue.toString(); }
	public String	getFormat() 					{ Object attrValue = getAttribute("format");		return attrValue == null ? null : attrValue.toString(); }
	public String	getName()						{ Object attrValue = getAttribute("name");			return attrValue == null ? null : attrValue.toString(); }
	public String	getNodeName()					{ return m_sNodeName; }

	public <T> void	setUnit(T sUnit) 				{ setAttribute("units", sUnit); }
	public <T> void	setDesc(T sDesc) 				{ setAttribute("description", sDesc); }
	public <T> void	setTime(T sTime) 				{ setAttribute("timestamp", sTime); }
	public <T> void	setFormat(T sForm)				{ setAttribute("format", sForm); }
	public <T> void	setName(T sName)				{ setAttribute("name", sName); }
	public void		setNodeName(String sNodeName)	{ m_sNodeName = sNodeName; }

	// Generic attribute accessors
	public HashMap<String, Data<?>> getAttributes() { return m_mAttr; }

	@SuppressWarnings("unchecked")
	public <T> T getAttribute(String sAttrName)
	{
		if( m_mAttr == null || !m_mAttr.containsKey(sAttrName))
		{
			return null;
		}
		Data<T> tVal = (Data<T>) m_mAttr.get(sAttrName);
		return tVal.getType().cast(tVal.getValue());
	}



	public <T> void setAttribute(String sAttrName, T oValue)
	{
		if( m_mAttr == null )
		{
			m_mAttr = new HashMap<String, Data<?>>();
		}

		Data<T> tValue = new Data<T>(oValue);
		m_mAttr.put(sAttrName, tValue);
	}

	@Override
	public void finalize() { m_oData = null; m_iDimS = null; m_iDimData = null; m_sNodeName = null; m_mAttr = null; }

	// ---------------------------------------------------------
	/// Public methods
	/**
	 * InitFromObject
	 * Initialization an empty DataSet with datas and infos guessed from a data object. i.e: type, rank, dimsize...
	 *
	 * @param oData datas permitting detection of type
	 */

	public void initFromData(Object oData) throws NexusException
	{
		Class<?> cDataClass  = oData.getClass();

		// Manage particular case of string
		if(oData instanceof String)
		{
			m_iType = NexusFile.NX_CHAR;
			m_iDimS = new int[] {((String) oData).length()};
			m_oData = new SoftReference<Object>(oData);
			m_iDimData = new int[] {((String) oData).length()};
			m_iStart = new int[] {0};
		}
		else if( oData instanceof PathNexus )
		{
		    m_iType = NexusFile.NX_BINARY;   
			m_iDimS = new int[] {0};
			m_oData = new SoftReference<Object>(oData);
			m_iDimData = new int[] {0};
			m_iStart = new int[] {0};
			return;
		}
		// Try to determine type of data (primitive only) and its dimension size
		else
		{
			//	If data is not array try to arrify it
			if( !oData.getClass().isArray() && !(oData instanceof String) )
			{
				Object oObj = java.lang.reflect.Array.newInstance(oData.getClass(), 1);
				java.lang.reflect.Array.set(oObj, 0, oData);
				oData = oObj;
				cDataClass  = oData.getClass();
			}
			
			// Check if data is array
			if( !cDataClass.isArray() )
			{
				throw new NexusException("Only strings and arrays are allowed to be manipulated!");
			}
			m_oData = new SoftReference<Object>(oData);
			m_iType = NexusFileReader.defineNexusFromType(oData);
			initDimSize();
		}
	}

	/**
	 * toString
	 * Give a string representation of the DataSet with its principal infos
	 */
	@Override
	public String toString()
	{
		StringBuffer str = new StringBuffer();
		str.append("     - Node name: " + m_sNodeName + "   Type: "+ m_iType + "\n     - Node Size: [");
		for(int i=0; i< m_iDimS.length;i++)
		{
			str.append(String.valueOf(m_iDimS[i])+ (( i<m_iDimS.length-1)? ", " : " "));
		}
		str.append("]   Slab Size: [");
		for(int i=0; i< m_iDimData.length;i++)
		{
			str.append(String.valueOf(m_iDimData[i])+ (( i<m_iDimData.length-1)? ", " : " "));
		}   
		str.append("]   Slab Pos: [");
		for(int i=0; i< m_iStart.length;i++)
		{
			str.append(String.valueOf(m_iStart[i])+ (( i<m_iStart.length-1)? ", " : " "));
		}
		str.append("]");

		String sAttrName;
		if( m_mAttr != null )
		{
			for(Iterator<String> iter = m_mAttr.keySet().iterator() ; iter.hasNext() ; )
			{
				sAttrName = (String) iter.next();
				str.append("\n     - " + sAttrName + ": " + getAttribute(sAttrName));
			}
		}
		str.append("\n     - Node Path: " + m_pnPath.toString());
		if( m_iType == NexusFile.NX_CHAR && null != m_oData && null != m_oData.get() ) {
			str.append("\n     - Value: "+ m_oData.get());
		}
		return str.toString();
	}

	public void arrayify()
	{
		Object data = m_oData.get();
		if( data != null && ! data.getClass().isArray() && !(data instanceof String) )
		{
			Object oTmp = java.lang.reflect.Array.newInstance(data.getClass(), 1);
			java.lang.reflect.Array.set(oTmp, 0, data);
			m_oData = new SoftReference<Object>(oTmp);
		}
	}

	// ---------------------------------------------------------
	/// Private methods
	/**
	 * getDataClass
	 * return the primitive type of the owned data according to the NexusFile.Type
	 */
	public Class<?> getDataClass()
	{
		switch( m_iType )
		{
			case NexusFile.NX_UINT32:
			case NexusFile.NX_INT32 :
				return Integer.TYPE;
			case NexusFile.NX_UINT16:
			case NexusFile.NX_INT16 :
				return Short.TYPE;
			case NexusFile.NX_FLOAT32 :
				return Float.TYPE;
			case NexusFile.NX_FLOAT64 :
				return Double.TYPE;
			case NexusFile.NX_UINT64:
			case NexusFile.NX_INT64 :
				return Long.TYPE;
			case NexusFile.NX_CHAR :
				return String.class;
			case NexusFile.NX_BOOLEAN :
				return Boolean.TYPE;
			case NexusFile.NX_BINARY :
			default:
				return Byte.TYPE;

		}
	}

	public boolean isSingleRawArray()
	{
		return m_bSingleRaw;
	}

	protected void isSingleRawArray(boolean bSingleRaw)
	{
		m_bSingleRaw = bSingleRaw;
	}

	/**
	 * InitDimSize
	 * Initialize member dimension sizes 'm_iDimS' according to defined member data 'm_oData'
	 */
	private void initDimSize() throws NexusException
	{
		Object oArray = m_oData.get();
		// Check data existence
		if( oArray == null )
			throw new NexusException("No data to determine Nexus data dimension sizes!");

		// Determine rank of array (by parsing data array class name)
		String sClassName = oArray.getClass().getName();
		int iRank  = 0;
		int iIndex = 0;
		char cChar;
		while (iIndex < sClassName.length()) {
			cChar = sClassName.charAt(iIndex);
			iIndex++;
			if (cChar == '[') {
				iRank++;
			}
		}

		// Set dimension rank
		m_iDimData = new int[iRank];
		m_iDimS    = new int[iRank];
		m_iStart   = new int[iRank];

		// Fill dimension size array
		for ( int i = 0; i < iRank; i++) {
			m_iDimS[i] = java.lang.reflect.Array.getLength(oArray);
			m_iDimData[i] = m_iDimS[i];
			m_iStart[i] = 0;
			oArray = java.lang.reflect.Array.get(oArray,0);
		}
	}

	private void loadData()
	{
		NexusFileReader nfrFile = new NexusFileReader(m_pnPath.getFilePath());
		nfrFile.isSingleRawResult(m_bSingleRaw);
		try
		{
			nfrFile.openFile(NexusFile.NXACC_READ);
			if( m_iStart == null) {
				m_oData = new SoftReference<Object>(nfrFile.readData(m_pnPath).getData());
			} else {
				m_oData = new SoftReference<Object>(nfrFile.readDataSlab(m_pnPath, m_iStart, m_iDimData).getData());
			}

			nfrFile.closeFile();
		}
		catch(NexusException ne)
		{
			try
			{
				nfrFile.closeFile();
			} catch(NexusException n) {}
		}
	}
	
	public Object getData(int[] pos, int[] shape) {
		// save current position and shape
		boolean newSlab = !(java.util.Arrays.equals(shape, m_iDimData) && java.util.Arrays.equals(m_iStart, pos));

		m_iStart = pos;
		m_iDimData = shape;
		
		if( m_oData == null || m_oData.get() == null || newSlab ) {
			loadData();
		}
		
		// restore previous position and shape
		return m_oData.get();
		
	}
}
