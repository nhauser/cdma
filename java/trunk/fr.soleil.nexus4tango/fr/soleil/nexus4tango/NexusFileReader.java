package fr.soleil.nexus4tango;

// Tools lib
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Stack;

import org.nexusformat.AttributeEntry;
import org.nexusformat.NexusException;
import org.nexusformat.NexusFile;

public class NexusFileReader extends NexusFileBrowser {
	// / Members
	private boolean m_bResultAsSingleRaw = true;

	// / Constructors
	public NexusFileReader() {
		super();
	}

	public NexusFileReader(String sFilePath) {
		super(sFilePath);
	}

	public NexusFileReader(String sFilePath, String sWorkDir) {
		super(sFilePath, sWorkDir);
	}

	// ---------------------------------------------------------
	// / Browse node to read value
	// ---------------------------------------------------------

	public void isSingleRawResult(boolean bSingleRawResult) {
		m_bResultAsSingleRaw = bSingleRawResult;
	}

	public boolean isSingleRawResult() {
		return m_bResultAsSingleRaw;
	}

	/**
	 * readData Read datas to the given path
	 * 
	 * @param oData
	 *            datas to be stored
	 * @param sPath
	 *            path to set datas in current file
	 */
	public DataSet readData(PathData pdPath) throws NexusException {
		DataSet oOutput;

		// Open target dataset
		openPath(pdPath);

		// Read data from dataset
		oOutput = getDataSet();

		// Return to document root
		closeAll();

		return oOutput;
	}
	
	public DataSet readDataSlab(PathData pPath, int[] iStart, int[] iShape) throws NexusException {
		DataSet oOutput;
		
		// Open target dataset
		openPath(pPath);

		// Read data from dataset
		oOutput = getDataSet(iStart, iShape);

		// Return to document root
		closeAll();

		return oOutput;
	}

	/**
	 * readAttr Read dataset's named attribute to the given path
	 * 
	 * @param sName
	 *            attribut's name of the dataset to be read
	 * @param sPath
	 *            path to dataset in current file
	 */
	public Object readAttr(String sName, PathNexus pnPath)
			throws NexusException {
		Object oOutput = null;

		// Open target dataset
		if (pnPath != null) {
			openPath(pnPath);
		}

		// Read data from dataset
		oOutput = getAttributeValue(sName);

		// Get back to document root
		if (pnPath != null) {
			closeAll();
		}

		// Return the result
		return oOutput;
	}

	// ---------------------------------------------------------
	// / Node reading value
	// ---------------------------------------------------------
	/**
	 * readDataInfo Return the DataSet fitting the opened dataset, without main
	 * data. The DataSet is initialized with dimsize, type... but the dataset
	 * isn't read.
	 * 
	 */
	public DataSet readDataInfo() throws NexusException {
		NexusNode nnNode = getCurrentRealPath().getCurrentNode();
		if (!nnNode.isGroup() && nnNode.getClassName().equals("NXtechnical_data")) {
			return getDataSet();
		}
		
		// Get infos on dataset (data type, rank, dimsize)
		int[] iDataInf = new int[2]; // iDataInf[0] = dataset rank ; iDataInf[1]= data type
		int[] iNodeSize = new int[RANK_MAX]; // whole dataset dimension's sizes
		int[] iDimSize; // signal dimension's sizes

		getNexusFile().getinfo(iNodeSize, iDataInf);

		// Initialize dimension's sizes
		iDimSize = new int[iDataInf[0]];
		System.arraycopy(iNodeSize, 0, iDimSize, 0, iDataInf[0]);

		// Check if dataset is linked to an external dataset
		DataSet dsData = new DataSet();
		dsData.setType(iDataInf[1]);
		dsData.setSize(iDimSize);
		dsData.setSlabSize(iDimSize);
		dsData.setNodeName(getCurrentRealPath().getDataSetName());
		dsData.setPath(getCurrentRealPath().clone());
		dsData.isSingleRawArray(m_bResultAsSingleRaw);

		// Initialize DataSet's attributes
		getDataSetAttribute(dsData);

		return dsData;
	}

	private Object readNodeValue(int[] iDataInf, int[] iDimSize, int iRank)
			throws NexusException {
		
		// We want a iNbDim dimension array of data, so dimensions are the
		// iNbDim last ones of the opened dataset
		int[] length = { 1 };
		int[] iStart;

		if (iRank < 0)
			iRank = iDataInf[0];
		if (iRank > iDataInf[0])
			throw new NexusException(
					"Requested dataset rank is too high: requested rank "
							+ iRank + " available of the dataset is "
							+ iDataInf[0]);

		iStart = new int[iRank];
		for (int iDim = (iDataInf[0] - iRank); iDim < iDataInf[0]; iDim++) {
			iStart[(iDim + iRank) - iDataInf[0]] = 0;
			length[0] *= iDimSize[iDim];
		}

		// Prepare an array data
		Object oOutput;
		Object oInput = defineArrayObject(iDataInf[1], length);

		// Set data into temporary array having a single raw shape
		getNexusFile().getdata(oInput);

		if (m_bResultAsSingleRaw) {
			oOutput = oInput;
		} else {
			// Changing the array into matrix (having iDimSize dimensions'
			// sizes) instead of single row
			oOutput = defineArrayObject(iDataInf[1], iDimSize);
			reshapeArray(iStart, iDimSize, oInput, oOutput);
		}

		// Converting byte[] to string in case of NX_CHAR data
		if (iDataInf[1] == NexusFile.NX_CHAR) {
		   oOutput = new String((byte[]) oOutput);
		}

		// Setting dataset's data
		if (iDataInf[1] == NexusFile.NX_BOOLEAN) {
			oOutput = convertArray(new DataSet(oOutput));
		}

		return oOutput;
	}

	/**
	 * readDataSet Return all datas and attributes of the currently opened
	 * dataset
	 * 
	 * @param iDataRank
	 *            rank of the required data (1 spectrum, 2 images...) (optional)
	 * @note if no rank is given, value of the dataset will be return
	 *       integrally: all slabs will be taken as one entire data
	 */
	private DataSet readDataSet(int iDataRank) throws NexusException {
		DataSet dsData = readDataInfo();
		
		dsData.setData(
						readNodeValue(
							new int[] { dsData.getSize().length, dsData.getType() }, 
							dsData.getSize(), 
							iDataRank
							)
					  );

		dsData.isSingleRawArray(m_bResultAsSingleRaw);

		return dsData;
	}

	/**
	 * reshapeArray Call the readSlabDataRec to reshape the oInput
	 * mono-dimensional array into a multi-dimensional array.
	 * 
	 * @param iStart
	 *            starting position of the slab to project into oOutput
	 * @param iDimSize
	 *            dimensions' size of the dataset
	 * @param oInput
	 *            input array (as a single row) containing data to reshape
	 * @param oOutput
	 *            output array having the proper shape for resize
	 * @throws NexusException
	 */
	protected void reshapeArray(int[] iStart, int[] iDimSize, Object oInput,
			Object oOutput) throws NexusException {
		reshapeArray(0, iStart, iDimSize, oInput, oOutput);
	}

	/**
	 * reshapeArray recursive method reshaping a filled mono-dimensional
	 * array into a multi-dimensional array.
	 * 
	 * @param iCurDim
	 *            reshape all dimensions greater than this one
	 * @param iStart
	 *            starting position of the slab to project into oOutput
	 * @param iDimSize
	 *            dimensions' size of the dataset
	 * @param oInput
	 *            input array (as a single row) containing data to reshape
	 * @param oOutput
	 *            output array having the proper shape for resize
	 * @throws NexusException
	 */
	private void reshapeArray(int iCurDim, int[] iStart, int[] iDimSize,
			Object oInput, Object oOutput) throws NexusException {
		int lStartRaw;
		int lLinearStart;
		if (iCurDim != iDimSize.length - 1) {
			for (int i = 0; i < iDimSize[iCurDim]; i++) {
				Object o = java.lang.reflect.Array.get(oOutput, i);
				reshapeArray(iCurDim + 1, iStart, iDimSize, oInput, o);
				
				if (iStart[iCurDim] < iDimSize[iCurDim] - 1)
					iStart[iCurDim] = i + 1;
				else
					iStart[iCurDim] = 0;
			}
		}

		if (iCurDim == iDimSize.length - 1) {
			lLinearStart = 0;
			for (int k = 1; k < iDimSize.length; k++) {
				lStartRaw = 1;
				for (int j = k; j < iDimSize.length; j++) {
					lStartRaw *= iDimSize[j];
				}
				lStartRaw *= iStart[k - 1];
				lLinearStart += lStartRaw;
			}
			System.arraycopy(oInput, lLinearStart, oOutput, 0, iDimSize[iDimSize.length - 1]);
		}
	}

	protected DataSet getDataSet() throws NexusException {
		return getDataSet(-1);
	}
	
	protected DataSet getDataSet(int iRank) throws NexusException {
		DataSet dataset;
		NexusNode nnNode = getCurrentRealPath().getCurrentNode();

		String sNodeName = nnNode.getNodeName();
		String sNodeClass = nnNode.getClassName();

		// If encountered a dataset get its datas
		if (!nnNode.isGroup() && !sNodeClass.equals("NXtechnical_data")) {
			dataset = readDataSet(iRank);
			dataset.setNodeName(sNodeName);

			return dataset;
		}
		// else if we encountered a NXtechnical_data group: we get its "data"
		// and "description"
		else if (!nnNode.isGroup() && sNodeClass.equals("NXtechnical_data")) {
			// Get the "data" node
			openData("data");
			dataset = readDataSet(iRank);
			dataset.setNodeName(sNodeName);
			closeData();
			// Try to get a description for the technical data
			try {
				// Set to "data" as description attribute the "description" node
				// beside
				openData("description");
				dataset.setDesc(readDataSet(iRank).getData().toString());
				closeData();
			} catch (NexusException ne) {
			}
			dataset.setPath(getCurrentRealPath().clone());

			return dataset;
		} else
			return null;
	}
	
	protected DataSet getDataSet(int[] iStartPos, int[] iShape) throws NexusException {
		DataSet dsData = readDataInfo();
		dsData.setStart(iStartPos);
		dsData.setSlabSize(iShape);
		
		int[] iDataInf = new int[] { dsData.getSize().length, dsData.getType() };
		
		// We want a iNbDim dimension array of data, so dimensions are the
		// iNbDim last ones of the opened dataset
		int[] length = { 1 };

		for (int iDim : iShape) {
			length[0] *= iDim;
		}

		// Prepare an array data
		Object oOutput;
		Object oInput = defineArrayObject(iDataInf[1], length);

		// Set data into temporary array having a single raw shape
		getNexusFile().getslab(iStartPos, iShape, oInput);
		
		if (m_bResultAsSingleRaw) {
			oOutput = oInput;
		} else {
			// Changing the array into matrix (having iDimSize dimensions'
			// sizes) instead of single row
			oOutput = defineArrayObject(iDataInf[1], iShape);
		}

		// Converting byte[] to string in case of NX_CHAR data
		if (iDataInf[1] == NexusFile.NX_CHAR) {
		   oOutput = new String((byte[]) oOutput);
		}

		// Setting dataset's data
		if (iDataInf[1] == NexusFile.NX_BOOLEAN) {
			oOutput = convertArray(new DataSet(oOutput));
		}

		dsData.setData(oOutput);

		dsData.isSingleRawArray(m_bResultAsSingleRaw);
		
		return dsData;
	}

	/**
	 * getChildrenDatas Scan currently opened group, to get all direct
	 * descendants' dataset and instrument informations (such as
	 * NXtechnical_data). Then return a list of DataSet.
	 * 
	 * @throws NexusException
	 */
	@SuppressWarnings("unchecked")
	protected Stack<DataSet> getChildrenDatas() throws NexusException {
		// Defining variables
		NexusNode nnCurNode;
		ArrayList<NexusNode> alNodeList;
		Stack<DataSet> alDataSet = new Stack<DataSet>();

		// Parse children
		alNodeList = (ArrayList<NexusNode>) listChildren().clone();
		for (int iIndex = 0; iIndex < alNodeList.size(); iIndex++) {
			nnCurNode = alNodeList.get(iIndex);
			if (!nnCurNode.isGroup()) {
				openData(nnCurNode.getNodeName());
				alDataSet.push(getDataSet());
				closeData();
			}
		}
		alDataSet.trimToSize();
		return alDataSet;
	}

	/**
	 * getDescendantsDatas Read all datasets that are descendants of the
	 * currently opened group
	 */
	protected Stack<DataSet> getDescendantsDatas() throws NexusException {
		NexusNode nnCurNode;
		ArrayList<NexusNode> lNodes;
		Stack<DataSet> sDatas = new Stack<DataSet>();

		// Get all direct descendants' datas
		sDatas.addAll(getChildrenDatas());

		// Get all direct descendants group
		lNodes = listChildren();

		// Start recursion
		for (int i = 0; i < lNodes.size(); i++) {
			nnCurNode = lNodes.get(i);
			if (nnCurNode.isGroup()) {
				openNode(nnCurNode);
				sDatas.addAll(getDescendantsDatas());
				closeGroup();
			}
		}

		return sDatas;
	}

	// ---------------------------------------------------------
	// / Node reading attribute
	// ---------------------------------------------------------
	/**
	 * getAttributeValue Return the value of the named attribute from the
	 * currently opened dataset
	 * 
	 * @param sAttrName
	 *            attribute's name from which value is requested
	 * @param hAttr
	 *            hashtable<string, string> containing list of attributes name
	 *            and their infos (first element type, second element length)
	 * @return an array object containing attributes value (in case of string
	 *         the string is directly returned)
	 */
	protected Object getAttributeValue(String sAttrName) throws NexusException {
		// Check that a file is opened
		if (!isFileOpened())
			throw new NexusException("No file opened!");

		Object oOutput = null;
		Object oTmpOut = null;

		try {
			oTmpOut = getAttribute(sAttrName);
		} catch (NexusException ne) {
			return null;
		}

		// Try to return the first array cell if there's no need of returning an
		// array
		if (oTmpOut != null && !(oTmpOut instanceof String)
				&& oTmpOut.getClass().isArray()) {
			if (java.lang.reflect.Array.getLength(oTmpOut) == 1)
				oOutput = java.lang.reflect.Array.get(oTmpOut, 0);
		} else
			oOutput = oTmpOut;

		return oOutput;
	}

	/**
	 * getDataSetAttribute Get attributs from a dataset. Those can be: name,
	 * description, unit and timestamp
	 * 
	 * @param dsData
	 * @throws NexusException
	 */
	protected void getDataSetAttribute(DataSet dsData) throws NexusException {
		// Get a map of dataset's attributs
		Hashtable<String, AttributeEntry> hAttrList = listAttribute();
		Object oAttrVal;
		String sAttrName;

		// Storing all encountered attributes
		for (Iterator<String> iter = hAttrList.keySet().iterator(); iter
				.hasNext();) {
			sAttrName = (String) iter.next();
			oAttrVal = getAttributeValue(sAttrName);
			dsData.setAttribute(sAttrName, oAttrVal);
		}
	}

	// ---------------------------------------------------------
	// / Checking methods
	// ---------------------------------------------------------
	/**
	 * checkData Check if given data fits the currently opened dataset
	 * 
	 * @param oData
	 *            array of data to compare with current node
	 * @throws NexusException
	 *             if node and data aren't compatible
	 */
	protected void checkDataMatch(Object oData) throws NexusException {
		DataSet dsData = new DataSet();
		dsData.initFromData(oData);

		checkDataMatch(dsData);
	}

	/**
	 * checkData Check if given data fits the currently opened dataset
	 * 
	 * @param dsData
	 *            DataSet properly initiated containing data to compare with
	 *            current node
	 * @throws NexusException
	 *             if node and data aren't compatible
	 */
	protected void checkDataMatch(DataSet dsData) throws NexusException {
		// Get infos on dataset (data type, rank, dimsize)
		int[] iDataInf = new int[2]; // iDataInf[0] = dataset rank ; iDataInf[1] = data type
		int[] iNodSize = new int[RANK_MAX]; // whole dataset dimension's sizes
		int[] iDimSize = dsData.getSize();  // DataSet dimension's sizes

		getNexusFile().getinfo(iNodSize, iDataInf);

		// Checking type compatibility
		if (dsData.getType() != iDataInf[1])
			throw new NexusException(
					"Datas and target node do not have compatible type!");

		// Checking rank compatibility
		if (iDimSize.length > iDataInf[0])
			throw new NexusException(
					"Datas and target node do not have compatible rank!");

		// Checking dimensions sizes compatibility
		{
			// The dimensions to check of the opened dataset are iDimSize.length
			// last ones
			int iDim = (iDataInf[0] - iDimSize.length);
			for (; iDim < iDataInf[0]; iDim++) {
				if (iDimSize[iDim + iDimSize.length - iDataInf[0]] != iNodSize[iDim])
					throw new NexusException(
							"Datas and target node do not have compatible dimension sizes!");
			}
		}
	}

	/**
	 * generateDataName Generate a name that doesn't exist yet, it uses the
	 * given base name to increment it.
	 * 
	 * @param pgPath
	 *            path to search a proper name in
	 * @param sNameBase
	 *            base name to be incremented until it isn't used
	 */
	protected String generateDataName(PathGroup pgPath, String sNameBase)
			throws NexusException {
		String sName;
		String sSepa = "_";
		int iNum = 0;

		// Open the requested path in current file
		try {
			openPath(pgPath);
		} catch (NexusException n) {
			// Close all nodes
			closeAll();

			// Try to open
			return sNameBase + sSepa + iNum;
		}

		// Try to find a name that doesn't exist
		while (true) {
			try {
				// Try to open
				sName = sNameBase + sSepa + iNum;
				openData(sName);
				// Succeed so we continue
				closeData();
			}
			// Name doesn't exist
			catch (NexusException ne) {
				// Close all nodes
				closeAll();

				// Return it as result
				return sNameBase;
			}
			iNum++;
		}
	}

	// ---------------------------------------------------------
	// ---------------------------------------------------------
	// / Protected static methods
	// ---------------------------------------------------------
	// ---------------------------------------------------------

	// ---------------------------------------------------------
	// / Conversion methods
	// ---------------------------------------------------------
	/**
	 * defineNexusFromType Return Nexus type according to the given data
	 * 
	 * @param oData
	 *            data from which Nexus type will be detected
	 */
	protected static int defineNexusFromType(Object oData)
			throws NexusException {
		// Check data existence
		if (oData == null)
			throw new NexusException("No data to determine Nexus data type!");

		int iType;

		String sClassName = oData.getClass().getName();
		sClassName = sClassName.substring(sClassName.lastIndexOf('[') + 1 );

		if (sClassName.startsWith("Ljava.lang.Byte") || sClassName.equals("B") )
		{
			iType = NexusFile.NX_BINARY;
		}
		else if (sClassName.startsWith("Ljava.lang.Integer") || sClassName.equals("I") )
		{
			iType = NexusFile.NX_INT32;
		}
		else if (sClassName.startsWith("Ljava.lang.Short") || sClassName.equals("S") )
		{
			iType = NexusFile.NX_INT16;
		}
		else if (sClassName.startsWith("Ljava.lang.Float") || sClassName.equals("F") )
		{
			iType = NexusFile.NX_FLOAT32;
		}
		else if (sClassName.startsWith("Ljava.lang.Double") || sClassName.equals("D") )
		{
			iType = NexusFile.NX_FLOAT64;
		}
		else if (sClassName.startsWith("Ljava.lang.Long") || sClassName.equals("J") )
		{
			iType = NexusFile.NX_INT64;
		}
		else if (sClassName.startsWith("Ljava.lang.Character") || sClassName.equals("C") || sClassName.startsWith("java.lang.String"))
		{
			iType = NexusFile.NX_CHAR;
		}
		else if (sClassName.startsWith("Ljava.lang.Boolean") || sClassName.equals("Z") )
		{
			iType = NexusFile.NX_BOOLEAN;
		}
		else {
			throw new NexusException("Unable to determine type of object!  '"+sClassName);
		}
		return iType;
	}

	/**
	 * defineNexusType This method returns the class corresponding to a
	 * NexusFile data type
	 * 
	 * @param iType
	 *            NexusFile data type (NX_INT16, NX_CHAR...)
	 * @throws NexusException
	 *             when the type is unknown
	 */
	protected static Class<?> defineTypeFromNexus(int iType)
			throws NexusException {
		Class<?> cObject;
		switch (iType) {
			case NexusFile.NX_INT16:
			case NexusFile.NX_UINT16:
				cObject = Short.TYPE;
				break;
			case NexusFile.NX_INT32:
			case NexusFile.NX_UINT32:
				cObject = Integer.TYPE;
				break;
			case NexusFile.NX_INT64:
			case NexusFile.NX_UINT64:
				cObject = Long.TYPE;
				break;
			case NexusFile.NX_FLOAT32:
				cObject = Float.TYPE;
				break;
			case NexusFile.NX_FLOAT64:
				cObject = Double.TYPE;
				break;
			case NexusFile.NX_CHAR:
			case NexusFile.NX_UINT8:
			case NexusFile.NX_INT8:
				cObject = Byte.TYPE;
				break;
			default:
				throw new NexusException("Unknown data type!");
		}
		return cObject;
	}

	/**
	 * initTypeFromNexus This method returns the class 0 value corresponding to
	 * a NexusFile data type
	 * 
	 * @param iType
	 *            NexusFile data type (NX_INT16, NX_CHAR...)
	 * @throws NexusException
	 *             when the type is unknown
	 */
	protected static Object initTypeFromNexus(int iType) throws NexusException {
		Object cObject;
		switch (iType) {
			case NexusFile.NX_INT16:
			case NexusFile.NX_UINT16:
				cObject = Short.valueOf((short) 0);
				break;
			case NexusFile.NX_INT32:
			case NexusFile.NX_UINT32:
				cObject = Integer.valueOf(0);
				break;
			case NexusFile.NX_INT64:
			case NexusFile.NX_UINT64:
				cObject = Long.valueOf(0L);
				break;
			case NexusFile.NX_FLOAT32:
				cObject = Float.valueOf(0);
				break;
			case NexusFile.NX_FLOAT64:
				cObject = Double.valueOf(0);
				break;
			case NexusFile.NX_CHAR:
			case NexusFile.NX_UINT8:
			case NexusFile.NX_INT8:
				cObject = Byte.valueOf((byte) 0);
				break;
			default:
				throw new NexusException("Unknown data type!");
		}
		return cObject;
	}

	/**
	 * defineArrayObject Returns an array of the corresponding type and
	 * dimension sizes
	 * 
	 * @param iType
	 *            integer corresponding to a Nexus data type
	 * @param iDimSize
	 *            array of sizes for each dimension
	 * @throws NexusException
	 */
	protected static Object defineArrayObject(int iType, int[] iDimSize)
			throws NexusException {
		Object oArray;
		Class<?> cObject = NexusFileReader.defineTypeFromNexus(iType);
		oArray = java.lang.reflect.Array.newInstance(cObject, iDimSize);
		return oArray;
	}

	/**
	 * convertArray Returns an array of values corresponding to writable type by
	 * the HDF library. It converts automatically the data type into its
	 * corresponding type. i.e: if dsData is a boolean array it converts it into
	 * byte array, if it's a byte array it converts it into a bool array...
	 * 
	 * @param dsData
	 *            the data to be converted into
	 * @throws NexusException
	 * 
	 * @note: conversions are the following: bool => byte, byte => bool
	 */
	protected Object convertArray(DataSet dsData) throws NexusException {
		String sClassName = dsData.getData().getClass().getName();
		sClassName = sClassName.substring(sClassName.lastIndexOf('[') + 1);
		Object oOutput;

		int[] iShape = detectArrayShape(dsData.getData());

		// Converting boolean to byte because Nexus API don't accept it
		if (sClassName.startsWith("Ljava.lang.Boolean")
				|| sClassName.equals("Z"))
			oOutput = java.lang.reflect.Array.newInstance(Byte.class, iShape);
		else if (sClassName.startsWith("Ljava.lang.Byte")
				|| sClassName.equals("B"))
			oOutput = java.lang.reflect.Array
					.newInstance(Boolean.class, iShape);

		/*
		 * ******** Warning: jnexus API doesn't support NX_INT64 for writing
		 * those lines convert NX_INT64 into NX_FLOAT64, so long to double //
		 * Converting long as double because the JAVA Nexus API uses C++ native
		 * methods whom don't support integer 64 bit else if(
		 * sClassName.startsWith("Ljava.lang.Long") || sClassName.equals("J") )
		 * oOutput = java.lang.reflect.Array.newInstance( Double.class,
		 * dsData.getSize());
		 */
		// Nothing to do
		else
			throw new NexusException("No data conversion requested!");

		return convertArray(oOutput, dsData.getData());
	}

	/**
	 * detectArrayShape Returns the shape of the given array. i.e each
	 * dimension's size of the array
	 * 
	 * @param oArray
	 *            input array which we want the shape detection
	 * @note if givzen Object isn't an array returns int[1] = {1}
	 */
	protected int[] detectArrayShape(Object oArray) {
		// Ensure it's an array
		if (!oArray.getClass().isArray()) {
			return new int[] { 1 };
		}

		int[] iRes;

		String sClassName = oArray.getClass().getName();

		// Detect number of dimension
		iRes = new int[sClassName.lastIndexOf('[') + 1];

		// Detect shape
		int iDimIndex = 0;
		while (sClassName.startsWith("[")) {
			sClassName = sClassName.substring(sClassName.indexOf('[') + 1);
			iRes[iDimIndex] = java.lang.reflect.Array.getLength(oArray);
			oArray = java.lang.reflect.Array.get(oArray, 0);
			iDimIndex++;
		}

		return iRes;
	}
	
    /**
     * checkLinkTarget open the target file and check if pointed data is a
     * dataset. Returns a string array of two elements. The first is the
     * attribute to distinguish if a dataset or nxgroup is pointed, the seconds
     * is the full sibling path (completed by missing class or names).
     * 
     * @param prTgtPath
     *            pointed node by the link
     * @param paSrcPath
     *            path of the starting node for the relative link
     * @return the corresponding absolute path if found, else return null
     */
    protected PathNexus checkRelativeLinkTarget(PathNexus prTgtPath,
            PathData paSrcPath) {
        PathNexus pnTarget = null;
        if (paSrcPath.isRelative())
            return null;

        // Construct an absolute path using the source and the target
        if (!prTgtPath.isRelative())
            pnTarget = prTgtPath;

        // Try to open the requested absolute path
        try {
            if (pnTarget == null) {
                PathRelative prRelTgtPath = new PathRelative(prTgtPath);
                pnTarget = prRelTgtPath
                        .generateAbsolutePath((PathNexus) paSrcPath);
            }

            openPath(pnTarget);
            pnTarget = getCurrentRealPath().clone();
            closeAll();
            return pnTarget;
        } catch (NexusException ne) {
            return null;
        }
    }

	// ---------------------------------------------------------
	// ---------------------------------------------------------
	// / Private methods
	// ---------------------------------------------------------
	// ---------------------------------------------------------

	// ---------------------------------------------------------
	// / Conversion methods
	// ---------------------------------------------------------
	/**
	 * convertArray Fills the oOutput array with oInput's values converted into
	 * (oOutput's) type. The value conversion is given by the convertValue()
	 * method
	 * 
	 * @param oOutput
	 *            an array having same dimension sizes than oInput
	 * @param oInput
	 *            an array of original values to be converted
	 */
	private Object convertArray(Object oOutput, Object oInput) {
		// Convert boolean into short
		if (oInput.getClass().isArray()) {
			Object oTmpArr = oInput;
			int iLength = java.lang.reflect.Array.getLength(oTmpArr);
			for (int j = 0; j < iLength; j++) {
				Object o = java.lang.reflect.Array.get(oTmpArr, j);
				if (o.getClass().isArray()) {
					Object oTmpOut = java.lang.reflect.Array.get(oOutput, j);
					java.lang.reflect.Array.set(oOutput, j, convertArray(
							oTmpOut, o));
				} else {
					java.lang.reflect.Array.set(oOutput, j, convertValue(o));
				}
			}
		} else
			java.lang.reflect.Array.set(oOutput, 0, convertValue(oInput));

		return oOutput;
	}

	private Object convertValue(Object o) {
		if (o.getClass().equals(Boolean.class))
			return (Boolean) o ? Byte.valueOf((byte) 1) : Byte.valueOf((byte) 0);
		else if (o.getClass().equals(Byte.class))
			return ((Byte) o).equals(Byte.valueOf((byte) 1)) ? Boolean.TRUE : Boolean.FALSE;
		else if (o.getClass().equals(Long.class))
			return ((Long) o).doubleValue();
		else
			return null;
	}
}
