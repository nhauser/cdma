In the Nexus4Tango API, there is only 4 classes to manipulate: AcquisitionData, PathGroup / PathData and DataSet. Others are only for internal use.

**********************************
************* Classes
**********************************
AcquisitionData has two modes:
- first one corresponds to a standard use (like in DataRecorder)
- second one is used to write and read working files

AcquisitionData first mode, permits to scan and read a targeted NeXus file. More precisely:
- to get a list of scan sessions, 
- to get a list of instruments belonging to a scan, 
- to get an instrument's datas, 
- to count images belonging to a scan 
- to get the content of an image dataset

AcquisitionData second mode, permits to read and write datas directly in a targeted file. Datas will be stored following an XML tree, in which user 
will determine each node. Allowed actions are:
- write data into specified path
- read data from a specified path
- write external links pointing on a group or dataset belonging to another file
- read external links as if they were directly in the file containing the link (links are transparent)

DataSet, is used only to manipulate datas. Each time a data is asked a dataset will be returned. It gives some informations about data:
- the NeXus data type
- its dimensions et rank
- value of the data, for any kind of data it will be an array (but for string datas, the string will be directly returned)
- its attributs: unit, timestamp,...

PathGroup / PathData: it works in the same way, it wears info on how to find a file and how to open its contained group or dataset.
Allowed actions to perform are:
- adding a node (group or dataset)
- removing a node
- get numbered node item
- get its value as a link (relative or absolute)

**********************************
************* Classes
**********************************
Below are the main methods signature:

************* AcquisitionData (standard use like DataRecorder)
	// Constructors
	/**
	 * @param sNexusFilePath path to reach targeted NeXus file (optionnal)
	 */
	public AcquisitionData()
	public AcquisitionData(String sNexusFilePath)

	/**
	 * GetFile/SetFile
	 * Getter and setter of the targeted Nexus file's path
	 */
	public String	getFile()
	public void	setFile(String sFilePath)

	/**
	 * getData2D
	 * Read a 2D data from a NexusFile.
	 * 
	 * @param iIndex position of the 2D data to seek in the acquisition
	 * @param iAcqui number of the acquisition to seek in the current file (optionnal)
	 * @note if no iAcqui number is given, then the first encoutered one will be opened
	 */
	public DataSet getData2D(int iIndex ) throws NexusException
	public DataSet getData2D(int iIndex, int iAcqui) throws NexusException

	/**
	 * GetAcquiList
	 * return a string array string containing all Acquisition's name as direct children of a NeXus file
	 */
	public String[] getAcquiList() throws NexusException	

	/**
	 * GetInstrumentList
	 * return a string array containing all instrument belonging to an Acquisition (i.e.: NXpositionner, NXinsertion_device...)
	 * @note instruments aren't direct descendant of an Acquisition, they are children of an NXinstrument node
	 */
	public String[] getInstrumentList(String sAcquiName) throws NexusException

	/**
	 * GetInstrumentList
	 * return a string array containing all instrument having a specific type (i.e.: NXpositionner OR NXinsertion_device...) belonging to 
	 * an Acquisition
	 *
	 * @param sAcquiName name of the acquisition
	 * @param istInst type of instrument see
	 *
	 * @note see static enum belonging to AcquisitionData: AcquisitionData.Instrument.*
	 * @note instruments aren't direct descendant of an Acquisition, they are children of an NXinstrument node
	 */
	public String[] getInstrumentList(String sAcquiName, Instrument istInst) throws NexusException

	/**
	 * GetImageCount
	 * Count images belonging to an Acquisition
	 * 
	 * @param sAcquiName requested Acquisition to count images
	 */
	public int getImageCount(String sAcquiName) throws NexusException

	/**
	 * GetInstrumentData
	 * Return an array of DataSet filled by all datas stored in the targeted instrument
	 * 
	 * @param sAcquiName parent Acquisition of the requested instrument
	 * @param sInstrName instrument's name from which datas are requested
	 */
	public DataSet[] getInstrumentData(String sAcquiName, String sInstrName) throws NexusException


************* AcquisitionData (specific use: working on temporary file)
	/// Constructor
	/**
	 * @param sNexusFilePath path to reach targeted NeXus file (optionnal)
	 */
	public AcquisitionData()
	public AcquisitionData(String sNexusFilePath)

	/**
	 * GetFile/SetFile
	 * Getter and setter of the targeted Nexus file's path
	 */
	public String	getFile()
	public void	setFile(String sFilePath)

	/**
	 * ReadData
	 * Read datas to the given path
	 * 
	 * @param pdPath path to set datas in current file
	 */
	public DataSet readData(PathData pdPath) throws NexusException

	/**
	 * writeData
	 * Write a single data into the target path. Following methods are overload 
	 * of WriteData for all primitive types. 
	 * 
	 * @param tData a data value (single value or array) to be stored (of a primitive type)
	 * @param pdPath path to set datas in current file (can be absolute or local)
	 * @note if path don't exists it will be automtically created
	 */
	public <type> void writeData(type tData, PathData pdPath) throws NexusException

	/**
	 * WriteData
	 * Write datas to the given path
	 * 
	 * @param dsData a dataset object with values to be stored
	 * @param pdPath path to set datas in current file (can be absolute or local)
	 * @note if path don't exists it will be automtically created
	 */
	public void writeData(DataSet dsData, PathData pdPath) throws NexusException

	/**
	 * writeDataLink
	 * Create a link positioned in a dataset at pdPath in the current file. The dataset targeted by the link is stored
	 * in sFilePath following pdDataPath in the targeted file.
	 * 
	 * @param sFilePath  path to reach the cibling file
	 * @param pdDataPath path of the targeted data in cibling file
	 * @param pdPath	 path designing where to store the link in the current file
	 * @example We are currently working on file A and we want in A_path to link a dataset stored in B_path inside the B_file,
	 * the call will be: writeDataLink(B_file, B_path, A_path);
	 */
	public void writeDataLink(String sFilePath, PathData pdDataPath, PathData pdPath) throws NexusException
	
	/**
	 * writeGroupLink
	 * Create a link positioned at pgPath in the current file. The group targeted by the link is stored
	 * in sFilePath following pgDataPath in the targeted file.
	 * 
	 * @param sFilePath  path to reach the cibling file
	 * @param pgDataPath path of the targeted data in cibling file
	 * @param pgPath	 path designing where to store the link in the current file
	 * @example We are currently working on file A and we want in A_path to link a dataset stored in B_path inside the B_file,
	 * the call will be: writeGroupLink(B_file, B_path, A_path);
	 */
	public void writeGroupLink(String sFilePath, PathGroup pgDataPath, PathGroup pgPath) throws NexusException
	
	/**
	 * writeAttr
	 * Write an attribut on the node pointed by path.
	 * 
	 * @param tData the data attribut's value to be stored (template argument: iy can be any primitive or string)
	 * @param pnPath path to set datas in current file
	 * @note path must have the form: [/]name1/name2/...nameN/name_of_data
	 */
	public <type> void writeAttr(String sAttrName, type tData, PathNexus pnPath) throws NexusException

************* DataSet
	// Constructors
	/**
	 * DataSet
	 * @param oArray datas to store (if given type and size will be automatically set) (optionnal)
	 */
	public DataSet()
	public DataSet(Object oArray) throws NexusException

	// Getters
	/**
	 * getType
	 * @return the Nexus file dataset type
	 */
	public int 	getType()

	/**
	 * getSize
	 * @return an integer array containing each dimension size (the dataset's rank is given by GetSize().length )
	 * @note if data stored is null, an array with a single value equal to 0 will be returned
	 */
	public int[]	getSize()

	/**
	 * getData
	 * @return datas' stored in the dataset (in most cases it will be an array of a primitive type, but for NX_CHAR 
	 * dataset it will directly be the contained string)
	 * @note if data stored is null, the null object will be returned
	 */
	public Object	getData()

	/**
	 * getUnit
	 * @return datas' unit attribut
	 */
	public String	getUnit()

	/**
	 * getDesc
	 * @return datas' description attribut
	 */
	public String	getDesc()

	/**
	 * getTime
	 * @return datas' titmestamp attribut
	 */
	public String	getTime()

	/**
	 * toString
	 * @return a visual string representation of the dataset
	 */
	public String toString()

************* Object Path...
************* PathNexus 
	// protected Constructors
	protected PathNexus()
	protected PathNexus(String[] sGroups)
	protected PathNexus(NexusNode[] nnNodes)
	protected PathNexus(String[] sGroups, String sDataName)

************* PathGroup extends PathNexus
	/**
	 * PathGroup
	 * Create an object PathGroup
	 * @param sGroups array containing name of each group.
	 * @note group's class can be specified by adding "<" and ">" to a class name: i.e. "my_entry<NXentry>"
	 * @note be aware that it's better not to force the group class by default they are mapped by the API
	 */
	public PathGroup(String[] sGroups)
	
************* PathAcqui extends PathGroup
	/**
	 * PathAcqui
	 * Create an object PathAcqui
	 * 
	 * @param sAcquiName name of the acquisition
	 * @note group's class can be specified by adding "<" and ">" to a class name: i.e. "my_entry<NXentry>"
	 * @note BE AWARE that it's better not to force the group's class. By default they are mapped by the API to apply Nexus format DTD
	 */
	public PathAcqui(String sAcquiName)
	
************* PathInst extends PathAcqui
	/**
	 * PathInst
	 * Create an object PathInst
	 * 
	 * @param sAcquiName name of the acquisition which the instrument will belong to
	 * @param sInstName name of the instrument
	 * @note group's class can be specified by adding "<" and ">" to a class name: i.e. "my_entry<NXentry>"
	 * @note BE AWARE that it's better not to force the group's class. By default they are mapped by the API to apply Nexus format DTD
	 */

************* PathData extends PathNexus
	/**
	 * PathData
	 * Create an object PathData
	 * 
	 * @param sAcquiName name of the acquisition which the dataset will belong to
	 * @param sInstName name of the instrument which the dataset will belong to
	 * @param sDataName name of the dataset
	 * @note group's class can be specified by adding "<" and ">" to a class name: i.e. "my_entry<NXentry>"
	 * @note BE AWARE that it's better not to force the group's class. By default they are mapped by the API to apply Nexus format DTD
	 */
	public PathData(String sAcquiName, String sInstName, String sDataName)

************* Path... herited methods
	// public methods
	/**
	 * setFile
	 * Set the file's name
	 */
	public void setFile(String sFileName)
	
	/**
	 * setWorkingDir
	 * Set the current working directory (folder where to find file)
	 */
	public void setWorkingDir(String sWorkDir)
	
	/**
	 * getFilePath
	 * returns the path to reach the current file
	 */
	public String getFilePath()
	
	/**
	 * getWorkingDir
	 * returns the active working directory (folder where to find file)
	 */
	public String getWorkingDir()
	
	/**
	 * getDepth
	 * returns the path's depth (i.e number of group or dataset described inside)
	 */
	public int getDepth()
	
	/**
	 * getNode
	 * return node's name et class name at the iDepth depth in path 
	 * 
	 * @param iDepth depth position in path (starting to 0)
	 * @throws NexusException if a problem occurs
	 */
	public NexusNode getNode(int iDepth)
	
	/**
	 * getValue
	 * return the path value in a Nexus file as a String
	 */
	public String getValue()
	
	/**
	 * getRelativeLink
	 * return the path link value to reach an item in the current Nexus file 
	 */
	public String getRelativeLink() throws NexusException
	
	/**
	 * getAbsoluteLink
	 * return the absolute path link value to reach an item in the current Nexus file 
	 */
	public String getAbsoluteLink() throws NexusException
	
	/**
	 * getDataSetName
	 * returns the name of the dataset in path (node having isGroup == false)
	 * @note if no group group provided returns null
	 */
	public String getDataSetName()
	
	/**
	 * pushNode
	 * Add specified node to the last position in the path
	 * @param nnNode node to add in end of the path
	 */
	public void pushNode(NexusNode nnNode)
	
	/**
	 * popNode
	 * Remove the last node from the path and returns it
	 * @return node removed
	 */
	public NexusNode popNode()
	
	/**
	 * clearNodes
	 * Remove all node in the path (equivalent to document root)
	 */
	public void clearNodes()
	
	/**
	 * insertNode
	 * Insert a node in path to the iIndex position.
	 * @param iIndex position of insertion
	 * @param nnNode node to be inserted
	 */
	public void insertNode(int iIndex, NexusNode nnNode) throws NexusException
	
	/**
	 * locateRelativePath
	 * return a relative path to reach target file from the currently opened file
	 * @param sFilePath path pointing on target (can be relative or absolute)
	 */
	public String locateRelativePath(String sFilePath) throws NexusException

************* NexusFile definitions
    public final static int NX_FLOAT32 = 5; 
    public final static int NX_FLOAT64 = 6; 
    public final static int NX_INT8    = 20; 
    public final static int NX_BINARY  = 20;
    public final static int NX_UINT8   = 21; 
    public final static int NX_BOOLEAN = 21; 
    public final static int NX_INT16   = 22; 
    public final static int NX_UINT16  = 23; 
    public final static int NX_INT32   = 24; 
    public final static int NX_UINT32  = 25; 
    public final static int NX_INT64   = 26; 
    public final static int NX_UINT64  = 27; 
    public final static int NX_CHAR    = 4;

************* AcquisitionData.Instrument
	public enum Instrument {
		APERTURE			(...),
		ATTENUATOR			(...),
		BEAM_STOP			(...),
		BENDING_MAGNET		(...),
		COLLIMATOR			(...),
		CRYSTAL				(...),
		DETECTOR			(...),
		DISK_CHOPPER		(...),
		FERMI_CHOPPER		(...),
		FILTER				(...),
		FLIPPER				(...),
		GUIDE				(...),
		INSERTION_DEVICE	(...),
		INTENSITY_MONITOR	(...),
		MIRROR				(...),
		MODERATOR			(...),
		MONOCHROMATOR		(...),
		POLARIZER			(...),
		POSITIONER			(...),
		SOURCE				(...),
		VELOCITY_SELECTOR	(...)
	}

**********************************
************* Sample
**********************************


************* AcquisitionData (standard use)
String file_path = "my_file";

// Instanciate main object
AcquisitionData acqui = new AcquisitionData();

// Defining targeted NeXus file
acqui.setFile(file_path);

// Getting scan list (NXentry)
String[] list_acqui = acqui.getAcquiList();

// Getting instruments list belonging to a scan
String[] list_inst = acqui.getInstrumentList( list_acqui[0] );

// Getting images count belonging to a scan
int image_count = acqui.getImageCount( list_acqui[0] );

// Getting all datas' informations belonging to an instrument
DataSet[] datas = acqui.getInstrumentData( list_acqui[0], list_inst[0] );

// Getting an image
int num_image = 0;
int num_acqui = 0;
DataSet image = acqui.getData2D( num_image, num_acqui);



************* AcquisitionData (temporary file use)
String file_path1 = "../Samples/my_file_1.nxs";
String file_path2 = "../Samples/my_file_2.nxs";

// Instanciate main object
AcquisitionData acqui = new AcquisitionData();

// Defining targeted file
acqui.setFile(file_path1);

// Writing some datas
PathData pPath  = new PathData("system", "number", "value");
PathData pPath2;
acqui.writeData(8, pPath);

pPath = new PathData("system", "tab", "data");
acqui.writeData(new double[] { 12756.2, 40075.02, 5974200000000000000000000. }, pPath);

pPath = new PathData("system", "texte", "data1");
acqui.writeData("Hello world", pPath);

pPath = new PathData("system", "texte", "value_of_path_to_reach_this_dataset");
acqui.writeData("i'm in "+file_path1, pPath);

// Write a link into file_path2 pointing onto file_path_1 in system/texte/value_of_link
acqui.setFile(file_path2);
pPath2 = new PathData("system", "external_link", "value");
acqui.writeDataLink(file_path1, pPath, pPath2);
pPath2 = null;

// Reading some datas
acqui.setFile(file_path1);
System.out.println("Reading from file: " + file_path1);

DataSet dsData;

pPath  = new PathData("system", "tab", "data");
dsData = acqui.readData(pPath);
System.out.println("Array values: " + ((double[]) dsData.getData())[0] + " / " + ((double[]) dsData.getData())[1] + " / " + ((double[]) dsData.getData())[2]);

pPath  = new PathData("system", "texte", "data1");
dsData = acqui.readData(pPath);
System.out.println("Texte 1 (data1): " + dsData.getData());

pPath  = new PathData("system", "texte", "value_of_path_to_reach_this_dataset");
dsData = acqui.readData(pPath);
System.out.println("Texte 2 (value_of_link): " + dsData.getData());

// Reading link in second file
acqui.setFile(file_path2);
System.out.println("\nOpening new file: " + file_path2);
pPath2 = new PathData("system", "external_link", "value");
System.out.println("Reading data from dataset in: "+pPath2);
dsData = acqui.readData(pPath2);
System.out.println("Value of dataset: " + dsData.getData());


// Freeing memory
acqui = null;

