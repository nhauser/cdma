package fr.soleil.nexus;


// Tools lib
import java.io.File;

// Nexus lib
import org.nexusformat.NexusException;
import org.nexusformat.NexusFile;

public class NexusFileInstance {
	/// Definitions
	protected static final String	PATH_SEPARATOR	= "/";				// Node separator in path when having a string representation
	protected final static int		RANK_MAX		= 32;				// Maximum dimension rank

	///	Member attributes
	private String 		  	   m_sFilePath; 	// Path to current file
	private NexusFileHandler   m_nfFile;		// Current file
	private int				   m_iAccessMode; // The current access mode to the file: read / write

	/// Constructors
	protected NexusFileInstance()					{ m_sFilePath = ""; }
	protected NexusFileInstance(String sFilePath)	{ m_sFilePath = sFilePath; }

	/// Accessors
	public    String	getFilePath() 				{ return m_sFilePath; }
	public	  void		setFile(String sFilePath)	{ m_sFilePath = sFilePath; try { closeFile(); } catch(Throwable t) {} }
	protected NexusFileHandler getNexusFile() throws NexusException
	{
		if(m_nfFile == null)
			throw new NexusException("No file currently opened!");
		return m_nfFile;
	}

	public boolean isFileOpened()
	{
		return m_nfFile != null;
	}

	/**
	 * getFileAccessMode
	 * @return the currently access mode for the file
	 * @throws NexusException if no file opened
	 */
	public int getFileAccessMode() throws NexusException
	{
		if( isFileOpened() )
			return m_iAccessMode;
		else
			throw new NexusException("No file currently opened!");
	}

	/**
	 * finalize
	 * Free ressources
	 */
	@Override
	protected void finalize() throws Throwable
	{
		closeFile();
		m_sFilePath = null;
		m_nfFile 	= null;
	}

	// ---------------------------------------------------------
	/// Protected methods

	// ---------------------------------------------------------
	/// File manipulation
	// ---------------------------------------------------------
	/**
	 * openFile
	 *
	 * @param sFilePath the full path to reach the NeXus file (including its name) (optionnal)
	 * @param iAccesMode the requested acces mode (read and/or write) (optionnal)
	 * @note if path isn't specified, the default will be the AcquisitionData specified one
	 * @note if access mode isn't, the default will be read-only
	 * @throws NexusException
	 */
	public void openFile() 						throws NexusException	{ openFile(m_sFilePath, NexusFile.NXACC_READ); }
	protected void openFile(String sFilePath)	throws NexusException	{ openFile(sFilePath, NexusFile.NXACC_READ); }
	protected void openFile(int iAccessMode)	throws NexusException	{ openFile(m_sFilePath, iAccessMode); }
	protected void openFile(PathNexus p)		throws NexusException	{ openFile(p, NexusFile.NXACC_READ); }
	protected void openFile(PathNexus p, int iAccessMode) throws NexusException
	{
		if( ! m_sFilePath.equals(p.getFilePath()) )
		{
			openFile(p.getFilePath(), iAccessMode);
			return;
		}
		openFile(iAccessMode);
	}

	protected void openFile(String sFilePath, int iAccessMode) throws NexusException
	{
		boolean openFile = false;

		// No file yet opened
		if( null == m_nfFile)
		{
			openFile = true;
		}
		else
		{
			// Check file isn't opened yet
			String sCurFile = "";
			try
			{
				sCurFile = m_nfFile.inquirefile();
			}
			catch(NexusException ne )
			{
				openFile = true;
			}

			// Check file opened is the good one and have correct write access
			if( !sFilePath.equals(sCurFile) || iAccessMode != m_iAccessMode )
			{
				m_nfFile.close();
				openFile = true;
			}
		}

		// open the file if needed
		if( openFile )
		{
			open(sFilePath, iAccessMode);
			m_iAccessMode = iAccessMode;
		}
	}

	private void open(String sFilePath, int iAccessMode) throws NexusException
	{
		// Try to open file
		m_sFilePath = sFilePath;

		// Transform path into absolute
		File file = new File(sFilePath);
		sFilePath = file.getAbsolutePath();

		if( NexusFile.NXACC_RDWR == iAccessMode )
		{

			if( file.exists() )
			{
				m_nfFile = new NexusFileHandler(sFilePath, iAccessMode);
			}
			else
				m_nfFile = new NexusFileHandler(sFilePath, NexusFile.NXACC_CREATE5);
		}
		else if( file.exists() )
		{
			m_nfFile = new NexusFileHandler(sFilePath, iAccessMode);
		}
		else
			throw new NexusException("Can't open file for read: " + sFilePath + " doesn't exist!");
		m_iAccessMode = iAccessMode;
	}

	/**
	 * closeFile
	 * Close the current file, but keep its path so we can easily open it again.
	 */
	protected void closeFile() throws NexusException
	{
		try
		{
			if( m_nfFile != null )
				m_nfFile.finalize();
		}
		catch(Throwable t)
		{
			if( t instanceof NexusException )
			{
				throw (NexusException) t;
			}
			else
				t.printStackTrace();
		}
		m_nfFile = null;
	}

	// ---------------------------------------------------------
	/// Debugging methods
	// ---------------------------------------------------------
	public static void ShowFreeMem(String lab )
	{
		System.out.println("Free mem " + Runtime.getRuntime().freeMemory() + " (octet) -> " + lab);
	}
}

