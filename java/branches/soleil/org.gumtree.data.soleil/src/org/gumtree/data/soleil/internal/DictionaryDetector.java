package org.gumtree.data.soleil.internal;


import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.soleil.navigation.NxsDataset;
import org.nexusformat.NexusException;
/*
import fr.soleil.nexus4tango.NexusFileWriter;
import fr.soleil.nexus4tango.NexusNode;
import fr.soleil.nexus4tango.PathData;
import fr.soleil.nexus4tango.PathGroup;
import fr.soleil.nexus4tango.PathNexus;
*/

public final class DictionaryDetector {
/*	private NxsDataset mDataset;
	private String     mBeamline;
	private String     mModel;
	private boolean    mDetected;
	
	public DictionaryDetector( NxsDataset dataset ) {
		mDataset  = dataset;
		mBeamline = "UNKNOWN";
		mModel    = "UNKNOWN";
		mDetected = false;
	}
	
	public String getDictionaryName() {
		if( ! mDetected ) {
			detectBeamline();
			detectModel();
			mDetected = true;
		}
		
		return mBeamline + "_" + mModel + ".xml";
	}

	private void detectBeamline() {
		
	}
	
	private void detectModel() {
		
	}
	*/
	
	
	
	
	
	/*
	static final String SEPARATOR = "_";
	static final String EXTENSION = ".xml";
	private NxsDataset mDataset;
	private Beamline   mBeamline;
	private DataModel  mModel;
	
	public DictionaryDetector(NxsDataset dataset) {
		mDataset = dataset;
	}

	public String getDictionaryName() throws FileAccessException {
		String fileName = null;
		if( mBeamline == null ) {
			detectBeamline();
		}
		if( mModel == null ) {
			detectDataModel();
		}
		
		fileName = mBeamline.getName() + SEPARATOR + mModel.getName() + EXTENSION;
		return fileName;
	}
	
	public Beamline detectBeamline() {
		mBeamline = null;
		PathNexus path = new PathGroup(new String[] {"<NXentry>", "<NXinstrument>"});
		NexusFileWriter handler = mDataset.getHandler();
	
		try {
			handler.openPath(path);
			String name = handler.getCurrentPath().getCurrentNode().getNodeName();
			mBeamline = Beamline.valueOf(name);
		} catch (Exception e) {
			mBeamline = Beamline.UNKNOWN;
		}
		
		try {
			handler.closeAll();
		} catch (NexusException e) {}
		
		return mBeamline;
	}
	
	public DataModel detectDataModel() {
		if( mBeamline == null ) {
			detectBeamline();
		}
		
		switch( mBeamline ) {
			case ANTARES:
				mModel = detectDataModelAntares();
				break;
			case SWING:
				mModel = detectDataModelSwing();
				break;
			case CONTACQ:
				mModel = detectDataModelICA();
				break;
			case METROLOGIE:
				mModel = detectDataModelMetrologie();
				break;
			case DISCO:
				mModel = detectDataModelDisco();
				break;
			case SAMBA:
				mModel = detectDataModelSamba();
				break;
			case UNKNOWN:
			default:
				mModel = DataModel.UNKNOWN;
				break;
		}
		return mModel;
	}
	
	
	// ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
    /// protected methods: to guess the data model associated to a specific beamline
    // ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
	protected DataModel detectDataModelAntares() {
		return detectStandardDataModel();
	}
	
	protected DataModel detectDataModelSwing() {
		return detectStandardDataModel();
	}
	
	protected DataModel detectDataModelICA() {
		return detectStandardDataModel();
	}
	
	protected DataModel detectDataModelMetrologie() {
		return detectStandardDataModel();
	}
	
	protected DataModel detectDataModelDisco() {
		return detectStandardDataModel();
	}
	
	protected DataModel detectDataModelSamba() {
		return detectStandardDataModel();
	}

	// ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
    /// protected methods: to guess the data model in general way
    // ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
	protected boolean isScanServer() {
		boolean result = false;
		NexusFileWriter handler = mDataset.getHandler();
		try {
			PathNexus path = new PathGroup(new String[] {"<NXentry>", "scan_data<NXdata>"});
			handler.openPath(path);
			handler.openNode( new NexusNode("time_1", "SDS") );
			result = true;
			handler.closeAll();
		}
		catch (NexusException e) { }
		
		try {
			handler.closeAll();
		} catch (NexusException e) { }
		
		return result;
	}
	
	protected boolean isFlyScan() {
		boolean result = false;
		NexusFileWriter handler = mDataset.getHandler();
		try {
			PathNexus path = new PathGroup(new String[] {"<NXentry>"});
			Object attr = handler.readAttr("model", path);
			if( attr != null && attr instanceof String ) {
				String stringAttr = (String) attr;
				result = (stringAttr.equalsIgnoreCase("flyscan"));
			}
			else
			{
				try {
					path = new PathData(new String[] {"<NXentry>", "scan_data<NXdata>"}, "time_1");
					handler.openPath(path);
				} catch (NexusException e) {
					result = true;
				}
			}
		}
		catch (NexusException e) { }
		
		try {
			handler.closeAll();
		} catch (NexusException e) { }
		
		return result;
	}
	
	protected boolean isQuickExafs() {
		boolean result = false;
		NexusFileWriter handler = mDataset.getHandler();
		try {
			PathNexus path = new PathGroup(new String[] {"QuickEXAFS*<NXentry>"});
			handler.openPath(path);
			result = true;
		}
		catch (NexusException e) {
			result = false;
		}
		
		try {
			handler.closeAll();
		} catch (NexusException e) { }
		
		return result;
	}
	
	protected DataModel detectStandardDataModel() {
		DataModel model = DataModel.UNKNOWN;
		
		if( isScanServer() ) {
			model = DataModel.SCANSERVER;
		}
		else if( isQuickExafs() ) {
			model = DataModel.QUICKEXAFS;
		}
		else if( isFlyScan() ) {
			model = DataModel.FLYSCAN;
		}
		else {
			model = DataModel.PASSERELLE;
		}
		
		return model;
	}
	*/
}
