package org.gumtree.data.soleil.internal;

import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.soleil.NxsDataSet;
import org.nexusformat.NexusException;

import fr.soleil.nexus4tango.NexusFileWriter;
import fr.soleil.nexus4tango.NexusNode;
import fr.soleil.nexus4tango.PathGroup;
import fr.soleil.nexus4tango.PathNexus;

public class DictionaryDetector {
	final static String SEPARATOR = "_";
	final static String EXTENSION = ".xml";
	private NxsDataSet m_dataset;
	private Beamline   m_beamline;
	private DataModel  m_model;
	
	public DictionaryDetector(NxsDataSet dataset) {
		m_dataset = dataset;
	}

	public String getDictionaryName() throws FileAccessException {
		String fileName = null;
		if( m_beamline == null ) {
			detectBeamline();
		}
		if( m_model == null ) {
			detectDataModel();
		}
		
		//if( m_beamline != Beamline.UNKNOWN && m_model != DataModel.UNKNOWN ) {
			fileName = m_beamline.getName() + SEPARATOR + m_model.getName() + EXTENSION;
		//}
/*		else {
			throw new FileAccessException("Unable to detect a convenient mapping dictionary!\nSee: " + fileName);
		}
*/		
		return fileName;
	}
	
	public Beamline detectBeamline() {
		m_beamline = null;
		PathNexus path = new PathGroup(new String[] {"<NXentry>", "<NXinstrument>"});
		NexusFileWriter handler = m_dataset.getHandler();
		
		try {
			handler.openPath(path);
			m_beamline = Beamline.valueOf(handler.getCurrentPath().getCurrentNode().getNodeName().toUpperCase());
		} catch (Exception e) {
			m_beamline = Beamline.UNKNOWN;
		}
		
		try {
			handler.closeAll();
		} catch (NexusException e) {}
		
		return m_beamline;
	}
	
	public DataModel detectDataModel() {
		if( m_beamline == null ) {
			detectBeamline();
		}
		
		switch( m_beamline ) {
			case ANTARES:
				m_model = detectDataModelAntares();
				break;
			case SWING:
				m_model = detectDataModelSwing();
				break;
			case CONTACQ:
				m_model = detectDataModelICA();
				break;
			case METROLOGIE:
				m_model = detectDataModelMetrologie();
				break;
			case DISCO:
				m_model = detectDataModelDisco();
				break;
			case UNKNOWN:
			default:
				m_model = DataModel.UNKNOWN;
				break;
		}
		return m_model;
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

	// ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
    /// protected methods: to guess the data model in general way
    // ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
	protected boolean isScanServer() {
		boolean result = false;
		NexusFileWriter handler = m_dataset.getHandler();
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
		NexusFileWriter handler = m_dataset.getHandler();
		try {
			PathNexus path = new PathGroup(new String[] {"<NXentry>", "scan_data<NXdata>"});
			handler.openPath(path);
			try {
				handler.openNode( new NexusNode("time_1", "SDS") );
			} catch (NexusException e) {
				result = true;
			}
		}
		catch (NexusException e) { }
		
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
		else if( isFlyScan() ) {
			model = DataModel.FLYSCAN;
		}
		else {
			model = DataModel.PASSERELLE;
		}
		
		return model;
	}
}
