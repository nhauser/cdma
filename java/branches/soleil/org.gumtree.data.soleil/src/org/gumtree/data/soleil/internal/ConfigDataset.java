package org.gumtree.data.soleil.internal;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.gumtree.data.interfaces.IDataset;
import org.jdom.Element;

public class ConfigDataset {
	private ConfigCriteria mCriteria;             // Criteria dataset should match for this configuration
	private Map<String, ConfigParameter> mParams; // Param name/value to set in the plugin when using this configuration
	private String mConfigLabel;                  // Name of the configuration

	/**
	 * Constructor of the dataset configuration
	 * need a dom element named: "config_dataset"
	 * @param config_dataset
	 */
	public ConfigDataset(Element config_dataset) {
		mConfigLabel = config_dataset.getAttributeValue("name");
		mParams = new HashMap<String, ConfigParameter>();
		init(config_dataset);
	}
	
	public String getLabel() {
		return mConfigLabel;
	}
	
	public Map<String, ConfigParameter> getParameters() {
		return mParams;
	}

	public String getParameter(String label, IDataset dataset) {
		String result = "";
		if( mParams.containsKey(label) ) {
			result = mParams.get(label).getValue(dataset);
		}
		return result;
	}

	public ConfigParameter getParameter(String label) {
		return mParams.get(label);
	}

	public ConfigCriteria getCriteria() {
		return mCriteria;
	}
	
	public void addParameter(String label, ConfigParameter param) {
		mParams.put(label, param);
	}
	
	
	// ---------------------------------------------------------
	/// Private methods
	// ---------------------------------------------------------
	private void init(Element config_dataset) {
		List<?> nodes;
		Element elem;
		Element section;
		ConfigParameter parameter;
		String name;
		String value;
		
		// Managing criteria
		mCriteria = new ConfigCriteria();
		section = config_dataset.getChild("criteria");
		if( section != null ) {
			mCriteria.add( section );
		}
		
		// Managing plugin parameters
		section = config_dataset.getChild("plugin");
		if( section != null ) {
			Element javaSection = section.getChild("java");
			if( javaSection != null ) {
				nodes = javaSection.getChildren("set");
				for( Object set : nodes ) {
					elem  = (Element) set;
					name  = elem.getAttributeValue("name");
					value = elem.getAttributeValue("value");
					parameter = new ConfigParameterStatic( name, value );
					mParams.put( name, parameter );
				}
			}
		}

		// Managing dynamic parameters
		section = config_dataset.getChild("parameters");
		if( section != null ) {
			nodes = section.getChildren("parameter");
			for( Object node : nodes ) {
				elem  = (Element) node;
				name  = elem.getAttributeValue("name");
				value = elem.getAttributeValue("value");

				// Dynamic parameter
				if( value == null ) {
					parameter = new ConfigParameterDynamic(elem);
				}
				// Static parameter (constant)
				else {
					parameter = new ConfigParameterStatic(name, value);
				}
				mParams.put(name, parameter);
			}
		}
	}
}
