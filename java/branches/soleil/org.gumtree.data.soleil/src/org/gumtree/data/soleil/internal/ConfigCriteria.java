package org.gumtree.data.soleil.internal;

import java.util.List;
import java.util.Vector;

import org.gumtree.data.soleil.internal.ConfigParameter.CriterionValue;
import org.gumtree.data.soleil.navigation.NxsDataset;
import org.jdom.Element;

public class ConfigCriteria {
	private Vector<ConfigParameterCriterion> mCriterion;
	
	public ConfigCriteria() {
		mCriterion = new Vector<ConfigParameterCriterion>();
	}
	
	public void add(Element domCriteria) {
		// Managing criterion
		List<?> nodes = domCriteria.getChildren("if");
		for( Object node : nodes ) {
			mCriterion.add( new ConfigParameterCriterion((Element) node) );
		}
	}
	
	public ConfigCriteria(Vector<ConfigParameterCriterion> item) {
		mCriterion = item;
	}
	
	public void add(ConfigParameterCriterion item) {
		mCriterion.add(item);
	}
	
	public boolean match( NxsDataset dataset ) {
		boolean result = true;
		
		for( ConfigParameterCriterion criterion : mCriterion ) {
			if( criterion.getValue(dataset).equals(CriterionValue.FALSE.toString()) ) {
				result = false;
				break;
			}
		}
		
		return result;
	}
}
