package org.gumtree.data.soleil.internal;

import java.io.IOException;
import java.util.List;

import org.gumtree.data.exception.NoResultException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IGroup;
import org.jdom.Attribute;
import org.jdom.Element;

public class ConfigParameterCriterion implements ConfigParameter {
	private String         mPath;    // Path to seek in
	private CriterionType  mType;    // Type of operation to do
	private CriterionValue mTest;    // Expected value of the test
	private String         mValue;   // Comparison value with the expected property

	ConfigParameterCriterion(Element dom) {
		Attribute attribute;
		String value;
		String name;
		
		// Checking each attribute of the current node
		List<?> attributes = dom.getAttributes();
		for( Object att : attributes ) {
			attribute = (Attribute) att;
			name  = attribute.getName();
			value = attribute.getValue();

			// Case of a path
			if( name.equals("target") ) {
				mPath = value;
			}
			// Case of exist attribute
			else if( name.equals("exist") ) {
				mType  = CriterionType.EXIST;
				mTest  = CriterionValue.valueOf(value.toUpperCase());
				mValue = "";
				
			}
			// Case of exist attribute
			else if( name.equals("equal") ) {
				mType  = CriterionType.EQUAL;
				mTest  = CriterionValue.TRUE;
				mValue = value;
			}
			// Case of exist attribute
			else if( name.equals("not_equal") ) {
				mType = CriterionType.EQUAL;
				mTest  = CriterionValue.FALSE;
				mValue = value;
			}
			else {
				mTest = CriterionValue.NONE;
				mType = CriterionType.NONE;
			}
		}
		
		/*
		
		
		
		mPath = dom.getAttributeValue("path");
		tmp = dom.getAttributeValue("exist");
		if( tmp != null ) {
			mType  = CriterionType.EXIST;
			mValue = tmp;
		}
		else
		tmp = dom.getAttributeValue("value");
		if( tmp != null ) {
			mValue = CriterionValue.valueOf(tmp.toUpperCase());
		}
		mValueTest = dom.getAttributeValue("constant");
		*/
	}
	
	@Override
	public String getValue(IDataset dataset) {
		CriterionValue result;
		switch( mType ) {
			case EXIST:
				result = existPath(dataset);
				break;
			case EQUAL:
				result = getValuePath(dataset);
				break;
			default:
				result = CriterionValue.NONE;
				break;
		}
		
		if( result.equals(mTest) ) {
			result = CriterionValue.TRUE;
		}
		else {
			result = CriterionValue.FALSE;
		}
		
		return result.toString();
	}

	@Override
	public String getName() {
		return "";
	}

	
	
	
	// ---------------------------------------------------------
	/// Private methods
	// ---------------------------------------------------------
	private CriterionValue existPath( IDataset dataset ) {
		IGroup root = dataset.getRootGroup();
		IContainer result = null;
		try {
			result = root.findContainerByPath(mPath);
		} catch (NoResultException e) {
		}
		return result != null ? CriterionValue.TRUE : CriterionValue.FALSE;
	}
	
	private CriterionValue getValuePath( IDataset dataset ) {
		IGroup root = dataset.getRootGroup();
		IContainer item;
		String value = null;
		CriterionValue result;
		try {
			item = root.findContainerByPath(mPath);
			if( item instanceof IDataItem ) {
				IArray data = ((IDataItem) item).getData();
				if( data != null && data.getElementType() == String.class ) {
					value = (String) data.getObject( data.getIndex() );
				}
			}
		} catch (NoResultException e) {
		} catch (IOException e) {
		}
		
		if( mValue != null && mValue.equals(value) ) {
			result = CriterionValue.TRUE;
		}
		else {
			result = CriterionValue.FALSE;
		}
		
		return result;
	}
}
