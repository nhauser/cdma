package org.gumtree.data.soleil.internal;

import java.io.IOException;

import org.gumtree.data.exception.NoResultException;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IGroup;
import org.jdom.Attribute;
import org.jdom.Element;

public final class ConfigParameterDynamic implements ConfigParameter {
	private String         mName;        // Parameter name
	private String         mPath;        // Path to seek in
	private CriterionType  mType;        // Type of operation to do
	private CriterionValue mTest;        // Expected property
	//private CriterionValue mValue;       // Expected property
	private String         mValue;       // Comparison value with the expected property
	
	
	ConfigParameterDynamic(Element parameter) {
		init(parameter);
		/*
		String tmp;
		mName = name;
		mPath = domCriterion.getAttributeValue("target");
		tmp = domCriterion.getAttributeValue("type");
		if( tmp != null ) {
			mType = CriterionType.valueOf(domCriterion.getAttributeValue("type").toString().toUpperCase());
		}
		tmp = domCriterion.getAttributeValue("value");
		if( tmp != null ) {
			mValue = CriterionValue.valueOf(tmp.toUpperCase());
		}
		mValueTest = domCriterion.getAttributeValue("constant");
		*/
	}
	
	public String getValue(IDataset dataset) {
		String result = "";
		IContainer cnt;
		switch( mType ) {
			case EXIST:
				cnt = openPath(dataset);
				CriterionValue crt;
				if( cnt != null ) {
					crt = CriterionValue.TRUE;
				}
				else {
					crt = CriterionValue.FALSE;
				}
				result = mTest.equals( crt ) ? CriterionValue.TRUE.toString() : CriterionValue.FALSE.toString();
				break;
			case NAME:
				cnt = openPath(dataset);
				if( cnt != null ) {
					 result = cnt.getShortName();
				}
				else {
					result = null;
				}
				break;
			case VALUE:
				cnt = openPath(dataset);
				if( cnt != null && cnt instanceof IDataItem ) {
					 try {
						result = ((IDataItem) cnt).getData().getObject(((IDataItem) cnt).getData().getIndex()).toString();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
				else {
					result = null;
				}
				break;
			case CONSTANT:
				result = mValue;
				break;
			case EQUAL:
				cnt = openPath(dataset);
				if( cnt != null && cnt instanceof IDataItem ) {
					String value;
					try {
						value = ((IDataItem) cnt).getData().getObject(((IDataItem) cnt).getData().getIndex()).toString();
						if( value.equals(mValue) ) {
							crt = CriterionValue.TRUE;
						}
						else {
							crt = CriterionValue.FALSE;
						}
						result = mTest.equals( crt ) ? CriterionValue.TRUE.toString() : CriterionValue.FALSE.toString();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
				else {
					result = null;
				}
				break;
			case NONE:
			default:
				result = null;
				break;
		}
		return result;
	}
	
	public String getName() {
		return mName;
	}
	
	// ---------------------------------------------------------
	/// Private methods
	// ---------------------------------------------------------
	private void init(Element dom) {
		// Given element is <parameter>
		Element valueNode = dom.getChild("value");
		mName = dom.getAttributeValue("name");
		mPath = valueNode.getAttributeValue("target");
		Attribute attribute;
		
		String test;
		
		attribute = valueNode.getAttribute("type");
		if( attribute != null )
		{
			mType  = CriterionType.valueOf(attribute.getValue().toUpperCase());
			mTest  = CriterionValue.NONE;
			String value = valueNode.getAttributeValue("value");
			if( value != null ) {
				mValue = value;
			}
			else {
				value = valueNode.getAttributeValue("constant");
				if( value != null ) {
					mValue = value;
				}
			}

			test = valueNode.getAttributeValue("test");
			if( test != null ) {
				mTest = CriterionValue.valueOf(test.toUpperCase());
			}
			mValue = value;
			return;
		}
		else {
			attribute = valueNode.getAttribute("exist");
			if( attribute != null ) {
				mType  = CriterionType.EXIST;
				mTest  = CriterionValue.valueOf(attribute.getValue().toUpperCase());
				mValue = "";
				return;
			}
				
			if( attribute == null ) {
				attribute = valueNode.getAttribute("equal");
				if( attribute != null ) {
					mType  = CriterionType.EQUAL;
					mTest  = CriterionValue.TRUE;
					mValue = attribute.getValue();
					return;
				}
			}
			if( attribute == null ) {
				attribute = valueNode.getAttribute("not_equal");
				if( attribute != null ) {
					mType  = CriterionType.EQUAL;
					mTest  = CriterionValue.FALSE;
					mValue = attribute.getValue();
					return;
				}
			}
			if( attribute == null ) {
				attribute = valueNode.getAttribute("constant");
				if( attribute != null ) {
					mType  = CriterionType.CONSTANT;
					mTest  = CriterionValue.NONE;
					mValue = attribute.getValue();
					return;
				}
			}
		}
		
		
		
		/*
		// Checking each attribute of the current node
		List<?> attributes = valueNode.getAttributes();
		for( Object att : attributes ) {
			attribute = (Attribute) att;
			name  = attribute.getName();
			value = attribute.getValue();

			// Case of a path
			if( name.equals("path") ) {
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
				mType  = CriterionType.EQUAL;
				mTest  = CriterionValue.FALSE;
				mValue = value;
			}
/*			else if( name.equals("value") ) {
				mType  = CriterionType.VALUE;
				mTest  = CriterionValue.NONE;
				mValue = value;
			}
			else if ( name.equals("constant") ) {
				mType  = CriterionType.VALUE;
				mTest  = CriterionValue.NONE;
				mValue = value;
			}
		}
		*/
	}
	
	private IContainer openPath( IDataset dataset ) {
		IGroup root = dataset.getRootGroup();
		IContainer result = null;
		try {
			result = root.findContainerByPath(mPath);
		} catch (NoResultException e) {
		}
		return result;
	}
}
