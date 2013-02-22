package org.cdma.engine.archiving.internal.attribute;

import java.util.HashMap;
import java.util.Map;

import org.cdma.engine.archiving.internal.SqlFieldConstants;

/**
 * Class that manages the path of an archived attribute. It is compound of:
 * domain, family, member, attribute name
 * It is a helper to determine which is the next step to reach the attribute using its
 * full name (i.e. path having the form: "domain/family/member/attribute_name").
 * 
 * @author rodriguez
 * 
 */
public class AttributePath implements Cloneable {
	final static public String SEPARATOR = "/";

	private String mDomain;
	private String mFamily;
	private String mMember;
	private String mAttName;
	private String mField;

	public AttributePath() {
		mDomain  = null;
		mFamily  = null;
		mMember  = null;
		mAttName = null;
		mField   = null;
	}
	
	public AttributePath(String path) {
		String[] pieces = path.split(SEPARATOR);
		mDomain  = pieces.length > 0 ? pieces[0] : null;
		mFamily  = pieces.length > 1 ? pieces[1] : null;
		mMember  = pieces.length > 2 ? pieces[2] : null;
		mAttName = pieces.length > 3 ? pieces[3] : null;
		mField   = pieces.length > 4 ? pieces[4] : null;
	}

	public AttributePath(String domain, String family, String member, String attribute) {
		mDomain  = domain;
		mFamily  = family;
		mMember  = member;
		mAttName = attribute;
		mField   = null;
	}

	public String getDomain() {
		return mDomain;
	}

	public void setDomain(String domain) {
		this.mDomain = domain;
	}

	public String getFamily() {
		return mFamily;
	}

	public void setFamily(String family) {
		this.mFamily = family;
	}

	public String getMember() {
		return mMember;
	}
	
	public String getField() {
		return mField;
	}
	
	public void setField( String field ) {
		this.mField = field;
	}

	public void setMember(String member) {
		this.mMember = member;
	}

	public void setAttributeName(String name) {
		this.mAttName = name;
	}
	

	@Override
	public String toString() {
		StringBuffer buf = new StringBuffer();

		// Domain
		buf.append(mDomain != null ? mDomain : "");
		buf.append(SEPARATOR);

		// Family
		buf.append(mFamily != null ? mFamily : "");
		buf.append(SEPARATOR);

		// Member
		buf.append(mMember != null ? mMember : "");
		buf.append(SEPARATOR);
		
		// Attribute name
		buf.append(mAttName != null ? mAttName : "");
		buf.append(SEPARATOR);
		
		// Field name
		buf.append(mField != null ? mField : "");
		buf.append(SEPARATOR);

		return buf.toString();
	}
	
	public boolean isEmpty() {
		return (mDomain == null && mFamily == null && mMember == null && mAttName == null);
	}
	
	/**
	 * Return true if this path is fully qualified to target an archived attribute
	 * @return true / false
	 */
	public boolean isFullyQualified() {
		return (getNextDbFieldName() == null);
	}
	
	/**
	 * Return the field name of the table where to seek for the next 
	 * step to have a complete path.
	 * 
	 * @return the DB field name or null if the path is complete
	 */
	public String getNextDbFieldName() {
		String result = null;
		
		if( mDomain == null ) {
			result = SqlFieldConstants.ADT_FIELDS_DOMAIN;
		}
		else if( mFamily == null ) {
			result = SqlFieldConstants.ADT_FIELDS_FAMILY;
		}
		else if( mMember == null ) {
			result = SqlFieldConstants.ADT_FIELDS_MEMBER;
		}
		else if( mAttName == null ) {
			result = SqlFieldConstants.ADT_FIELDS_ATT_NAME;
		}
		
		return result;
	}

	/**
	 * Set the name of the next field to reach the children attribute
	 * @param value name of the next node
	 */
	public void setNextDbField(String value) {
		if( mDomain == null ) {
			mDomain = value;
		}
		else if( mFamily == null ) {
			mFamily = value;
		}
		else if( mMember == null ) {
			mMember = value;
		}
		else if( mAttName == null ) {
			mAttName = value;
		}
	}
	
	public String getShortName() {
		String result;
		
		if( mField != null ) {
			result = mField;
		}
		else if( mAttName != null ) {
			result = mAttName;
		}
		else if( mMember != null ) {
			result = mMember;
		}
		else if( mFamily != null ) {
			result = mFamily;
		}
		else {
			result = mDomain;
		}
		return result;
	}
	
	public void setShortName(String name) {
		if( mField != null ) {
			mField = name;
		}
		else if( mAttName != null ) {
			mAttName = name;
		}
		else if( mMember != null ) {
			mMember = name;
		}
		else if( mFamily != null ) {
			mFamily = name;
		}
		else {
			mDomain = name;
		}
	}
	
	/**
	 * The fully qualified attribute's name: the complete path to reach it. 
	 */
	public String getName() {
		StringBuffer buf = new StringBuffer();
	
		// Domain
		if( mDomain != null ) {
			buf.append(mDomain != null ? mDomain : "");

			// Family
			if( mFamily != null ) {
				buf.append(SEPARATOR);
				buf.append(mFamily != null ? mFamily : "");
				
				// Member
				if( mMember != null ) {
					buf.append(SEPARATOR);
					buf.append(mMember != null ? mMember : "");

					// Attribute name
					if( mAttName != null ) {
						buf.append(SEPARATOR);
						buf.append(mAttName != null ? mAttName : "");
						
						// Field name (optional)
						if( mField != null ) {
							buf.append(SEPARATOR);
							buf.append(mField != null ? mField : "");
						}
					}
				}
			}
		}

		return buf.toString();
	}
	
	/**
	 * Return a map having the field name for entry associated to
	 * the value.
	 */
	public Map<String, String> getSqlCriteria() {
		Map<String, String> result = new HashMap<String, String>();
		
		if( mDomain != null ) {
			result.put( SqlFieldConstants.ADT_FIELDS_DOMAIN, mDomain );
		}
		if( mFamily != null ) {
			result.put( SqlFieldConstants.ADT_FIELDS_FAMILY, mFamily );
		}
		if( mMember != null ) {
			result.put( SqlFieldConstants.ADT_FIELDS_MEMBER, mMember );
		}
		if( mAttName != null ) {
			result.put( SqlFieldConstants.ADT_FIELDS_ATT_NAME, mAttName );
		}
		
		return result;
	}
	
	@Override
	public AttributePath clone() {
		AttributePath path = new AttributePath();
		path.mDomain  = this.mDomain;
		path.mMember  = this.mMember;
		path.mFamily  = this.mFamily;
		path.mAttName = this.mAttName;
		path.mField   = this.mField;
		return path;
	}
	
	public String[] getNodes() {
		return getName().split(SEPARATOR);
	}
}
