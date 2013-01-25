package org.cdma.plugin.xml.navigation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.xml.array.XmlAttribute;

public abstract class XmlContainer implements IContainer, Cloneable {

	public static final String GROUP_SEPARATOR = "!";
	public static final String SHORT_NAME_SEPARATOR = "__";

	private final String mFactoryName;
	private String mName;
	private String mShortName;
	private final int mIndex;
	private final IDataset mDataset;
	private IGroup mParent;
	private final Map<String, IAttribute> mListAttributes = new HashMap<String, IAttribute>();

	public XmlContainer(String factory, String name, int index,
			IDataset dataset, IGroup parent) {
		mFactoryName = factory;
		mDataset = dataset;
		mParent = parent;
		mName = name;
		mIndex = index;
		if (index >= 0) {
			mShortName = mName + SHORT_NAME_SEPARATOR + mIndex;
		} else {
			mShortName = mName;
		}
	}

	@Override
	public abstract IContainer clone();

	@Override
	public String getFactoryName() {
		return mFactoryName;
	}

	@Override
	public void addOneAttribute(IAttribute attribute) {
		if (attribute != null && attribute.getName() != null) {
			mListAttributes.put(attribute.getName(), attribute);
		}
	}

	@Override
	public void addStringAttribute(String name, String value) {
		if (name != null && value != null) {
			mListAttributes.put(name, new XmlAttribute(mFactoryName, name,
					value));
		}
	}

	@Override
	public IAttribute getAttribute(String name) {
		IAttribute result = null;
		if (name != null) {
			result = mListAttributes.get(name);
		}
		return result;
	}

	// public IAttribute getAttribute(String name) {
	// IAttribute result = null;
	// if (name != null) {
	// mListAttributes.get(name);
	// }
	// return result;
	// }

	@Override
	public List<IAttribute> getAttributeList() {
		ArrayList<IAttribute> result = new ArrayList<IAttribute>(
				mListAttributes.values());
		return result;
	}

	@Override
	public IDataset getDataset() {
		return mDataset;
	}

	@Override
	public String getLocation() {
		StringBuilder builder = new StringBuilder();
		if (mParent != null) {
			builder.append(mParent.getLocation());
			builder.append(GROUP_SEPARATOR);
		}
		builder.append(getShortName());
		return builder.toString();
	}

	@Override
	public String getName() {
		return mName;
	}

	@Override
	public IGroup getParentGroup() {
		return mParent;
	}

	@Override
	public IGroup getRootGroup() {
		return mDataset.getRootGroup();
	}

	@Override
	public String getShortName() {
		return mShortName;
	}

	public int getIndex() {
		return mIndex;
	}

	@Override
	public boolean hasAttribute(String name, String value) {
		if (name != null && value != null) {
			IAttribute attr = mListAttributes.get(name);
			if (attr != null) {
				return (attr.getStringValue() == value);
			}
		}
		return false;
	}

	@Override
	public boolean removeAttribute(IAttribute attribute) {
		if (attribute != null) {
			IAttribute attr = mListAttributes.remove(attribute.getName());
			return (attr != null);
		}
		return false;
	}

	@Override
	public void setName(String name) {
		mName = name;
	}

	@Override
	public void setShortName(String name) {
		mShortName = name;
	}

	@Override
	public void setParent(IGroup group) {
		mParent = group;
	}

	@Override
	public long getLastModificationDate() {
		return -1;
	}

	public static int retrieveContainerIndex(String attributeName) {
		int result = -1;
		String[] splitAttr = attributeName.split(SHORT_NAME_SEPARATOR);
		if (splitAttr != null && splitAttr.length == 2) {
			try {
				result = Integer.parseInt(splitAttr[1]);
			} catch (NumberFormatException e) {
				// nothing to do, the result stay at -1
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return result;
	}
}
