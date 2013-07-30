package org.cdma.plugin.edf.abstraction;

import java.util.ArrayList;
import java.util.List;

import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.edf.utils.StringUtils;

public abstract class AbstractObject implements IContainer, Cloneable {

    protected String name;
    protected IGroup parentGroup;
    protected ArrayList<IAttribute> attributes;

    public AbstractObject() {
        super();
        name = null;
        parentGroup = null;
        this.attributes = new ArrayList<IAttribute>();
    }

    @SuppressWarnings("unchecked")
    @Override
    public IContainer clone() {
        try {
            AbstractObject clone = (AbstractObject) super.clone();
            if (attributes != null) {
                clone.attributes = (ArrayList<IAttribute>) attributes.clone();
            }
            return clone;
        }
        catch (CloneNotSupportedException e) {
            // Should never happen, because this class implements Cloneable
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public void addOneAttribute(IAttribute attribute) {
        if ((attributes != null) && (attribute != null) && (!attributes.contains(attribute))) {
            attributes.add(attribute);
        }
    }

    @Override
    public List<IAttribute> getAttributeList() {
        if (attributes == null) {
            return null;
        }
        ArrayList<IAttribute> clone = new ArrayList<IAttribute>();
        clone.addAll(attributes);
        return clone;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }

    @Override
    public boolean hasAttribute(String name, String value) {
        if (attributes != null) {
            for (IAttribute attribute : attributes) {
                if ((attribute != null)) {
                    if (StringUtils.isSameString(name, attribute.getName())
                            && StringUtils.isSameString(value, attribute.getStringValue())) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    @Override
    public IAttribute getAttribute(String name) {
        if (attributes != null) {
            for (IAttribute attribute : attributes) {
                if ((attribute != null)) {
                    if (StringUtils.isSameString(name, attribute.getName())) {
                        return attribute;
                    }
                }
            }
        }
        return null;
    }

    @Override
    public boolean removeAttribute(IAttribute attribute) {
        if (attributes != null) {
            return attributes.remove(attribute);
        }
        return false;
    }

    @Override
    public IGroup getRootGroup() {
        IGroup group = parentGroup;
        while (true) {
            if ((group == null) || group.isRoot()) {
                break;
            }
            group = group.getParentGroup();
        }
        return group;
    }

    @Override
    public IGroup getParentGroup() {
        return parentGroup;
    }

    @Override
    public void setParent(IGroup group) {
        if ((parentGroup == null) || (!parentGroup.equals(group))) {
            this.parentGroup = group;
        }
    }

}
