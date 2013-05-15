package org.cdma.plugin.edf.abstraction;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.filter.IFilter;
import org.cdma.interfaces.IKey;

public class AbstractKey implements IKey, Cloneable {
    protected String id; // key name
    protected ArrayList<IFilter> filters; // filters

    public AbstractKey() {
        this("");
    }

    public AbstractKey(String name) {
        super();
        id = name;
        filters = new ArrayList<IFilter>();
    }

    @Override
    public List<IFilter> getFilterList() {
        return filters;
    }

    @Override
    public String getName() {
        return id;
    }

    @Override
    public void setName(String name) {
        id = name;
    }

    @Override
    public boolean equals(Object key) {
        if ((key != null) && (key.getClass().equals(getClass()))) {
            return (getName() == null && ((IKey) key).getName() == null)
                    || getName().equals(((IKey) key).getName());
        }
        else {
            return false;
        }
    }


    @Override
    public int hashCode() {
        int hash = 8;
        hash = hash * 121 + getClass().getName().hashCode();
        hash = hash * 121 + (getName() == null ? 0 : getName().hashCode());
        return hash;
    }

    @Override
    public String toString() {
        StringBuffer buffer = new StringBuffer(getClass().getSimpleName());
        buffer.append(" - key name: '").append(id);
        buffer.append("', filters: ").append(filters);
        return buffer.toString();
    }

    @Override
    public void pushFilter(IFilter filter) {
        filters.add(filter);
    }

    @Override
    public IFilter popFilter() {
        if (filters.size() > 0) {
            return filters.remove(0);
        }
        else {
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public IKey clone() {
        try {
            AbstractKey key = (AbstractKey) super.clone();
            key.filters = (ArrayList<IFilter>) filters.clone();
            return key;
        }
        catch (CloneNotSupportedException e) {
            // Should not happen because this class is Cloneable
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public String getFactoryName() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public int compareTo(Object arg0) {
        // TODO Auto-generated method stub
        return 0;
    }
}
