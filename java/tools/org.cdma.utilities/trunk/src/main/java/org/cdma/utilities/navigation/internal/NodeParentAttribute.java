/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.utilities.navigation.internal;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.cdma.IFactory;
import org.cdma.exception.BackupException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;

public abstract class NodeParentAttribute extends NodeParent {

    private final Map<String, IAttribute> mAttributes; // Attributes of this
    private boolean mInitialized;

    public NodeParentAttribute(String factory, IDataset dataset, IGroup parent, String name) throws BackupException {
        super(factory, dataset, parent, name);
        mAttributes = new HashMap<String, IAttribute>();
    }

    public NodeParentAttribute(NodeParentAttribute object) throws BackupException {
        super(object);
        mAttributes = new HashMap<String, IAttribute>(object.mAttributes);
    }

    public void addOneAttribute(IAttribute attribute) {
        if (attribute != null) {
            initialize();
            mAttributes.put(attribute.getName(), attribute);
        }
    }

    public void addStringAttribute(String name, String value) {
        initialize();
        IFactory factory = getFactory();
        IAttribute attr = factory.createAttribute(name, value);
        addOneAttribute(attr);
    }

    public IAttribute getAttribute(String name) {
        initialize();
        IAttribute result = mAttributes.get(name);
        return result;
    }

    public final List<IAttribute> getAttributeList() {
        initialize();
        List<IAttribute> result = new ArrayList<IAttribute>(mAttributes.values());
        return result;
    }

    public boolean hasAttribute(String name, String value) {
        initialize();
        boolean result = false;
        if (name != null) {
            IAttribute attribute = getAttribute(name);
            if (attribute != null) {
                String attrVal = attribute.getStringValue();
                if ((value == null && attrVal == null) || value.equals(attrVal)) {
                    result = true;
                }
            }
        }
        return result;
    }

    public boolean removeAttribute(IAttribute attribute) {
        initialize();
        boolean result = false;
        if (attribute != null) {
            IAttribute attr = mAttributes.remove(attribute.getName());
            result = attr != null;
        }
        return result;
    }

    private void initialize() {
        synchronized (this) {
            if (!mInitialized) {
                mInitialized = true;
                initAttributes();
            }
        }
    }

    /**
     * Initialize internal values: children items and groups, attributes, dimensions
     */
    abstract protected void initAttributes();

}
