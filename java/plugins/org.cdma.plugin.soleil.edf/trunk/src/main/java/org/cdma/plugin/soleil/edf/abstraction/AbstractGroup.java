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
package org.cdma.plugin.soleil.edf.abstraction;

import org.cdma.interfaces.IGroup;
import org.cdma.utils.Utilities.ModelType;

public abstract class AbstractGroup extends AbstractObject implements IGroup {

    public AbstractGroup() {
        super();
    }

    @Override
    public ModelType getModelType() {
        return ModelType.Group;
    }

    @Override
    public IGroup getRootGroup() {
        if (isRoot()) {
            return this;
        }
        return super.getRootGroup();
    }

    @Override
    public IGroup clone() {
        return (IGroup) super.clone();
    }

}
