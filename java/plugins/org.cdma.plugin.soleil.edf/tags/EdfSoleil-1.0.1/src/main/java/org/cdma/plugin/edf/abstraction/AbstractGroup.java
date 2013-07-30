package org.cdma.plugin.edf.abstraction;

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
