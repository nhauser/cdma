package org.cdma.internal.dictionary;

/// @cond internal

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.exception.CDMAException;
import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;

/**
 * Solver class is used <b>internally</b> by the <i>Extendend Dictionary mechanism</i>.
 * It permits to define how a key should be resolved to obtain the corresponding CDMA object.
 * According how the solver has been constructed, the result of the solve method will be:
 * - if constructed using a IKey it will produce a LogicalGroup
 * - if constructed using a Path it will produce an IDataItem
 * - if constructed using an IPluginMethod it will return the result of the executed method
 * <p>
 * A Solver can only have one of the following: IPath, Path or IPluginMethod. Each given
 * parameter is exclusive regarding the others.
 * 
 * @author rodriguez
 *
 */

public class Solver {
    private IKey mKey;             // LogicalGroup to create
    private Path mPath;            // Physical path to open
    private IPluginMethod mMethod; // Method call to call
    
    public Solver( IKey key ) {
        mKey    = key;
        mPath   = null;
        mMethod = null;
    }
    
    public Solver( Path path ) {
        mPath   = path;
        mKey    = null;
        mMethod = null;
    }
    
    public Solver( IPluginMethod method ) {
        mMethod = method;
        mPath   = null;
        mKey    = null;
        
    }
    
    public List<IContainer> solve(Context context) {
        List<IContainer> result = null;
        
        // If the solver is a path
        if( mPath != null ) {
            // Clear the context of previously found nodes
            context.clearContainers();
            
            // Get the dataset
            result = new ArrayList<IContainer>();
            IGroup root = context.getDataset().getRootGroup();
            
            // Try to get all nodes at the targeted path
            try {
                result = root.findAllContainerByPath( mPath.getValue() );
            } catch (NoResultException e) {
                e.printStackTrace();
            }
        }
        // If the solver is a call on a method
        else if( mMethod != null ) {
            result = new ArrayList<IContainer>();
            try {
                // Execute the method
                mMethod.execute(context);
                
                // Get all items added by the method
                result.addAll( context.getContainers() );
            } catch (CDMAException e) {
                e.printStackTrace();
            }
        }
        // If the solver is a key create a LogicalGroup
        else if( mKey != null ) {
            result = new ArrayList<IContainer>();
            result.add( new LogicalGroup((LogicalGroup) context.getCaller(), mKey, context.getDataset()) );
        }
        // Return empty list
        else {
            result = new ArrayList<IContainer>();
        }
        
        // Update context with last executed solver 
        context.addSolver(this);
        return result;
    }
    
    /**
     * Give an access to the given IKey that constructed this object.
     * @return IKey implementation
     */
    public IKey getKey() {
        return mKey;
    }
    
    /**
     * Give an access to the given Path that constructed this object.
     * @return Path object
     */
    public Path getPath() {
        return mPath;
    }
    
    /**
     * Give an access to the given IPluginMethod that constructed this object.
     * @return IPluginMethod implementation
     */
    public IPluginMethod getPluginMethod() {
        return mMethod;
    }
    
}

/// @endcond internal