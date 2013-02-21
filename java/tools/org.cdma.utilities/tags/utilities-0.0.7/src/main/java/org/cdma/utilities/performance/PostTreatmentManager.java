package org.cdma.utilities.performance;

import java.lang.Thread.State;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Set;

import org.cdma.utilities.performance.internal.DelayedTreatment;

/**
 * This class is a helper to start asynchronous treatments. It permits to have the following
 * behaviors:<br/>
 *   - a queuing thread executing sequentially treatments immediately<br/>
 *   - a queuing thread that will not executing a treatment before a certain amount of time<br/>
 *   - parallel threads each one executing a given treatment
 *   
 * @author rodriguez
 */
public final class PostTreatmentManager {
    /**
     * Add the PostTreatment in the stack of pending process to be executed
     * as soon as possible. 
     * @param treatment to be executed as soon as possible
     * 
     * @note they are computed sequentially one after the other
     */
    static public void registerTreatment(PostTreatment treatment) {
        synchronized (pendingProcesses) {
        	// add the treatment to the queue
            pendingProcesses.add(treatment);
            
            // launch the thread if necessary
            launch(true);
        }
    }

    /**
     * Add the PostTreatment in the stack of pending process to be executed
     * at least the 'delay' in millisecond later.
     * @param treatment to be post-executed
     * @param delay in millisecond before executing
     * 
     * @note they are computed sequentially one after the other
     */
    static public void registerTreatment(PostTreatment treatment, long delay) {
        synchronized (delayedProcesses) {
        	// Create a new delayed treatment
        	long wait = System.currentTimeMillis() + delay;
        	DelayedTreatment delayed = new DelayedTreatment( treatment, wait );

        	// add the treatment to the queue
        	delayedProcesses.add( delayed );
            
            // launch the thread if necessary
            launch(false);
        }		
	}
    
    /**
     * Launch a dedicated new thread to execute the given treatment
     * @param treatment to be executed immediately
     * 
     * @return the Thread that where instantiated and started
     */
    static public Thread launchParallelTreatment(final PostTreatment treatment) {
    	return launchParallelTreatment(treatment, -1);
    }
    
    /**
     * Launch a dedicated new thread to execute the given treatment in at least 'delay' milliseconds
     * 
     * @param treatment to be executed
     * @param delay amount of milliseconds to wait before the process will be executed
     * 
     * @return the Thread that where instantiated and started (eventually sleeping)
     * @note if delay is 0 or negative the execution will be immediate
     */
    static public Thread launchParallelTreatment(final PostTreatment treatment, final long delay) {
    	Thread result = null;
    	if( treatment != null ) {
    		// create runnable
    		Runnable runner = new Runnable() {
				
				@Override
				public void run() {
					// if a delay is present
					if( delay > 0 ) {
						// wait for the given time
						try {
							Thread.sleep( delay );
						} catch (InterruptedException e) {
						}
					}
					
					// execute the treatment
					treatment.process();
				}
			};
			// create a new thread and start it
    		result = new Thread( runner );
    		result.start();
    	}
    	return result;
    }
    
    // ------------------------------------------------------------------------
    // Private members
    // ------------------------------------------------------------------------
	static private LinkedList<DelayedTreatment> delayedProcesses;
	static private LinkedList<PostTreatment>    pendingProcesses;
    static private LinkedList<PostTreatment>    currentProcesses;
    static private Runnable process;
    static private Thread   thread;
    static private long    lastTransfert = 0L;
    static private boolean waitingForWakeUp;
    static private boolean keepRunning;
    
    // ------------------------------------------------------------------------
    // Private methods
    // ------------------------------------------------------------------------
	static private PostTreatment getNextTreatment() {
    	PostTreatment result = null;
        // If there's more treatment to process in the queue
        if ( !currentProcesses.isEmpty() ) {
        	if( System.currentTimeMillis() - lastTransfert > 1000 ) {
        		transfertTreatment();
        	}
        	
        	// take the next one
        	result = currentProcesses.removeFirst();
        }
        // Else check no more are pending
        else {
        	transfertTreatment();
            if ( !currentProcesses.isEmpty() ) {
            	result = currentProcesses.removeFirst();
            }
        }
        return result;
	}
    
    /**
     * Transfers all treatment to be executed into the execution treatment task
     */
    static private void transfertTreatment() {
    	
    	// copy delayed treatment to the current queue
        synchronized (delayedProcesses) {
        	// removing list of treatments
        	Set<DelayedTreatment> removeList = new HashSet<DelayedTreatment>();
        	
        	// current time
        	long currentTime = System.currentTimeMillis();
        	for( DelayedTreatment delayed : delayedProcesses ) {
        		// check if execution time has come
        		if( delayed.getDelay() < currentTime ) {
        			removeList.add(delayed);
        			
        			// add to the queue 
    				currentProcesses.add( 0, delayed.getTreatment() );
        		}
        	}
        	// remove all pending element that have been transfered into the queue
        	delayedProcesses.removeAll(removeList);
        	
        	// check if there are element to be processed
        	keepRunning = ! delayedProcesses.isEmpty();
        }
    	
    	// copy pending treatment into the processing queue 
        synchronized (pendingProcesses) {
        	lastTransfert = System.currentTimeMillis();
            currentProcesses.addAll(0, pendingProcesses);
            pendingProcesses.clear();
            
            // check if there are element to be processed
            keepRunning = keepRunning || ! currentProcesses.isEmpty();
        }
    }
    
    /**
     * return the time when the earlier treatment is expected to be executed
     */
    static private long getSleepingTime() {
    	long result = -1;
        synchronized (delayedProcesses) {
        	long earlier = Long.MAX_VALUE;
        	for( DelayedTreatment delayed : delayedProcesses ) {
        		if( delayed.getDelay() < earlier ) {
        			earlier = delayed.getDelay();
        			result = earlier;
        		}
        	}
        }
        return result;
    }
    
    /**
     * Launch or awake the thread if necessary 
     * 
     * @param awake the thread if true
     */
    static private void launch(boolean awake) {
        // If no thread is running we launch a new one
    	boolean newThread = false;
    	if( thread == null ) {
        	synchronized( PostTreatmentManager.class ) {
        		if ( thread == null ) {
        			thread = new Thread(process, "CDMA post treatments");
            		thread.start();
            		newThread = true;
        		}
        	}
    	}
		if( ! newThread  ) {
			if( State.TERMINATED == thread.getState() ) {
				synchronized( thread ) {
					thread = new Thread(process, "CDMA post treatments");
					thread.start();
				}
			}
			else if( State.TIMED_WAITING == thread.getState() && awake ) {
				synchronized( thread ) {
					if( State.TIMED_WAITING == thread.getState() && ! waitingForWakeUp ) {
						waitingForWakeUp = true;
						thread.notify();
					}
				}
			}
		}
    }
    
    // ------------------------------------------------------------------------
    // Static block
    // ------------------------------------------------------------------------
    static {
    	synchronized( PostTreatmentManager.class ) {
    		waitingForWakeUp = true;
	        pendingProcesses = new LinkedList<PostTreatment>();
	        currentProcesses = new LinkedList<PostTreatment>();
	        delayedProcesses = new LinkedList<DelayedTreatment>();
	        keepRunning = true;
	    	process = new Runnable() {
	            @Override
	            public void run() {
	            	keepRunning = true;
	            	
	            	// Start pop-ing treatments
	                PostTreatment treatment;
	                
	                // While there are still treatment
	                while ( keepRunning ) {
	                    // get next treatment
	                    treatment = getNextTreatment();
	                	
	                    // execute the process
	                    if( treatment != null ) {
	                    	treatment.process();
	                    }
	                    // make the thread sleep until next treatment has to be executed
	                    else {
							long delay = getSleepingTime() - System.currentTimeMillis();
							if( delay > 0 ) {
								try {
									waitingForWakeUp = false;
									Thread.sleep(delay);
									
								} catch (InterruptedException e) {
								}
							}
	                    }
	                }
	                waitingForWakeUp = false;
	            }
	        };
    	}
    }
    
    /**
     * Avoid the construction of this class no instance are allowed
     */
    private PostTreatmentManager() {}
}
