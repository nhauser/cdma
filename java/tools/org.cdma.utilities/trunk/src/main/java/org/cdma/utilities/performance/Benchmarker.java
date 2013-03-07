//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr)
// See AUTHORS file
//******************************************************************************
package org.cdma.utilities.performance;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.logging.Level;

import org.cdma.Factory;

public class Benchmarker {
    private static Map<String, Long> timers   = new TreeMap<String, Long>();
    private static Map<String, Long> counters = new TreeMap<String, Long>();
    private static Map<String, Long> starters = new TreeMap<String, Long>();
    private static Map<String, Long> nbcalls  = new TreeMap<String, Long>();
    private static Map<String, Long> nbthread = new TreeMap<String, Long>();
    private static Map<String, Long> memFree  = new TreeMap<String, Long>();
    private static Map<String, Long> memCost  = new TreeMap<String, Long>();
    

    /**
     * Start a timer with the given label.
     * @param label
     */
    public static void start(String label) {
        Long currentTime = System.currentTimeMillis();
        synchronized (Benchmarker.class) {
            if (counters.containsKey(label)) {
                Long counter = counters.get(label);
                Long call = nbcalls.get(label);
                if (counter <= 0) {
                    counters.put(label, (long) 1);
                    starters.put(label, currentTime);
                } else {
                    counters.put(label, ++counter);
                    Long thread = nbthread.get(label);
                    if( thread < counter ) {
                        nbthread.put(label, counter);
                    }
                }
                nbcalls.put(label, ++call);
                long run  = Runtime.getRuntime().freeMemory();
                long free = memFree.get(label);
                if( free != 0 ) {
                	long cost = memCost.get(label) + (free - run);
                	memCost.put(label, cost);
                }
                memFree.put(label, run);
            } else {
                counters.put(label, 1L);
                starters.put(label, currentTime);
                timers.put(label, 0L);
                nbcalls.put(label, 1L);
                nbthread.put(label, 1L);
                memFree.put(label, Runtime.getRuntime().freeMemory());
                memCost.put(label, 0L);
            }
        }
    }

    /**
     * Stop the timer having the given label.
     * @param label
     */
    public static void stop(String label) {
        Long currentTime = System.currentTimeMillis();
        synchronized (Benchmarker.class) {
            if (counters.containsKey(label)) {
                Long counter = counters.get(label);
                
                if (counter > 1) {
                    counter--;
                    counters.put(label, counter);
                    long run  = Runtime.getRuntime().freeMemory();
                    long free = memFree.get(label);
                    if( free != 0 ) {
                    	long cost = memCost.get(label) + (free - run);
                    	memCost.put(label, cost);
                    }
                    memFree.put(label, run);
                } else if (counter > 0) {
                    counter--;
                    Long starter = starters.get(label);
                    Long time = timers.get(label);
                    time += (currentTime - starter);
                    timers.put(label, time);
                    starters.put(label, 0L);
                    counters.put(label, counter);
                    
                    long run  = Runtime.getRuntime().freeMemory();
                    long free = memFree.get(label);
                    if( free != 0 ) {
                    	long cost = memCost.get(label) + (free - run);
                    	memCost.put(label, cost);
                    }
                    memFree.put(label, 0L);
                    
                } else {
                    Factory.getLogger().log( Level.INFO, ">>>>>>>>>>>>>>>>>> Benchmark  <<<<<<<<<<<<<<<<<<<<<<");
                    Factory.getLogger().log( Level.INFO, "To much stop for: " + label);
                    Factory.getLogger().log( Level.INFO, ">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<");
                }
            } else {
                Factory.getLogger().log( Level.INFO, ">>>>>>>>>>>>>>>>>> Benchmark  <<<<<<<<<<<<<<<<<<<<<<");
                Factory.getLogger().log( Level.INFO, "Stopping inexistent timer: " + label);
                Factory.getLogger().log( Level.INFO, ">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<");
            }
        }
    }
    
    /**
     * Trace the calling sequence with the full stack trace.
     */
    public static void traceCall() {
    	traceCall(-1);
    }
    
    /**
     * Trace the calling sequence with the 'depthOfCall' last stack trace.
     * 
     * @param depthOfCall depth of the stack trace to be considered
     */
    public static void traceCall(int depthOfCall) {
    	Exception trace = new Exception();
    	String label = buildStringFromThrowable(trace, depthOfCall);
    	
        synchronized (Benchmarker.class) {
            if (counters.containsKey(label)) {
                Long call = nbcalls.get(label);
                nbcalls.put(label, ++call);
            } else {
                counters.put(label, 0L);
                starters.put(label, 0L);
                timers.put(label,   0L);
                nbcalls.put(label,  1L);
                nbthread.put(label, 0L);
                memCost.put(label, 0L);
                memFree.put(label, 0L);
            }
        }
    }

    /**
     * Return the map containing all started timers
     */
    public static Map<String, Long> getTimers() {
        return Collections.unmodifiableMap(timers);
    }
    
    /**
     * Return a String representation of all the static
     */
    public static String print() {
    	StringBuilder result = new StringBuilder();
    	synchronized (Benchmarker.class) {
	        int max = 0;
	        for( String label : timers.keySet() ) {
	        	String lines[] = label.split("\n");
	        	for( String line : lines ) {
	        		if( line.length() > max ) {
	        			max = line.length() + 2;
	        		}
	        	}
	        }
	        
	        for (String label : timers.keySet()) {
	            result.append( print( label + ": ", max ) );
	            result.append( print( roundNumber(Benchmarker.getTimers().get(label) / (float) 1000, 3) + " s", 10) );
	            result.append( print( " nb calls: " + nbcalls.get(label), 20 ) );
	            result.append( print( " max thread: " +  nbthread.get(label), 20) );
	            result.append( print( " canonical cost: " + roundNumber( Benchmarker.getTimers().get(label) / (float) nbcalls.get(label), 3 ) + " ms ..", 30 )  );
	            result.append( print( " used mem var: " + roundNumber( Benchmarker.memCost.get(label) / (float) 1000000, 3), 25 ) + " Mo\n" );
	        }
    	}
    	
        return result.toString();
    }
    
    private static String roundNumber( Number num, int nbDigits ) {
    	String format = "#.";
    	char[] digits = new char[nbDigits];
    	Arrays.fill(digits, '#');
    	format += String.copyValueOf(digits);
    	DecimalFormat df = new DecimalFormat(format);
    	return df.format(num);
    }

    /**
     * Reset all informations
     */
    public static void reset() {
        synchronized (Benchmarker.class) {
            boolean reset = true;
            List<String> running = new ArrayList<String>();
            for( Entry<String,Long> counter : counters.entrySet() ) {
                if( counter.getValue() > 0 ) {
                	running.add(counter.getKey());
                    reset = false;
                }
            }
            if( reset ) {
                timers   = new TreeMap<String, Long>();
                counters = new TreeMap<String, Long>();
                starters = new TreeMap<String, Long>();
                nbcalls  = new TreeMap<String, Long>();
                nbthread = new TreeMap<String, Long>();
                memFree  = new TreeMap<String, Long>();
                memCost  = new TreeMap<String, Long>();
            }
            else {
            	StringBuffer log = new StringBuffer();
            	log.append( ">>>>>>>>>>>>>>>>>> Benchmark  <<<<<<<<<<<<<<<<<<<<<<\n" );
            	log.append( "Timers are still running!!!!\n" );
            	log.append( "Please stop the following timers:\n" );
                for( String timer : running ) {
                	log.append( "   - " + timer + "\n" );
                }
                log.append( ">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<" );
                Factory.getLogger().log( Level.INFO, log.toString() );
            }
        }
    }

    // ------------------------------------------------------------------------
    // private methods
    // ------------------------------------------------------------------------
    private static String print(String value, int length ) {
        String result = value;

        if( value.length() < length ) {
            for( int i = 0; i < length - value.length(); i++ ) {
                result += ".";
            }
        }
        return result;
    }
    
    private static String buildStringFromThrowable(Throwable e, int depth) {
	    StringBuilder sb = new StringBuilder();
	    int i = -1;
	    int depthCall = depth > e.getStackTrace().length ? e.getStackTrace().length : depth;
	    for (StackTraceElement element : e.getStackTrace()) {
	    	if( i > -1 && ( i < depthCall || depth < 0 ) ) {
	    		sb.append(element.toString());
	    	}
	    	else if( i == depth && depth >= 0 ) {
	    		break;
	    	}
	    	if( i < depthCall - 1 && i > -1 ) {
	    		sb.append("\n");
	    	}
	    	i++;
	    }
	    return sb.toString();
    }
}
