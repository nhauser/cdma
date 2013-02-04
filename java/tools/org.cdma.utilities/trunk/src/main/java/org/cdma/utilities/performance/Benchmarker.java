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

import java.util.ArrayList;
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
            } else {
                counters.put(label, (long) 1);
                starters.put(label, currentTime);
                timers.put(label, (long) 0);
                nbcalls.put(label, (long) 1);
                nbthread.put(label, (long) 1);
            }
        }
    }

    public static void stop(String label) {
        Long currentTime = System.currentTimeMillis();
        synchronized (Benchmarker.class) {
            if (counters.containsKey(label)) {
                Long counter = counters.get(label);
                if (counter > 1) {
                    counter--;
                    counters.put(label, counter);
                } else if (counter > 0) {
                    counter--;
                    Long starter = starters.get(label);
                    Long time = timers.get(label);
                    time += (currentTime - starter);
                    timers.put(label, time);
                    starters.put(label, (long) 0);
                    counters.put(label, counter);
                } else {
                    Factory.getLogger().log( Level.INFO, ">>>>>>>>>>>>>>>>>> Benchmark  <<<<<<<<<<<<<<<<<<<<<<");
                    Factory.getLogger().log( Level.INFO, "To much stop for: " + label);
                    Factory.getLogger().log( Level.INFO, ">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<");
                }
            } else {
                Factory.getLogger().log( Level.INFO, ">>>>>>>>>>>>>>>>>> Benchmark  <<<<<<<<<<<<<<<<<<<<<<");
                Factory.getLogger().log( Level.INFO, "Stoping inexistent timer: " + label);
                Factory.getLogger().log( Level.INFO, ">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<");
            }
        }
    }

    public static Map<String, Long> getTimers() {
        return timers;
    }

    public static String print() {
        String result = "";
        synchronized (Benchmarker.class) {
            for (String label : timers.keySet()) {
                result += print( label
                        + ": "
                        + ((Benchmarker.getTimers().get(label)) / (float) 1000) + " s",
                        60)
                        + print( " nb calls: " + nbcalls.get(label), 20 )
                        + print( " max thread: " +  nbthread.get(label), 20)
                        + ( " canonical cost: " + ( Benchmarker.getTimers().get(label) / (float) nbcalls.get(label) ) + " ms")
                        + "\n";
            }
        }
        return result;
    }

    private static String print(String value, int length ) {
        String result = value;

        if( value.length() < length ) {
            for( int i = 0; i < length - value.length(); i++ ) {
                result += " ";
            }
        }
        return result;
    }

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
            System.out.println("reset!!");
            timers   = new TreeMap<String, Long>();
            counters = new TreeMap<String, Long>();
            starters = new TreeMap<String, Long>();
            nbcalls  = new TreeMap<String, Long>();
            nbthread = new TreeMap<String, Long>();
            if (!reset) {
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

}
