package org.cdma.utilities.performance.internal;

import java.util.Date;

import org.cdma.utilities.performance.PostTreatment;


public class DelayedTreatment {
	private PostTreatment process;
	private long delay;
	
	public DelayedTreatment(PostTreatment process, long delay) {
		this.process = process;
		this.delay = delay;
	}
	
	public PostTreatment getTreatment() {
		return process;
	}
	
	public long getDelay() {
		return delay;
	}
	
	public String toString() {
		return "DelayedTreatment( "+ process.toString() +", " + new Date(delay).toString() + "  )"; 
	}
}