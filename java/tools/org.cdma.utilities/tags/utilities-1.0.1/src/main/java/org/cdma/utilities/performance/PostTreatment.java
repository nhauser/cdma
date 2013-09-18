package org.cdma.utilities.performance;

/**
 * This interface defines what is expected by the {@link PostTreatmentManager} to execute
 * asynchronous treatment.
 * @author rodriguez
 */
public interface PostTreatment {
	/**
	 * Method containing the code that should be executed asynchronously
	 */
	public void process();
	
	/**
	 * Name of the treatment
	 */
	public String getName();

}