package matchings;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import bayonet.distributions.Random;
import blang.core.LogScaleFactor;
import blang.distributions.Generators;
import blang.mcmc.ConnectedFactor;
import blang.mcmc.SampledVariable;
import blang.mcmc.Sampler;
import blang.mcmc.internals.Callback;
import briefj.collections.UnorderedPair;

/**
 * Each time a Permutation is encountered in a Blang model, 
 * this sampler will be instantiated. 
 */
public class PermutationSampler implements Sampler {
  /**
   * This field will be populated automatically with the 
   * permutation being sampled. 
   */
  @SampledVariable Permutation permutation;
  /**
   * This will contain all the elements of the prior or likelihood 
   * (collectively, factors), that depend on the permutation being 
   * resampled. 
   */
  @ConnectedFactor List<LogScaleFactor> numericFactors;

  @Override
  public void execute(Random rand) {
      // Fill this.
	  List<Integer> conBefore = permutation.getConnections();
      final double logBefore = logDensity();
      ArrayList<Integer> deepconBefore = new ArrayList<Integer>(conBefore);
      permutation.sampleUniform(rand);
      List<Integer> conAfter = permutation.getConnections();
	  final double logAfter = logDensity();
	  final double ratio = Math.exp(logAfter - logBefore);
	  boolean u = Generators.bernoulli(rand, Math.min(1.0, ratio));
	  if(!u) {
		  for (int i = 0; i < deepconBefore.size(); i ++ ) {
			  permutation.getConnections().set(i, deepconBefore.get(i));
		  }
	  }
  }
  
  private double logDensity() {
    double sum = 0.0;
    for (LogScaleFactor f : numericFactors)
      sum += f.logDensity();
    return sum;
  }
  
  public abstract void propose(Random random, Callback callback);

}
