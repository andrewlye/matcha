package matcha.engine;

import java.util.Arrays;

/**
 * Functionals: Loss
 * Contains various loss functions for Tensors.
 * @author andrewye
 */
public final class FN_Loss {
    private FN_Loss(){}

    public static Tensor cross_entropy(Tensor input, Tensor target){
        if (input.m_shape.length > 2) throw new UnsupportedOperationException("Error: K-dimensional loss not yet implemented.");
        // if containing class indices
        if (target.m_shape.length == 1) {
            return cross_entropy_indices(input, target);
        } else if (target.m_shape.length == 2) { // if containing class probabilities / one-hot encoding

        } 

        return null;
    }

    private static Tensor cross_entropy_indices(Tensor input, Tensor target){
        double[] targetPredictions = new double[target.m_shape[0]];
        for(int i = 0; i < targetPredictions.length; i++){
            targetPredictions[i] = input.get(new int[]{i, (int) target.m_data[i]});
        }

        double meanLoss = mean(m_log(targetPredictions));

        return new Tensor(new int[]{1}, new double[]{meanLoss}, true);
    }

    private static double[] m_log(double[] data){ return Arrays.stream(data).map(x -> -Math.log(x)).toArray(); }

    private static double mean(double[] data){ return Arrays.stream(data).average().orElse(0); }
    
}
