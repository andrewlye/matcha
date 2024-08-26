package matcha.engine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import matcha.engine.Tensor.AxisIterator;
import matcha.engine.Tensor.TensorIterator;
import matcha.utils.Tensors;

/**
 * Functionals: Loss functions
 * Contains various loss functions for input and target Tensors.
 * @author andrewye
 */
public final class FN_Loss {
    private FN_Loss(){}

    // Default log loss function uses mean reduction.
    public static Tensor log_loss(Tensor input, Tensor target){ return log_loss(input, target, "mean"); }

    /**
     * Computes the (negative) log loss of a input and target distribution.
     * @param input 
     * Tensor. (N,C) where C = # classes, or (N,C,d_1,d_2,d_3,...,d_k) in the case of k-dimensional loss.
     * Input is expected to be (non-log) probabilities. For log probabilities, refer to nll_loss.
     * @param target
     * (N) where each value is between 0 and C-1 or (N, d_1, d_2, d_3,...,d_k) for k-dimensional loss.
     * @param reduction 
     * Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 
     * 'none': no reduction will be applied, 
     * 'mean': the sum of the output will be divided by the number of elements in the output, 
     * 'sum': the output will be summed. 
     * @return
     */
    public static Tensor log_loss(Tensor input, Tensor target, String reduction){
        if (target.m_shape[0] != input.m_shape[0]) 
            throw new IllegalArgumentException("Error: expected target batch size (" + target.m_shape[0] + ") to match input batch size (" + input.m_shape[0] + ")."); 

        switch (reduction){
        case "none":
            return log_loss_none(input, target);
        case "sum":  // calculate summed minus log of target predictions
            return log_loss_sum(input, target);
        case "mean": // calculate mean minus log of target predictions
            return log_loss_mean(input, target);
        default:
            throw new IllegalArgumentException("Error: " + reduction + " is not a valid value for reduction.");
        }
    }

    /**
     * Computes the mean (negative) log loss of a input and target distribution.
     * @param input 
     * Tensor. (N,C) where C = # classes, or (N,C,d_1,d_2,d_3,...,d_k) in the case of k-dimensional loss.
     * Input is expected to be (non-log) probabilities. For log probabilities, refer to nll_loss.
     * @param target
     * (N) where each value is between 0 and C-1 or (N, d_1, d_2, d_3,...,d_k) for k-dimensional loss.
     */
    private static Tensor log_loss_mean(Tensor input, Tensor target){
        // aggregate values for target predictions
        double[] targetPredictions = new double[target.size()];
        List<int[]> samples = getIndicesAlong(input, 1); // iterate across samples.
        TensorIterator itTargets = target.iterator(); // iterate across every target value in order.

        for(int i = 0; i < samples.size(); i++) {
            // for each sample i in the batch, find corresponding value at target idx C i.e. (i,C,d_1,d_2,d_k,,..)
            int[] sampleStart = samples.get(i); 
            double targetIdx = itTargets.next();
            sampleStart[1] = (int) targetIdx;
            targetPredictions[i] = input.get(sampleStart); // add the input value @ target to the total list of predictions
        }

        Tensor t_C;

        if (input.m_gradEnabled){
            List<Tensor> children = new ArrayList<>();
            children.add(input);
            t_C = new Tensor(new int[]{1}, new double[]{mean(m_log(targetPredictions))}, input.m_gradEnabled, children);

            Backward back = () -> {
                List<int[]> samplesBack = getIndicesAlong(input, 1);
                TensorIterator itTargetsBack = target.iterator();
                int lossNorm = target.size(); // normalize by factor of the number of inputs (which includes additional dimensions in the k>2 dimensional case).
                for(int i = 0; i < samples.size(); i++){
                    int[] sampleStart = samplesBack.get(i);
                    double targetIdx = itTargetsBack.next();
                    sampleStart[1] = (int) targetIdx;
                    input.m_grad[input.storageIndex(sampleStart)] += - t_C.m_grad[0] / (input.m_data[input.storageIndex(sampleStart)] * lossNorm);
                }
            };

            t_C.m_backward = back;
            t_C.m_gradFn = GradFunctions.MeanLogLossBackward;

            return t_C;
        } else{
            
            t_C = new Tensor(new int[]{1}, new double[]{mean(m_log(targetPredictions))}, input.m_gradEnabled);

            return t_C;
        }
    }

    /**
     * Computes the summed (negative) log loss of a input and target distribution.
     * @param input 
     * Tensor. (N,C) where C = # classes, or (N,C,d_1,d_2,d_3,...,d_k) in the case of k-dimensional loss.
     * Input is expected to be (non-log) probabilities. For log probabilities, refer to nll_loss.
     * @param target
     * (N) where each value is between 0 and C-1 or (N, d_1, d_2, d_3,...,d_k) for k-dimensional loss.
     */
    private static Tensor log_loss_sum(Tensor input, Tensor target){
        // see log_loss_mean for comments on this portion, since the code is exactly the same and only different for the return value and backpropagation.
        double[] targetPredictions = new double[target.size()];
        List<int[]> samples = getIndicesAlong(input, 1);
        TensorIterator itTargets = target.iterator();

        for(int i = 0; i < samples.size(); i++) {
            int[] sampleStart = samples.get(i);
            double targetIdx = itTargets.next();
            sampleStart[1] = (int) targetIdx;
            targetPredictions[i] = input.get(sampleStart);
        }

        Tensor t_C;

        if (input.m_gradEnabled){
            List<Tensor> children = new ArrayList<>();
            children.add(input);
            t_C = new Tensor(new int[]{1}, new double[]{mean(m_log(targetPredictions))}, input.m_gradEnabled, children);

            Backward back = () -> {
                List<int[]> samplesBack = getIndicesAlong(input, 1);
                TensorIterator itTargetsBack = target.iterator();
                for(int i = 0; i < samples.size(); i++){
                    int[] sampleStart = samplesBack.get(i);
                    double targetIdx = itTargetsBack.next();
                    sampleStart[1] = (int) targetIdx;
                    input.m_grad[input.storageIndex(sampleStart)] += -1.0 / input.m_data[input.storageIndex(sampleStart)] * t_C.m_grad[0];
                }
            };

            t_C.m_backward = back;
            t_C.m_gradFn = GradFunctions.SumLogLossBackward;

            return t_C;
        } else{

            t_C = new Tensor(new int[]{1}, new double[]{sum(m_log(targetPredictions))}, input.m_gradEnabled);
            
            return t_C;
        }
    }

    /**
     * Computes the direct (negative) log loss of a input and target distribution.
     * @param input 
     * Tensor. (N,C) where C = # classes, or (N,C,d_1,d_2,d_3,...,d_k) in the case of k-dimensional loss.
     * Input is expected to be (non-log) probabilities. For log probabilities, refer to nll_loss.
     * @param target
     * (N) where each value is between 0 and C-1 or (N, d_1, d_2, d_3,...,d_k) for k-dimensional loss.
     */
    private static Tensor log_loss_none(Tensor input, Tensor target){
        // see log_loss_mean for comments on this portion, since the code is exactly the same and only different for the return vale and backpropagation.
        double[] targetPredictions = new double[target.size()];
        List<int[]> samples = getIndicesAlong(input, 1);
        TensorIterator itTargets = target.iterator();

        for(int i = 0; i < samples.size(); i++) {
            int[] sampleStart = samples.get(i);
            double targetIdx = itTargets.next();
            sampleStart[1] = (int) targetIdx;
            targetPredictions[i] = input.get(sampleStart);
        }

        Tensor t_C;

        if (input.m_gradEnabled){
            List<Tensor> children = new ArrayList<>();
            children.add(input);

            t_C = new Tensor(target.m_shape, m_log(targetPredictions), input.m_gradEnabled, children);

            Backward back = () -> {
                List<int[]> samplesBack = getIndicesAlong(input, 1);
                TensorIterator itTargetsBack = target.iterator();
                for(int i = 0; i < samples.size(); i++){
                    int[] sampleStart = samplesBack.get(i);
                    double targetIdx = itTargetsBack.next();
                    sampleStart[1] = (int) targetIdx;
                    input.m_grad[input.storageIndex(sampleStart)] += -1.0 / input.m_data[input.storageIndex(sampleStart)] * t_C.m_grad[t_C.storageIndex(itTargetsBack.index())];
                }
            };

            t_C.m_backward = back;
            t_C.m_gradFn = GradFunctions.LogLossBackward;

            return t_C;
        } else{

            t_C = new Tensor(target.m_shape, m_log(targetPredictions), input.m_gradEnabled);
            return t_C;
        }
    }

    public static Tensor mse_loss(Tensor input, Tensor target) { return mse_loss(input, target, "mean"); }

    public static Tensor mse_loss(Tensor input, Tensor target, String reduction){
        if (!(reduction.equals("mean") || reduction.equals("sum"))) 
            throw new IllegalArgumentException("Error: please specify a valid reduction ['mean', 'sum'].");

        if (!Arrays.equals(input.m_shape, target.m_shape)) 
            throw new IllegalArgumentException("Error: input and target tensors must be of the same shape.");
        
        double mse = 0.0;
        for(int i = 0; i < target.m_data.length; i++) mse += ( target.m_data[i] - input.m_data[i] ) * ( target.m_data[i] - input.m_data[i] );

        mse = (reduction.equals("mean")) ? mse / input.size() : mse;

        Tensor t_C;

        if (input.m_gradEnabled || target.m_gradEnabled){
            List<Tensor> children = new ArrayList<>();
            if (input.m_gradEnabled) children.add(input);
            if (target.m_gradEnabled) children.add(target);

            t_C = new Tensor(new int[]{1}, new double[]{mse}, input.m_gradEnabled || target.m_gradEnabled, children);

            Backward back;
            if (reduction.equals("mean")){
                back = () -> {
                    for(int i = 0; i < input.m_data.length; i++){
                        if (input.m_gradEnabled) input.m_grad[i] += -2.0*t_C.m_grad[0]*(target.m_data[i] - input.m_data[i]) / input.size();
                        if (target.m_gradEnabled) target.m_grad[i] += 2.0*t_C.m_grad[0]*(target.m_data[i] - input.m_data[i]) / input.size();
                    }
                };
            } else {
                back = () -> {
                    for(int i = 0; i < input.m_grad.length; i++){
                        if (input.m_gradEnabled) input.m_grad[i] += -2.0*t_C.m_grad[0]*(target.m_data[i] - input.m_data[i]);
                        if (target.m_gradEnabled) target.m_grad[i] += 2.0*t_C.m_grad[0]*(target.m_data[i] - input.m_data[i]);
                    }
                };
            }

            t_C.m_backward = back;
            t_C.m_gradFn = GradFunctions.MSELossBackward;
            return t_C;
        } else {
            t_C = new Tensor(new int[]{1}, new double[]{mse}, input.m_gradEnabled || target.m_gradEnabled);
            return t_C;
        }
    }

    public Tensor cross_entropy(Tensor input, Tensor target){
        throw new UnsupportedOperationException();
    }

    public Tensor nll_loss(Tensor input, Tensor target){
        throw new UnsupportedOperationException();
    }

    private static Tensor cross_entropy_distribution(Tensor input, Tensor target){
        if (!Arrays.equals(input.m_shape, target.m_shape)) throw new IllegalArgumentException("Error: input and targets must be of the same shape for loss with class probabilities.");

        double[] lossPerSample = new double[input.m_shape[0]];
        for(int i = 0; i < lossPerSample.length; i++){
            AxisIterator itInput = input.iterator(new int[]{i, 0}, 1);
            AxisIterator itTarget = target.iterator(new int[]{i, 0}, 1);
            while(itInput.hasNext()){

                lossPerSample[i] += -itTarget.next() * Math.log(itInput.next());
            }

        }

        return new Tensor(new int[]{1}, new double[]{mean(lossPerSample)}, input.m_gradEnabled);    
    }

    private static double[] m_log(double[] data){ return Arrays.stream(data).map(x -> -Math.log(x)).toArray(); }

    private static double mean(double[] data){ return Arrays.stream(data).average().orElse(0); }
    private static double sum(double[] data){ return Arrays.stream(data).sum(); }

    /**
     * Returns the starting points for each slice along an axis.
     * @param t_A, the tensor to call this operation on.
     * @param axis the axis to slice across.
     * @return a list of coordinates denoting the starting points for each slice.
     */
    private static List<int[]> getIndicesAlong(Tensor t_A, int axis){
        if (axis >= t_A.m_shape.length || axis < 0) throw new IllegalArgumentException("Error: axis " + axis + " out of bounds for shape " + t_A.formatShape() + ".");
        LinkedList<int[]> indexList = new LinkedList<>();
        insertIndicesAlong(t_A, indexList, axis, 0, new int[t_A.m_shape.length]);
        return indexList;
    }

    /**
     * Helper function for getIndicesAlong(int axis).
     * @param t_A, the tensor to call this operation on.
     * @param indexList the list of coordinates to return.
     * @param alongAxis the axis to slice along.
     * @param idx the current dimension to iterate over.
     * @param indices the current generated coordinates.
     */
    private static void insertIndicesAlong(Tensor t_A, List<int[]> indexList, int alongAxis, int idx, int[] indices){
        if (idx == t_A.m_shape.length-1){
            if(alongAxis == idx){
                indexList.add(indices);
            } else{
                for(int i=0; i< t_A.m_shape[idx]; i++){
                    int[] tempIndex = indices.clone();
                    tempIndex[idx] = i;
                    indexList.add(tempIndex);
                }
            }
        } else if (idx == alongAxis) {
            insertIndicesAlong(t_A, indexList, alongAxis, idx+1, indices);
        } else {
            for(int i=0; i < t_A.m_shape[idx]; i++){
                int[] tempIndex = indices.clone();
                tempIndex[idx] = i;
                insertIndicesAlong(t_A, indexList, alongAxis, idx+1, tempIndex);
            }
        }
    }
    
}
