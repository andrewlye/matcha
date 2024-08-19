package matcha.engine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import matcha.engine.Tensor.AxisIterator;
import matcha.utils.math.LinAlg;

/**
 * Functionals: Activations
 * Contains various activation functions for Tensors.
 * Used to be part of the Tensor class itself, but moved out for modularity.
 * All methods should be static.
 * @author andrewye
 */
public final class FN_Activations {
    private FN_Activations(){}
    
    /**
     * Calculates the hyperbolic tangent (tanh) of all elements in the tensor.
     * Unary Operation: tanh(A) = B, where A and B are tensors of the same shape and tanh is the hyperbolic tangent function.
     * @param t_A, the tensor to call this operation on.
     * @return t_B, B in tanh(A) = B, where A is the tensor being called.
     */
    public static Tensor tanh(Tensor t_A) {
        double[] dOut = Arrays.stream(t_A.m_data).map(x -> Math.tanh(x)).toArray();
    
        Tensor t_B;
    
        if (t_A.m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(t_A);
    
            t_B = new Tensor(t_A.m_shape, dOut, t_A.m_gradEnabled, children);
            // ddx tanh(x) = 1 - tanh^2(x)
            Backward back = () -> {
                for(int i = 0; i < t_A.m_grad.length; i++){
                    t_A.m_grad[i] += (1-(t_B.m_data[i]*t_B.m_data[i])) * t_B.m_grad[i];
                }
            };
            t_B.m_backward = back;
            t_B.m_gradFn = gradFunctions.TanhBackward;
    
        } else {
            t_B = new Tensor(t_A.m_shape, dOut);
        }
    
        return t_B;
    }

     /**
     * Returns the Rectified Linear Unit (ReLU) activation of this tensor.
     * Unary Operation: relu(A) = B, where A and B are tensors of the same shape.
     * @param t_A, the tensor to call this operation on.
     * @return B in relu(A) = B, where relu(A) denotes the operation max(0, A) and A is the tensor being called.
     */
    public static Tensor relu(Tensor t_A) {
        double[] dOut = Arrays.stream(t_A.m_data).map(x -> Math.max(x, 0)).toArray();
    
        Tensor t_B;
    
        if (t_A.m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(t_A);
    
            t_B = new Tensor(t_A.m_shape, dOut, t_A.m_gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < t_A.m_grad.length; i++){
                    t_A.m_grad[i] += ((t_A.m_data[i] > 0) ? 1 : 0) * t_B.m_grad[i];
                }
            };
            t_B.m_backward = back;
            t_B.m_gradFn = gradFunctions.ReLUBackward;
    
        } else {
            t_B = new Tensor(t_A.m_shape, dOut);
        }
    
        return t_B;
    }

    /**
     * Applies the softmax activation function along a specified axis.
     * Unary operation: softmax(A) = B, where B is the same shape as A.
     * This is a public wrapper for softmaxFunc(int axis) so that an axis of -1 can be used.
     * There is definitely a cleaner way to do this.
     * @param t_A, the tensor to call this operation on.
     * @param axis a dimension along which Softmax will be computed (so every slice along dim will sum to 1). If axis=-1, then the last dim will be used.
     * @return B in softmax(A), where A is the tensor being called.
     */
    public static Tensor softmax(Tensor t_A, int axis){
        if (axis == -1) axis = t_A.m_shape.length - 1;
        return softmaxFunc(t_A, axis);
    }

    /**
     * Applies the softmax activation function along a specified axis.
     * Unary operation: softmax(A) = B, where B is the same shape as A.
     * @param t_A, the tensor to call this operation on.
     * @param axis a dimension along which Softmax will be computed (so every slice along dim will sum to 1).
     * @return B in softmax(A), where A is the tensor being called.
     */
    private static Tensor softmaxFunc(Tensor t_A, int axis){
        // create a psuedo tensor to match elements to their max value along a slice
        double[] maxData = fillDataAlong(t_A, getIndicesAlong(t_A, axis), getMaxAlong(t_A, axis), axis);
        double[] dataOut = t_A.m_data.clone();
        for(int i = 0; i < dataOut.length; i++) dataOut[i] -= maxData[i]; // subtract maximum from each sample as to not explode weights.
        dataOut = Arrays.stream(dataOut).map(x -> Math.exp(x)).toArray(); // exponentiate values.

        // create a psuedo tensor to match each exponentiated element with their sum along a slice.
        double[] expSums = fillDataAlong(t_A, getIndicesAlong(t_A, axis), getSumsAlong(t_A, axis, dataOut), axis);
        for(int i = 0; i < dataOut.length; i++) dataOut[i] /= expSums[i]; // normalize values.
        
        // create output tensor
        Tensor t_B;
        if (t_A.m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(t_A);

            t_B = new Tensor(t_A.m_shape, dataOut, t_A.m_gradEnabled, children);
            Backward back = () -> {
                List<int[]> samples = getIndicesAlong(t_A, axis); // get samples for each softmax slice

                // for each sample, we compute and set its gradient as the sum of its jacobian row
                for(int i = 0; i < samples.size(); i++){
                    int[] sample = samples.get(i);

                    // get softmax output along this sample
                    double[] sampleOutput = new double[t_A.m_shape[axis]];
                    AxisIterator it_B = t_B.iterator(sample.clone(), axis);
                    while(it_B.hasNext()) sampleOutput[it_B.iter()] = it_B.next();

                    // we represent this sample as a column vector
                    Tensor t_sample = new Tensor(new int[]{sampleOutput.length, 1}, sampleOutput, false);

                    // cast sample across an identity matrix
                    Tensor t_diag = LinAlg.diagFlat(sampleOutput);

                    // get pairwise products for each softmax output.
                    Tensor t_dot = t_sample.matmul(t_sample.T());

                    // subtract pairwise products from the diagonal -> jacobian of this sample.
                    Tensor jacobian = t_diag.sub(t_dot);

                    // sum and assign each jacobian row to the correct incoming gradient and outcoming gradient.
                    AxisIterator it_A = t_A.iterator(sample.clone(), axis);
                    while(it_A.hasNext()){
                        it_B = t_B.iterator(sample.clone(), axis);
                        for(int j = 0; j < jacobian.shape()[1]; j++){
                            t_A.m_grad[t_A.storageIndex(it_A.it_pos)] += jacobian.get(new int[]{it_A.it_iter, j}) * t_B.m_grad[t_A.storageIndex(it_B.it_pos)];
                            it_B.next();
                        }
                        it_A.next();
                    }
                }
            };
            t_B.m_backward = back;
            t_B.m_gradFn = gradFunctions.SoftmaxBackward;
        } else {
            t_B = new Tensor(t_A.m_shape, dataOut);
        }

        return t_B;
    }

    /**
     * Fills every slice along an axis with respective data;
     * @param t_A, the tensor to call this operation on.
     * @param partitions starting points for each slice.
     * @param fillData an array of scalars such that the ith slice is filled with the ith element in fillData.
     * @param axis the axis which to fill along.
     * @return a data array of the same representation with each partition slice filled with the specified data along an axis.
     *
     */
    private static double[] fillDataAlong(Tensor t_A, List<int[]> partitions, double[] fillData, int axis){
        if (partitions.size() != fillData.length) throw new IllegalArgumentException("Error: number of elements in data and partition should be equal.");

        double[] data = new double[t_A.m_data.length];
        for(int i = 0; i < partitions.size(); i++){
            AxisIterator it = t_A.iterator(partitions.get(i), axis);
            while (it.hasNext()){
                data[t_A.storageIndex(it.it_pos)] = fillData[i];
                it.next();
            }
        }

        return data;
    }

    /**
     * Returns an array containing the sum of each slice along an axis. Used in softmax backpropagation.
     * @param t_A, the tensor to call this operation on.
     * @param axis the axis to sum along.
     * @param data the data to sum along. Assumed to be stored using the same representation as the calling tensor.
     * @return an array such that the ith element contains the ith slice's sum along the specified axis in data.
     */
    private static double[] getSumsAlong(Tensor t_A, int axis, double[] data){
        List<int[]> indices = getIndicesAlong(t_A, axis);
        double[] sums = new double[indices.size()];
        for(int i = 0; i < sums.length; i++){
            double sum = 0;
            int[] idx = indices.get(i);
            AxisIterator it = t_A.iterator(idx.clone(), axis);
            it.it_data = data;
            while(it.hasNext()) sum += it.next();
            sums[i] = sum;
        }

        return sums;
    }

    /**
     * Returns an array containing the maximum of each slice along an axis. Used in softmax backpropagation.
     * @param t_A, the tensor to call this operation on.
     * @param axis the axis to compute the max along.
     * @param data the data to max along. Assumed to be stored using the same representation as the calling tensor.
     * @return an array such that the ith element contains the ith slice's maximum along the specified axis in data.
     */
    private static double[] getMaxAlong(Tensor t_A, int axis){
        List<int[]> indices = getIndicesAlong(t_A, axis);
        double[] maxes = new double[indices.size()];
        for(int i = 0; i < maxes.length; i++){
            double max = Integer.MIN_VALUE;
            int[] idx = indices.get(i);
            AxisIterator it = t_A.iterator(idx.clone(), axis);
            while(it.hasNext()) max = Math.max(max, it.next());
            maxes[i] = max;
        }
        
        return maxes;
    }

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
