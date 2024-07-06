package matcha.engine;

import matcha.utils.math.LinAlg;
import java.util.Arrays;

/**
 * Tensor - in accordance with its more primitive predecessor Value, the core data structure behind matcha operations.
 * @author andrewye
 */
public class Tensor {
    private int[] shape;
    private double[] data; // data is stored as a 1-d array in memory, with shapes being row-major indexed.

    /**
     * Creates a zero-initialized tensor.
     * @shape the shape of the tensor.
     */
    public Tensor(int[] shape) {
        this.shape = shape;

        int numElements = 1;
        for (int i = 0; i < shape.length; i++)
            numElements *= shape[i];
        this.data = new double[numElements];
    }

    public Tensor(Object d){
        this.shape = LinAlg.getDims(d);
    }

    public double get(int[] idxs) throws Exception {
        return data[LinAlg.indexRowMajor(data, shape, idxs)];
    }

    public void set(int[] idxs, double x) throws Exception {
        data[LinAlg.indexRowMajor(data, shape, idxs)] = x;
    }

    public double[] data() {
        return data;
    }

    public int[] shape() {
        return shape;
    }

}
