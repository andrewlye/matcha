package matcha.utils.math;

import matcha.engine.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * LinAlg - implements several useful algebraic manipulations and functions on matrices and tensors.
 * @author  andrewye
 */
public class LinAlg {

    public static Tensor eye(int n, int m){
        return eye(new int[]{n,m}, false);
    }

    public static Tensor eye(int[] shape, boolean gradEnabled){
        return diag(shape, 1, gradEnabled);
    }

    public static Tensor diagFlat(double[] diag){
        return diagFlat(diag, false);
    }

    public static Tensor diagFlat(double[] diag, boolean gradEnabled){
        return diag(new int[]{diag.length, diag.length}, diag, gradEnabled);
    }

    public static Tensor diag(int[] shape, double dVal, boolean gradEnabled){
        double[] data = new double[sizeOf(shape)];
        int[] diagIndex = new int[shape.length];
        for(int i = 0; i < min(shape); i++){
            diagIndex = fill(diagIndex, i);
            data[rmo(shape, diagIndex)] = dVal;
        }

        return new Tensor(shape, data, gradEnabled);
    }

    public static Tensor diag(int[] shape, double[] dVal, boolean gradEnabled){
        if (dVal.length < min(shape)) throw new IllegalArgumentException("Error: diagonal values passed must be at least equal in length to the smallest dimension.");

        double[] data = new double[sizeOf(shape)];
        int[] diagIndex = new int[shape.length];
        for(int i = 0; i < min(shape); i++){
            diagIndex = fill(diagIndex, i);
            data[rmo(shape, diagIndex)] = dVal[i];
        }

        return new Tensor(shape, data, gradEnabled);
    }


    private static int[] fill(int[] in, int n){
        return Arrays.stream(in).map(x -> n).toArray();
    }

    private static int min(int[] in){
        return Arrays.stream(in).min().orElse(0);
    }

    private static int sizeOf(int[] shape){
        int numElements = 1;
        for (int i = 0; i < shape.length; i++)
            numElements *= shape[i];
        
        return numElements;
    }
    

    /**
     * Returns the Row-Major index of an element in a list of elements, parametrized by the given shape and in-shape index.
     * For more information, see https://en.wikipedia.org/wiki/Row-_and_column-major_order#
     * @param shape the shape in which to parametrize the elements.
     * @param idxs the shape-indices of the element to retrieve.
     * @return The data index of the element associated at the shape index in row-major order.
     */
    public static int rmo(int[] shape, int[] idxs){
        if (shape.length != idxs.length) {
            throw new IllegalArgumentException("Error: number of indexes must match dimensionality of object!");
        }

        for (int i = 0; i < idxs.length; i++) {
            if (idxs[i] < 0 || idxs[i] >= shape[i]) {
                System.out.println("Warning: indices are inconsistent with the shape of the data.");
            }
        }

        int n_i = 0;
        int N_i = 1;
        int rmo = idxs[n_i];

        while (N_i < shape.length) {
            rmo = shape[N_i++] * rmo + idxs[++n_i];
        }

        return rmo;
    }
    
    /**
     * Returns the dimensions of an array (or nested array) of objects.
     * @param the object to retrieve the shape of.
     * @return shape of the object.
     */
    public static int[] getDims(Object v){
        return getDims(v, new ArrayList<Integer>());
    }

    private static int[] getDims(Object v, List<Integer> dims) {
        if (v instanceof Object[]) {
            Object[] v_arr = (Object[]) v;
            dims.add(v_arr.length);
            return (v_arr.length == 0) ? dims.stream().mapToInt(x -> x).toArray() : getDims(v_arr[0], dims);
        }
        if (v instanceof byte[]) {
            byte[] v_arr = (byte[]) v;
            dims.add(v_arr.length);
            return (v_arr.length == 0) ? dims.stream().mapToInt(x -> x).toArray() : getDims(v_arr[0], dims);
        }
        if (v instanceof short[]) {
            short[] v_arr = (short[]) v;
            dims.add(v_arr.length);
            return (v_arr.length == 0) ? dims.stream().mapToInt(x -> x).toArray() : getDims(v_arr[0], dims);
        }
        if (v instanceof int[]) {
            int[] v_arr = (int[]) v;
            dims.add(v_arr.length);
            return (v_arr.length == 0) ? dims.stream().mapToInt(x -> x).toArray() : getDims(v_arr[0], dims);
        }
        if (v instanceof long[]) {
            long[] v_arr = (long[]) v;
            dims.add(v_arr.length);
            return (v_arr.length == 0) ? dims.stream().mapToInt(x -> x).toArray() : getDims(v_arr[0], dims);
        }
        if (v instanceof float[]) {
            float[] v_arr = (float[]) v;
            dims.add(v_arr.length);
            return (v_arr.length == 0) ? dims.stream().mapToInt(x -> x).toArray() : getDims(v_arr[0], dims);
        }
        if (v instanceof double[]) {
            double[] v_arr = (double[]) v;
            dims.add(v_arr.length);
            return (v_arr.length == 0) ? dims.stream().mapToInt(x -> x).toArray() : getDims(v_arr[0], dims);
        }
        if (v instanceof boolean[]) {
            boolean[] v_arr = (boolean[]) v;
            dims.add(v_arr.length);
            return (v_arr.length == 0) ? dims.stream().mapToInt(x -> x).toArray() : getDims(v_arr[0], dims);
        }
        if (v instanceof char[]) {
            char[] v_arr = (char[]) v;
            dims.add(v_arr.length);
            return (v_arr.length == 0) ? dims.stream().mapToInt(x -> x).toArray() : getDims(v_arr[0], dims);
        }

        return dims.stream().mapToInt(x -> x).toArray();
    }

}
