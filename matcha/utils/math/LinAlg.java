package matcha.utils.math;

import matcha.engine.*;
import java.util.List;
import java.util.ArrayList;

/**
 * LinAlg - implements several useful algebraic manipulations and functions on matrices and tensors.
 * @author  andrewye
 */
public class LinAlg {
    /** 
     * Returns a standard square identity matrix.
     * @param dim the number of rows (or cols).
     * @return An identity matrix of dimensions dim*dim.
     */
    public static double[][] eye(int dim) {
        double[][] I = new double[dim][dim];

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if (i == j)
                    I[i][j] = 1.0;
            }
        }

        return I;
    }

    /**
     * Returns the Row-Major index of an element in a list of elements, parametrized by the given shape and in-shape index.
     * For more information, see https://en.wikipedia.org/wiki/Row-_and_column-major_order#
     * @param elements the list of elements.
     * @param shape the shape in which to parametrize the elements.
     * @param idxs the shape-indices of the element to retrieve.
     * @return The index of the list element associated at the shape index in row-major order.
     */
    public static int rmo(int elements, int[] shape, int[] idxs){
        if (shape.length != idxs.length) {
            throw new IllegalArgumentException("Error: number of indexes must match dimensionality of object!");
        } 

        int elementsInShape = 1;
        for(int i = 0; i < shape.length; i++) elementsInShape *= shape[i];
        if (elements != elementsInShape){
            throw new IllegalArgumentException("Error: the number of elements specified by the shape does not match the data.");
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
     * Returns a 2-D square matrix with the elements of input as the diagonal.
     * @param v the input elements to be diagonalized.
     * @return diagonalized matrix of the input vector.
     */
    public static double[][] diagFlat(Value[] v) {
        double[][] A = new double[v.length][v.length];

        for (int i = 0; i < v.length; i++) {
            for (int j = 0; j < v.length; j++) {
                if (i == j)
                    A[i][j] = v[i].data();
            }
        }

        return A;
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
