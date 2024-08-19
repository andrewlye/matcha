package matcha.utils;

import java.util.Arrays;

import matcha.engine.DataRepresentation;
import matcha.engine.Tensor;
import matcha.utils.math.LinAlg;


/**
 * utils/Tensors - contains static methods for viewing/visualizing tensors.
 * @author andrewye
 */
public class Tensors {
    /**
     * Returns a string-representation of a Tensor object
     * @param t the tensor object to read
     * @return a string-representation of the tensor
     */
    public static String toString(Tensor t){
        StringBuilder sb = new StringBuilder();
        sb.append(t.toString() + '\n');
        sb.append(toString(t.data(), t.shape(), new int[t.shape().length], 0, new StringBuilder(), t.dataLayout));

        return sb.toString();
    }

    public static String showGrad(Tensor t) {
        return toString(t.grad(), t.shape(), new int[t.shape().length], 0, new StringBuilder(), t.dataLayout);
    }


    /**
     * Returns the string-representation of an array parameterized by a shape
     * @param data the data to read
     * @param shape the shape that specified the data, row-major order is assumed
     * @param idxs for higher (>2) dimensional data, fixes dimensions to print their components
     * @param d the current dimension to fix
     * @param sb StringBuilder
     * @return
     */
    private static String toString(double[] data, int[] shape, int[] idxs, int d, StringBuilder sb, DataRepresentation data_layout){
        if (d == shape.length - 2){
            sb.append("[");
            idxs[d+1] = -1;
            for (int i = 0; i < shape[d]*shape[d+1]; i++){
                idxs[d+1]++;
                int j = d+1;
                if(idxs[j] > shape[j] - 1){
                    idxs[j--] = 0;
                    idxs[j]++;
                    sb.append("\n");
                    for(int sp = 0; sp <= d; sp++){
                        sb.append(" ");
                    }
                }
                
                switch(data_layout){
                case ROW_MAJOR:
                default: sb.append(data[LinAlg.rmo(shape, idxs)]);
                }
                
                if (i != shape[d] * shape[d+1] - 1)
                    sb.append(", ");
            }
            sb.append("]");
            return sb.toString();
        }
        else {
            sb.append("[");
            for(int i = 0; i < shape[d]; i++){
                int[] new_idxs = idxs.clone();
                new_idxs[d] = i;
                sb.append(toString(data, shape, new_idxs, d+1, new StringBuilder(), data_layout));
                if (i != shape[d] - 1){
                    sb.append(",\n");
                    for(int sp = 0; sp <= d; sp++){
                        sb.append(" ");
                    }
                }
            }
            sb.append("]");
            return sb.toString();
        }
    }
}
