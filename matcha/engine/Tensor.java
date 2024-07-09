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
    private boolean gradEnabled;
    private double[] grad;

    /**
     * Creates a zero-initialized tensor.
     * @shape the shape of the tensor.
     */
    public Tensor(int[] shape) throws Exception{
        // null shape arrays become scalars
        if (shape.length == 0) shape = new int[] {1, 1};
        // Shape dimensions must be greater than 0
        for (int d : shape) if (d < 1) throw new Exception("Error: dimensions must be > 0!");
        // if a single number d is passed, creates a dx1 column vector
        if (shape.length == 1){
            int[] colVec = new int[2];
            colVec[0] = shape[0];
            colVec[1] = 1;
            shape = colVec;
        }

        this.shape = shape;

        int numElements = 1;
        for (int i = 0; i < shape.length; i++)
            numElements *= shape[i];
        this.data = new double[numElements];
    }

    public Tensor(int[] shape, double[] data) {
        this.shape = shape;
        this.data = data;
        this.gradEnabled = false;
    }

    public Tensor matmul(Tensor t) throws Exception{
        if (t.shape.length == 2){
            if (this.shape[1] != t.shape[0]){
                throw new Exception("Error: dimensions " + this.formatShape() + " and " + t.formatShape() + " are invalid for this operation.");            
            }
            
            double C[] = new double[this.shape[0] * t.shape[1]];
            int[] CShape = new int[] {this.shape[0], t.shape[1]};

            for(int r = 0; r < this.shape[0]; r++){
                for(int c = 0; c < t.shape[1]; c++){
                    for(int k = 0; k < t.shape[0]; k++){
                        C[LinAlg.rmo(C.length, CShape, new int[]{r, c})] += data[LinAlg.rmo(data.length, shape, new int[]{r, k})] * t.data[LinAlg.rmo(t.data.length, t.shape, new int[]{k, c})];
                    }
                }
            }

            return new Tensor(CShape, C);
        }

        return new Tensor(new int[3]);
    }

    public double get(int[] idxs) throws Exception {
        return data[LinAlg.rmo(data.length, shape, idxs)];
    }

    public void set(int[] idxs, double x) throws Exception {
        data[LinAlg.rmo(data.length, shape, idxs)] = x;
    }

    public void withGrad(){
        gradEnabled = true;
    }

    public void noGrad(){
        gradEnabled = false;
    }

    public double[] data() {
        return data;
    }

    public boolean gradEnabled() {
        return gradEnabled;
    }

    public int[] shape() {
        return shape;
    }

    @Override
    public String toString() {
        return "[ Tensor of shape: " + formatShape() + ", grad enabled=" + gradEnabled + " ]";
    }

    public String formatShape(){
        StringBuilder s = new StringBuilder("(");
        for (int i = 0;  i < shape.length; i++){
            s.append(shape[i]);
            if (i != shape.length - 1) s.append(", ");
        }
        s.append(")");
        return s.toString();
    }
}
