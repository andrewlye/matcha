package matcha.engine;

import matcha.utils.math.LinAlg;

import java.security.spec.ECFieldF2m;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.DoubleStream;

/**
 * Tensor - in accordance with its more primitive predecessor Value, the core data structure behind matcha operations.
 * @author andrewye
 */
public class Tensor {
    private int[] shape; // shape of tensor
    // data is stored as a 1-d array in memory, with shapes being row-major indexed. See https://pytorch.org/docs/stable/storage.html
    private double[] data;
    private boolean gradEnabled;
    private double[] grad;

    private List<Tensor> prev;
    private Backward backward;

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

    public Tensor(int[] shape, double[] data) throws Exception{
        // null shape arrays become scalars
        if (shape.length == 0) shape = new int[] {1, 1};
        // Shape dimensions must be greater than 0
        for (int d : shape) if (d < 1) throw new Exception("Error: dimensions must be > 0!");
        // number of elements must be consistent with dimension of shape
        int numElements = 1;
        for (int i = 0; i < shape.length; i++)
            numElements *= shape[i];
        if (numElements != data.length) throw new Exception("Error: number of elements specified by dimensions (" + numElements +  ") are inconsistent with the length of data provided (" + data.length + ").");

        // if a single number d is passed, creates a dx1 column vector
        if (shape.length == 1){
            int[] colVec = new int[2];
            colVec[0] = shape[0];
            colVec[1] = 1;
            shape = colVec;
        }

        this.shape = shape;
        this.data = data;
        this.gradEnabled = false;
    }

    public Tensor(int[] shape, double[] data, boolean gradEnabled) throws Exception{
        // null shape arrays become scalars
        if (shape.length == 0) shape = new int[] {1, 1};
        // Shape dimensions must be greater than 0
        for (int d : shape) if (d < 1) throw new Exception("Error: dimensions must be > 0!");
        // number of elements must be consistent with dimension of shape
        int numElements = 1;
        for (int i = 0; i < shape.length; i++)
            numElements *= shape[i];
        if (numElements != data.length) throw new Exception("Error: number of elements specified by dimensions (" + numElements +  ") are inconsistent with the length of data provided (" + data.length + ").");

        // if a single number d is passed, creates a dx1 column vector
        if (shape.length == 1){
            int[] colVec = new int[2];
            colVec[0] = shape[0];
            colVec[1] = 1;
            shape = colVec;
        }

        this.shape = shape;
        this.data = data;
        
        withGrad(gradEnabled);
    }

    private Tensor(int[] shape, double[] data, boolean gradEnabled, List<Tensor> children) throws Exception{
        // null shape arrays become scalars
        if (shape.length == 0) shape = new int[] {1, 1};
        // Shape dimensions must be greater than 0
        for (int d : shape) if (d < 1) throw new Exception("Error: dimensions must be > 0!");
        // number of elements must be consistent with dimension of shape
        int numElements = 1;
        for (int i = 0; i < shape.length; i++)
            numElements *= shape[i];
        if (numElements != data.length) throw new Exception("Error: number of elements specified by dimensions (" + numElements +  ") are inconsistent with the length of data provided (" + data.length + ").");
        
        // if a single number d is passed, creates a dx1 column vector
        if (shape.length == 1){
            int[] colVec = new int[2];
            colVec[0] = shape[0];
            colVec[1] = 1;
            shape = colVec;
        }

        this.shape = shape;
        this.data = data;
        this.prev = children;
        
        withGrad(gradEnabled);
    }

    public Tensor add(Tensor t) throws Exception{
        if (!Arrays.equals(this.shape, t.shape)){
            //TO-DO: broadcasting. See https://numpy.org/doc/stable/user/basics.broadcasting.html.
            throw new Exception("Error: shapes " + this.formatShape() + " and " + t.formatShape() + " are not compatible for this operation.");
        }
        
        int numElements = 1;
        for (int i = 0; i < shape.length; i++)
            numElements *= shape[i];
        double dOut[] = new double[numElements];
        for (int i = 0; i < dOut.length; i++){
            dOut[i] = this.data[i] + t.data[i];
        }

        Tensor tOut;
        
        if (gradEnabled || t.gradEnabled){
            List<Tensor> children = new ArrayList<>();
            children.add(this);
            children.add(t);

            tOut = new Tensor(this.shape, dOut, this.gradEnabled || t.gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < grad.length; i++){
                    if (this.gradEnabled) this.grad[i] += 1.0 * tOut.grad[i];
                    if (t.gradEnabled) t.grad[i] += 1.0 * tOut.grad[i];
                }
            };
            tOut.backward = back;

        } else {
            tOut = new Tensor(this.shape, dOut);
        }

        return tOut;
    }

    public Tensor sub(Tensor t) throws Exception {
        return this.add(t.mul(-1.0));
    }

    public Tensor mul(double C) throws Exception{
        double[] dOut = Arrays.stream(data).map(x -> C * x).toArray();
        Tensor tOut;

        if (gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);

            tOut = new Tensor(shape, dOut, gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < grad.length; i++){
                    this.grad[i] += C * tOut.grad[i];
                }
            };
            tOut.backward = back;
        } else {
            tOut = new Tensor(shape, dOut);
        }

        return tOut;
    }

    public Tensor mul(Tensor t) throws Exception {
        if (!Arrays.equals(this.shape, t.shape)){
            //TO-DO: broadcasting. See https://numpy.org/doc/stable/user/basics.broadcasting.html.
            throw new Exception("Error: shapes " + this.formatShape() + " and " + t.formatShape() + " are not compatible for this operation.");
        }

        double[] dOut = data.clone();
        for (int i = 0; i < data.length; i++){
            dOut[i] *= t.data[i];
        }

        Tensor tOut;

        if (this.gradEnabled || t.gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);
            children.add(t);  
            tOut = new Tensor(shape, dOut, gradEnabled || t.gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < grad.length; i++){
                    if (this.gradEnabled) this.grad[i] += t.data[i] * tOut.grad[i];
                    if (t.gradEnabled) t.grad[i] += this.data[i] * tOut.grad[i];
                }
            };
            tOut.backward = back;
        }
        else {
            tOut = new Tensor(shape, dOut);
        }

        return tOut;
    }

    public Tensor div(double C) throws Exception{
        return mul(1/C);
    }

    public Tensor div(Tensor t) throws Exception{
        return mul(t.pow(-1));
    }

    public Tensor pow(double C) throws Exception{
        double[] dOut = Arrays.stream(data).map(x -> Math.pow(x,C)).toArray();
        Tensor tOut;

        if (gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);

            tOut = new Tensor(shape, dOut, gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < grad.length; i++){
                    this.grad[i] += C * Math.pow(this.data[i], C-1) * tOut.grad[i];
                }
            };
            tOut.backward = back;
        } else {
            tOut = new Tensor(shape, dOut);
        }

        return tOut;
    }

    public Tensor pow(Tensor t) throws Exception{
        if (!Arrays.equals(this.shape, t.shape)){
            throw new Exception("Error: shapes " + this.formatShape() + " and " + t.formatShape() + " are not compatible for this operation.");
        }

        double[] dOut = new double[data.length];
        for(int i = 0; i < data.length; i++){
            dOut[i] = Math.pow(data[i], t.data[i]);
        }
        
        Tensor tOut;

        if (gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);
            children.add(t);

            tOut = new Tensor(shape, dOut, gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < grad.length; i++){
                    this.grad[i] += t.data[i] * Math.pow(this.data[i], t.data[i]-1) * tOut.grad[i];
                    t.grad[i] += Math.pow(this.data[i], t.data[i]) * Math.log(t.data[i]) * tOut.grad[i];
                }
            };
            tOut.backward = back;

        } else {
            tOut = new Tensor(shape, dOut);
        }

        return tOut;
    }

    public Tensor exp() throws Exception {
        double[] dOut = Arrays.stream(data).map(x -> Math.exp(x)).toArray();

        Tensor tOut;

        if (gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);

            tOut = new Tensor(shape, dOut, gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < grad.length; i++){
                    this.grad[i] += tOut.data[i] * tOut.grad[i];
                }
            };
            tOut.backward = back;

        } else {
            tOut = new Tensor(shape, dOut);
        }

        return tOut;
    }

    public Tensor tanh() throws Exception {
        double[] dOut = Arrays.stream(data).map(x -> Math.tanh(x)).toArray();

        Tensor tOut;

        if (gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);

            tOut = new Tensor(shape, dOut, gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < grad.length; i++){
                    this.grad[i] += (1-(tOut.data[i]*tOut.data[i])) * tOut.grad[i];
                }
            };
            tOut.backward = back;

        } else {
            tOut = new Tensor(shape, dOut);
        }

        return tOut;
    }

    public Tensor relu() throws Exception {
        double[] dOut = Arrays.stream(data).map(x -> Math.max(x, 0)).toArray();

        Tensor tOut;

        if (gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);

            tOut = new Tensor(shape, dOut, gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < grad.length; i++){
                    this.grad[i] += ((this.data[i] > 0) ? 1 : 0) * tOut.grad[i];
                }
            };
            tOut.backward = back;

        } else {
            tOut = new Tensor(shape, dOut);
        }

        return tOut;
    }

    /**
     * Performs conventional matrix multiplication is both arguments are 2D.
     * TO-DO: Add N-D matmul
     * @param t the tensor to multiply
     * @return this @ t
     * @throws Exception in the case of dimension mismatches
     */
    public Tensor matmul(Tensor t) throws Exception{
        if (t.shape.length == 2 && this.shape.length == 2){
            if (this.shape[1] != t.shape[0]){
                throw new Exception("Error: dimensions " + this.formatShape() + " and " + t.formatShape() + " are invalid for this operation.");            
            }
            
            double dataOut[] = new double[this.shape[0] * t.shape[1]];
            int[] shapeOut = new int[] {this.shape[0], t.shape[1]};

            for(int r = 0; r < this.shape[0]; r++){
                for(int c = 0; c < t.shape[1]; c++){
                    for(int k = 0; k < t.shape[0]; k++){
                        dataOut[LinAlg.rmo(dataOut.length, shapeOut, new int[]{r, c})] += data[LinAlg.rmo(data.length, shape, new int[]{r, k})] * t.data[LinAlg.rmo(t.data.length, t.shape, new int[]{k, c})];
                    }
                }
            }


            Tensor tOut;
            if (gradEnabled || t.gradEnabled){
                List<Tensor> children = new ArrayList<>();
                children.add(this);
                children.add(t);
                
                tOut = new Tensor(shapeOut, dataOut, this.gradEnabled || t.gradEnabled, children);

                Backward back = () -> {
                    for(int r = 0; r < this.shape[0]; r++){
                        for(int c = 0; c < t.shape[1]; c++){
                            for(int k = 0; k < t.shape[0]; k++){
                                try {
                                    this.grad[LinAlg.rmo(data.length, shape, new int[]{r, k})] += t.data[LinAlg.rmo(t.data.length, t.shape, new int[]{k, c})] * tOut.grad[LinAlg.rmo(dataOut.length, shapeOut, new int[]{r, c})];
                                    t.grad[LinAlg.rmo(t.data.length, t.shape, new int[]{k, c})] += this.data[LinAlg.rmo(data.length, shape, new int[]{r, k})] * tOut.grad[LinAlg.rmo(dataOut.length, shapeOut, new int[]{r, c})];
                                } catch (Exception e) {
                                    e.printStackTrace();
                                }
                            }
                        }
                    }
                };

                tOut.backward = back;

            } else {
                tOut = new Tensor(shapeOut, dataOut);
            }

            return tOut;
        }
        else throw new Exception("Error: N-d (N>2) tensor multiplication not supported yet!");
    }

    public void step(double step_size) {
        for(int i = 0; i < data.length; i++)
            data[i] += step_size*grad[i];
    }

    public void backward() throws Exception{
        if (!gradEnabled)
            throw new Exception("Error: calling backprop on non grad-enabled Tensor.");

        List<Tensor> ordering = new ArrayList<>();
        buildTopo(this, new HashSet<>(), ordering);
        Collections.reverse(ordering);

        for (int i = 0; i < grad.length; i++){
            grad[i] = 1.0;
        }

        for(Tensor val : ordering){
            if (!val.gradEnabled)
                System.out.println("Warning: some tensors encountered in backprop have gradients disabled.");
            else
            val.backward.pass();
        }
    }

    private void buildTopo(Tensor parent, Set<Tensor> visited, List<Tensor> ordering) {
        if (!visited.contains(parent)) {
            visited.add(parent);
            if (parent.prev != null) {
                for(Tensor child : parent.prev){
                    buildTopo(child, visited, ordering);
                }
            }
            ordering.add(parent);
        }
    }

    public double[] data() {
        return data;
    }

    public double[] grad() {
        return grad;
    }

    public boolean gradEnabled() {
        return gradEnabled;
    }

    public int[] shape() {
        return shape;
    }

    public double get(int[] idxs) throws Exception {
        return data[LinAlg.rmo(data.length, shape, idxs)];
    }

    public void set(int[] idxs, double x) throws Exception {
        data[LinAlg.rmo(data.length, shape, idxs)] = x;
    }

    public void setGradient(double[] grad) {
        if (gradEnabled && grad.length == data.length)
            this.grad = grad;
    }

    public void zeroGrad() {
        if (gradEnabled)
            grad = new double[grad.length];
    }

    public void withGrad(boolean g){
        gradEnabled = g;
        if (gradEnabled){
            grad = new double[data.length];
            backward = () -> {};
        } else {
            if (grad != null) grad = null;
        }
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
