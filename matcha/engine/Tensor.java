package matcha.engine;

import matcha.engine.threads.matMulThread;
import matcha.utils.Tensors;
import matcha.utils.math.LinAlg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
 * The Tensor class holds the core structure and manipulation methods behind data operations in matcha.
 * @author andrewye
 */
public class Tensor implements Iterable<Double>{
    protected int[] m_shape; // shape of tensor
    // is true if gradients need to be computed for this tensor.
    protected boolean m_gradEnabled;
    // data is stored as a 1-d array in memory, with shapes being row-major indexed. See https://pytorch.org/docs/stable/storage.html
    protected double[] m_data;
    protected double[] m_grad;

    protected List<Tensor> m_prev;
    protected Backward m_backward; // backprop handling for this tensor.
    protected GradFunctions m_gradFn = GradFunctions.None;

    public DataRepresentation dataLayout = DataRepresentation.ROW_MAJOR;

    /**
     * Tensor constructor.
     * @param shape the shape of the tensor (e.g. [3, 5, 2] denotes a 3x5x2 tensor).
     * @param data the data held by the tensor, zero-initialized if null.
     * @param gradEnabled boolean indicating if auto differentiation is enabled. False by default.
     */
    public Tensor(int[] shape, double[] data, boolean gradEnabled){
        // if shape is null, tensor becomes a scalar (1x1)
        if (shape.length == 0) shape = new int[] {1, 1};
        
        // shape dimensions must be greater than 0
        for (int d : shape) if (d < 1) throw new IllegalArgumentException("Error: dimensions must be > 0!");
        
        // the number of elements in data must be consistent with dimension of shape
        int numElements = 1;
        for (int i = 0; i < shape.length; i++)
            numElements *= shape[i];
        if (data != null && numElements != data.length) 
            throw new IllegalArgumentException("Error: number of elements specified by dimensions (" + numElements +  ") are inconsistent with the length of data provided (" + data.length + ").");

        this.m_shape = shape;
        this.m_data = (data != null) ? data : new double[numElements];
        withGrad(gradEnabled);
    }

    // Creates a zero-initialized tensor of a specified shape.
    public Tensor(int... shape){
        this(shape, null, false);
    }

    // Creates a zero-initialized tensor of specified shape and data.
    public Tensor(int[] shape, double[] data){
        this(shape, data, false);
    }

    // Creates a zero-initialized tensor of a specified shape with or without differentiation.
    public Tensor(int[] shape, boolean gradEnabled){
       this(shape, null, true);
    }

    public Tensor(Object o) {
        this(o, false);
    }

    public Tensor(Object o, boolean gradEnabled) {
        this(LinAlg.getDims(o), LinAlg.flatten(o), gradEnabled);
    }

    // Private constructor used for auto differentiation and chain-ruling.
    protected Tensor(int[] shape, double[] data, boolean gradEnabled, List<Tensor> children){
        this(shape, data, gradEnabled);
        this.m_prev = children;
    }

    // --------------------------
    //    UNARY OPERATIONS
    // -------------------------

    /**
     * Transposes a N-dimensional tensor using a list of axes for the dimension mapping.
     * @param axes a permutation (0, 1, 2, ..., N-1) of the shape indices for the transposed tensor.
     * @return The transpose of the tensor being called along the specified axes.
     */
    public Tensor T(int[] axes){
        if (axes.length != m_shape.length) throw new IllegalArgumentException("Error: axes must contain a permutation of the tensor shape.");
        double[] dataOut = new double[m_data.length];
        int[] shapeOut = remap(m_shape, axes); // remap shape according to transpose axes
        Tensor t_B = new Tensor(shapeOut, m_gradEnabled);
        t_B.dataLayout = dataLayout; // make sure the transpose shares the same representation format

        TensorIterator it = this.iterator();
        while(it.hasNext()){ // cycle through and map all indices in the current tensor to their transpose.
            double n = it.next();
            dataOut[t_B.storageIndex(remap(it.it_pos, axes))] = n;
        }

        t_B.setData(dataOut);

        return t_B;
    }

    // Transposes a 2-dimensional tensor.
    public Tensor T(){
        if (m_shape.length != 2) throw new IllegalCallerException("Error: transpose with no arguments can only be called by 2-D tensors.");

        return T(new int[]{1,0});
    }

    /**
     * Multiplies a tensor by a scalar x.
     * Unary operation: xA = B, where A and B are tensors of the same shape and x is a scalar.
     * @param sc_x, x in xA = B, where A is the tensor being called and x is a scalar.
     * @return B in xA = B.
     */
    public Tensor mul(double sc_x){
        double[] dOut = Arrays.stream(m_data).map(x -> sc_x * x).toArray();
        Tensor t_B;

        if (m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);

            t_B = new Tensor(m_shape, dOut, m_gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < m_grad.length; i++){
                    this.m_grad[i] += sc_x * t_B.m_grad[i];
                }
            };
            t_B.m_backward = back;
            t_B.m_gradFn = GradFunctions.ScalarMulBackward;
        } else {
            t_B = new Tensor(m_shape, dOut);
        }

        return t_B;
    }

    /**
     * Divides this tensor element-wise by a non-zero scalar x.
     * Unary operation: A/x = B, where A and B are tensors and x is a scalar.
     * @param sc_x, x in A/x = B, where A is the tensor being called.
     * @return B in A/x= B, where B contains the elements of A divided by x.
     * TODO: maybe add exception for division by 0.
     */
    public Tensor div(double sc_x){
        return mul(1/sc_x);
    }

    /**
     * Raises the elements of a tensor to a scalar x.
     * Unary operation: A^x = B, where A and B are tensors of the same shape and x is a scalar.
     * @param sc_x, x in A^x = B, where A is the tensor being called.
     * @return tensor B in A^x = B, where B contains the elements in A raised to the x-th power.
     */
    public Tensor pow(double sc_x){
        double[] dOut = Arrays.stream(m_data).map(x -> Math.pow(x,sc_x)).toArray();
        Tensor t_B;
    
        if (m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);
    
            t_B = new Tensor(m_shape, dOut, m_gradEnabled, children);
            // d/dx x^n = n*x^(n-1)
            Backward back = () -> {
                for(int i = 0; i < m_grad.length; i++){
                    this.m_grad[i] += sc_x * Math.pow(this.m_data[i], sc_x-1) * t_B.m_grad[i];
                }
            };
            t_B.m_backward = back;
            t_B.m_gradFn = GradFunctions.ScalarPowBackward;
        } else {
            t_B = new Tensor(m_shape, dOut);
        }
    
        return t_B;
    }

    /**
     * Calculate the exponential of all elements in the tensor
     * Unary Operation: e^A = B, where A and B are tensors of the same shape and e is Euler's number (2.718281...)
     * @return t_B, B in e^A = B, where B contains the exponentials of the elements in A and A is the tensor being called.
     */
    public Tensor exp() {
        double[] dOut = Arrays.stream(m_data).map(x -> Math.exp(x)).toArray();
    
        Tensor t_B;
    
        if (m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);
    
            t_B = new Tensor(m_shape, dOut, m_gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < m_grad.length; i++){
                    this.m_grad[i] += t_B.m_data[i] * t_B.m_grad[i];
                }
            };
            t_B.m_backward = back;
            t_B.m_gradFn = GradFunctions.ExpBackward;
    
        } else {
            t_B = new Tensor(m_shape, dOut);
        }
    
        return t_B;
    }

    public double max() { return Arrays.stream(m_data).max().getAsDouble(); }

    /**
     * Returns the maximimums along a given axis of the tensor.
     * Remark: gradients not supported yet.
     */
    public Tensor max(int axis){
        if (axis == -1) axis = m_shape.length-1;
        if (axis < 0 || axis >= m_shape.length) 
            throw new IllegalArgumentException(
                String.format("Error: axis %d out of bounds for shape %s", axis, formatShape())
            );

        double[] maxes = FN_Activations.getMaxAlong(this, axis);
        int[] max_shape = new int[m_shape.length-1];
        for (int i = 0, j = 0; j < m_shape.length; j++) {
            if (j == axis) continue;
            max_shape[i++] = m_shape[j];
        }

        return new Tensor(max_shape, maxes, false);
    }

    public double min() { return Arrays.stream(m_data).min().getAsDouble(); }

    /**
     * Returns the minimums along a given axis of the tensor.
     * Remark: gradients not supported yet.
     */
    public Tensor min(int axis){
        if (axis == -1) axis = m_shape.length-1;
        if (axis < 0 || axis >= m_shape.length) 
            throw new IllegalArgumentException(
                String.format("Error: axis %d out of bounds for shape %s", axis, formatShape())
            );

        double[] mins = FN_Activations.getMinAlong(this, axis);
        int[] min_shape = new int[m_shape.length-1];
        for (int i = 0, j = 0; j < m_shape.length; j++) {
            if (j == axis) continue;
            min_shape[i++] = m_shape[j];
        }

        return new Tensor(min_shape, mins, false);
    }

    /**
     * Returns the indices of the maximimal elements along a given axis of the tensor.
     * Remark: gradients not supported yet.
     */
    public Tensor argmax(int axis){
        if (axis == -1) axis = m_shape.length-1;
        if (axis < 0 || axis >= m_shape.length) 
            throw new IllegalArgumentException(
                String.format("Error: axis %d out of bounds for shape %s", axis, formatShape())
            );

        double[] argmaxes = FN_Activations.getArgmaxAlong(this, axis);
        int[] max_shape = new int[m_shape.length-1];
        for (int i = 0, j = 0; j < m_shape.length; j++) {
            if (j == axis) continue;
            max_shape[i++] = m_shape[j];
        }

        return new Tensor(max_shape, argmaxes, false);
    }

    public double sum() { return Arrays.stream(m_data).sum(); }

     /**
     * Returns the sums along a given axis of the tensor.
     * Remark: gradients not supported yet.
     */
    public Tensor sum(int axis){
        if (axis == -1) axis = m_shape.length-1;
        if (axis < 0 || axis >= m_shape.length) 
        throw new IllegalArgumentException(
            String.format("Error: axis %d out of bounds for shape %s", axis, formatShape())
        );

        double[] sums = FN_Activations.getSumsAlong(this, axis, this.m_data);
        int[] sum_shape = new int[m_shape.length-1];
        for (int i = 0, j = 0; j < m_shape.length; j++) {
            if (j == axis) continue;
            sum_shape[i++] = m_shape[j];
        }

        return new Tensor(sum_shape, sums, false);
    }

    public Tensor prod(int axis){
        throw new UnsupportedOperationException();
    }

    // --------------------------
    //    BINARY OPERATIONS
    // --------------------------

    /**
     * Add two tensors together element wise.
     * Binary operation: A + B = C, where A, B, C are tensors of the same shape.
     * @param t_B, B in A + B = C, where A is the tensor being called.
     * @return t_C, C in A + B = C, where C is the sum of the elements in A and B.
     */
    public Tensor add(Tensor t_B){
        if (!Arrays.equals(this.m_shape, t_B.m_shape)){
            //TO-DO: broadcasting. See https://numpy.org/doc/stable/user/basics.broadcasting.html.
            throw new IllegalArgumentException("Error: shapes " + this.formatShape() + " and " + t_B.formatShape() + " are not compatible for this operation.");
        }
        
        double dOut[] = new double[m_data.length];
        for (int i = 0; i < dOut.length; i++){
            dOut[i] = this.m_data[i] + t_B.m_data[i];
        }

        Tensor t_C;
        
        if (m_gradEnabled || t_B.m_gradEnabled){
            List<Tensor> children = new ArrayList<>();
            children.add(this);
            children.add(t_B);

            t_C = new Tensor(this.m_shape, dOut, this.m_gradEnabled || t_B.m_gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < m_data.length; i++){
                    if (this.m_gradEnabled) this.m_grad[i] += 1.0 * t_C.m_grad[i];
                    if (t_B.m_gradEnabled) t_B.m_grad[i] += 1.0 * t_C.m_grad[i];
                }
            };
            t_C.m_backward = back;
            t_B.m_gradFn = GradFunctions.AddBackward;

        } else {
            t_C = new Tensor(this.m_shape, dOut);
        }

        return t_C;
    }

    public Tensor addBias(Tensor t_B){
        if (m_shape.length != 2 || t_B.m_shape.length != 2) throw new IllegalCallerException("Error: this method only works if both arguments are 2D.");
        if (t_B.m_shape[0] != 1 && t_B.m_shape[1] != m_shape[1]) throw new IllegalArgumentException("Error: first dimension of input shape must be 1 and second dimensions must match.");

        double dOut[] = new double[m_data.length];
        for(int i = 0; i < m_shape[0]; i++){
            for(int j = 0; j < m_shape[1]; j++){
                dOut[i*m_shape[1] + j] = m_data[i*m_shape[1] + j] + t_B.m_data[j];
            }
        }

        Tensor t_C;

        if (m_gradEnabled || t_B.m_gradEnabled){
            List<Tensor> children = new ArrayList<>();
            children.add(this);
            children.add(t_B);
            t_C = new Tensor(this.m_shape, dOut, this.m_gradEnabled || t_B.m_gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < m_data.length; i++){
                    if (this.m_gradEnabled) this.m_grad[i] += 1.0 * t_C.m_grad[i];
                    if (t_B.m_gradEnabled) t_B.m_grad[i % t_B.m_shape[1]] += 1.0 * t_C.m_grad[i];
                }
            };
            t_C.m_backward = back;
            t_C.m_gradFn = GradFunctions.AddBiasBackward;
        } else {
            t_C = new Tensor(this.m_shape, dOut);
        }

        return t_C;
    }

    /**
     * Subtracts two tensors together element wise.
     * Binary operation: A - B = C, where A, B, C are tensors of the same shape.
     * @param t_B, B in A - B = C, where A and B are tensors of the same shape and A is the tensor being called. 
     * @return C in A - B = C, where C is a tensor representing the subtraction of the elements of B from A element wise and of the same shape.
     */
    public Tensor sub(Tensor t_B) {
        return this.add(t_B.mul(-1.0));
    }

    /**
     * Multiplies two tensors together element wise (i.e. the Hadamard product).
     * Binary operation: A (*) B = C, where A, B, C are tensors of the same shape, where (*) denotes the Hadamard product.
     * @param t B in A (*) B = C, where A is the tensor being called.
     * @return C in A (*) B = C, where C contains the element-wise product of A and B.
     */
    public Tensor mul(Tensor t_B) {
        if (!Arrays.equals(this.m_shape, t_B.m_shape)){
            //TO-DO: broadcasting. See https://numpy.org/doc/stable/user/basics.broadcasting.html.
            throw new IllegalArgumentException("Error: shapes " + this.formatShape() + " and " + t_B.formatShape() + " are not compatible for this operation.");
        }
    
        double[] dOut = m_data.clone();
        for (int i = 0; i < t_B.m_data.length; i++){
            dOut[i] *= t_B.m_data[i];
        }
    
        Tensor t_C;
    
        if (this.m_gradEnabled || t_B.m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);
            children.add(t_B);  
            t_C = new Tensor(m_shape, dOut, m_gradEnabled || t_B.m_gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < m_data.length; i++){
                    if (this.m_gradEnabled) this.m_grad[i] += t_B.m_data[i] * t_C.m_grad[i];
                    if (t_B.m_gradEnabled) t_B.m_grad[i] += this.m_data[i] * t_C.m_grad[i];
                }
            };
            t_C.m_backward = back;
            t_B.m_gradFn = GradFunctions.MulBackward;
        }
        else {
            t_C = new Tensor(m_shape, dOut);
        }
    
        return t_C;
    }

    /**
     * Performs element-wise division of two tensors.
     * Binary operation: A / B = C, where A, B, and C are tensors of the same shape.
     * @param t_B, B in A / B = C, where A is the tensor being called.
     * @return C in A / B = C, where C contains the elements of A divided by the elements of B.
     */
    public Tensor div(Tensor t_B){
        return mul(t_B.pow(-1));
    }

    /**
     * Raises a tensor to the power of another tensor element wise.
     * Binary operation: A^B = C, where A, B, and C are tensors of the same shape.
     * @param t_B, B in A^B = C, where A is the tensor being called.
     * @return t_C, C in A^B = C, where C contains the elements of A raised to the corresponding elements of B.
     */
    public Tensor pow(Tensor t_B){
        if (!Arrays.equals(this.m_shape, t_B.m_shape)){
            throw new IllegalArgumentException("Error: shapes " + this.formatShape() + " and " + t_B.formatShape() + " are not compatible for this operation.");
        }
    
        double[] dOut = new double[m_data.length];
        for(int i = 0; i < m_data.length; i++){
            dOut[i] = Math.pow(m_data[i], t_B.m_data[i]);
        }
        
        Tensor t_C;
    
        if (m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);
            children.add(t_B);
    
            t_C = new Tensor(m_shape, dOut, m_gradEnabled || t_B.m_gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < m_data.length; i++){
                    if(m_gradEnabled) this.m_grad[i] += t_B.m_data[i] * Math.pow(this.m_data[i], t_B.m_data[i]-1) * t_C.m_grad[i];
                    if(t_B.m_gradEnabled) t_B.m_grad[i] += Math.pow(this.m_data[i], t_B.m_data[i]) * Math.log(t_B.m_data[i]) * t_C.m_grad[i];
                }
            };
            t_C.m_backward = back;
            t_B.m_gradFn = GradFunctions.PowBackward;
    
        } else {
            t_C = new Tensor(m_shape, dOut);
        }
    
        return t_C;
    }

    /**
     * The behaviour depends on the arguments in the following way:
     * If both tensors are 2-D:
     * Performs conventional matrix multiplication A * B = C, where A is a mxn tensor, B is a nxp tensor, and C is a mxp tensor.
     * If either tensor is > 2_d:
     * It is treated as a stack of matrices and broadcast accordingly.
     * TODO: Add N-D matmul
     * @param t_B, B in A * B = C, where A is the tensor being called.
     * @param threading, whether or not to use multi-threading.
     * @return C in A * B = C.
     */
    public Tensor matmul(Tensor t_B, boolean threading) {
        return (threading) ? matmulThreaded(t_B) : matmulSeq(t_B);
    }

    // default matmul method, uses multi-threading.
    public Tensor matmul(Tensor t_B){
        return matmul(t_B, true);
    }

    // Multi-threaded matrix multiplication.
    public Tensor matmulThreaded(Tensor t_B){
        if (t_B.m_shape.length == 2 && this.m_shape.length == 2){
            if (this.m_shape[1] != t_B.m_shape[0]){
                throw new IllegalArgumentException("Error: dimensions " + this.formatShape() + " and " + t_B.formatShape() + " are invalid for this operation.");            
            }


            // output shape is the first dimension of A and the second dimension of B
            int[] shapeOut = new int[] {this.m_shape[0], t_B.m_shape[1]};
    
            // create new thread worker class to perform matrix multiplication, default threads = 4
            matMulThread mm = new matMulThread(this, t_B, shapeOut);
            double dataOut[] = mm.matMul();
    
    
            Tensor t_C;
            if (m_gradEnabled || t_B.m_gradEnabled){
                List<Tensor> children = new ArrayList<>();
                children.add(this);
                children.add(t_B);
                
                t_C = new Tensor(shapeOut, dataOut, this.m_gradEnabled || t_B.m_gradEnabled, children);
                // backprop is essentially accumulation of multiplication + summation derivatives.
                Backward back = () -> {
                    for(int r = 0; r < this.m_shape[0]; r++){
                        for(int c = 0; c < t_B.m_shape[1]; c++){
                            for(int k = 0; k < t_B.m_shape[0]; k++){
                                if (m_gradEnabled) this.m_grad[storageIndex(new int[]{r, k})] += t_B.m_data[storageIndex(t_B.m_shape, new int[]{k, c}, t_B.dataLayout)] * t_C.m_grad[storageIndex(shapeOut, new int[]{r, c}, t_C.dataLayout)];
                                if (t_B.m_gradEnabled) t_B.m_grad[storageIndex(t_B.m_shape,new int[]{k, c}, t_B.dataLayout)] += this.m_data[storageIndex(new int[]{r, k})] * t_C.m_grad[storageIndex(shapeOut, new int[]{r, c}, t_C.dataLayout)];
                            }
                        }
                    }
                };
    
                t_C.m_backward = back;
                t_C.m_gradFn = GradFunctions.MatrixMultiplyBackward;
    
            } else {
                t_C = new Tensor(shapeOut, dataOut);
            }
    
            return t_C;
        }
        else throw new UnsupportedOperationException("Error: N-d (N>2) tensor multiplication not supported yet!");
    }

    // Sequential matrix multiplication (i.e. no threading).
    public Tensor matmulSeq(Tensor t_B){
        if (t_B.m_shape.length == 2 && this.m_shape.length == 2){
            if (this.m_shape[1] != t_B.m_shape[0]){
                throw new IllegalArgumentException("Error: dimensions " + this.formatShape() + " and " + t_B.formatShape() + " are invalid for this operation.");            
            }
            
            double dataOut[] = new double[this.m_shape[0] * t_B.m_shape[1]];
            int[] shapeOut = new int[] {this.m_shape[0], t_B.m_shape[1]};
            DataRepresentation layoutOut = (dataLayout == DataRepresentation.ROW_MAJOR || t_B.dataLayout == DataRepresentation.ROW_MAJOR) ? DataRepresentation.ROW_MAJOR : dataLayout;
    
            for(int r = 0; r < this.m_shape[0]; r++){
                for(int c = 0; c < t_B.m_shape[1]; c++){
                    for(int k = 0; k < t_B.m_shape[0]; k++){
                        dataOut[storageIndex(shapeOut, new int[]{r, c}, layoutOut)] += m_data[storageIndex(new int[]{r, k})] * t_B.m_data[storageIndex(t_B.m_shape, new int[]{k, c}, t_B.dataLayout)];
                    }
                }
            }
    
            Tensor t_C;
            if (m_gradEnabled || t_B.m_gradEnabled){
                List<Tensor> children = new ArrayList<>();
                children.add(this);
                children.add(t_B);
                
                t_C = new Tensor(shapeOut, dataOut, this.m_gradEnabled || t_B.m_gradEnabled, children);
    
                Backward back = () -> {
                    for(int r = 0; r < this.m_shape[0]; r++){
                        for(int c = 0; c < t_B.m_shape[1]; c++){
                            for(int k = 0; k < t_B.m_shape[0]; k++){
                                if (m_gradEnabled)
                                    this.m_grad[storageIndex(new int[]{r, k})] += t_B.m_data[storageIndex(t_B.m_shape, new int[]{k, c}, t_B.dataLayout)] * t_C.m_grad[storageIndex(shapeOut, new int[]{r, c}, t_C.dataLayout)];
                                if (t_B.m_gradEnabled) 
                                    t_B.m_grad[storageIndex(t_B.m_shape, new int[]{k, c}, t_B.dataLayout)] += this.m_data[storageIndex(new int[]{r, k})] * t_C.m_grad[storageIndex(shapeOut, new int[]{r, c}, t_C.dataLayout)];
                            }
                        }
                    }
                };
    
                t_C.m_backward = back;
                t_B.m_gradFn = GradFunctions.MatrixMultiplyBackward;
    
            } else {
                t_C = new Tensor(shapeOut, dataOut);
            }
    
            return t_C;
        }
        else throw new UnsupportedOperationException("Error: N-d (N>2) tensor multiplication not supported yet!");
    }


    // ---------------------
    //       ACCESS
    // --------------------

    /**
     * @return the data stored by this tensor.
     */
    public double[] data() {
        return m_data;
    }

    public double item() {
        if (m_data.length != 1) throw new IllegalStateException("Error: calling item() on a tensor that contains more than one item.");
        return m_data[0];
    }

    /**
     * @return the gradient data stored by this tensor.
     */
    public double[] grad() {
        return m_grad;
    }

    /**
     * @return if gradients are enabled for this tensor.
     */
    public boolean gradEnabled() {
        return m_gradEnabled;
    }

    /**
     * @return the shape of this tensor.
     */
    public int[] shape() {
        return m_shape;
    }

    public void reshape(int... shape) {
        if (shape.length == 0) shape = new int[] {1, 1};   
        // shape dimensions must be greater than 0
        for (int d : shape) if (d < 1 && d != -1) throw new IllegalArgumentException("Error: dimensions must be > 0 or -1!");
        
        // the number of elements in data must be consistent with dimension of shape
        int numElements = 1;
        for (int i = 0; i < shape.length; i++)
            numElements *= shape[i];
        if (m_data != null && numElements != m_data.length) 
            throw new IllegalArgumentException("Error: number of elements specified by dimensions (" + numElements +  ") are inconsistent with the length of data provided (" + m_data.length + ").");
        m_shape = shape;
    }

    /**
     * @return the number of elements in this tensor (i.e. product of its dimensions);
     */
    public int size() {
        int numElements = 1;
        for (int i = 0; i < m_shape.length; i++)
            numElements *= m_shape[i];
        return numElements;
    }

    /**
     * @param idxs array indices (x,y,z,...) of the same length as the shape.
     * @return the element stored at tensor index idx.
     */
    public double get(int... idxs) {
        return m_data[storageIndex(idxs)];
    }

    /**
     * Sets the element stored at tensor index idx to x.
     * @param idxs array indices (x,y,z,...) of the same length as the shape.
     * @param x double value to set.
     */
    public void set(double x, int... idxs) {
        m_data[storageIndex(idxs)] = x;
    }

    /**
     * Set the entire data stored in this tensor.
     * @param data the data to set to this tensor.
     */
    public void setData(double[] data) {
        if (data.length != m_data.length) throw new IllegalArgumentException("Error: input data must be of the same length as the number of elements specified by the shape of the tensor.");
        m_data = data;
    }

    // wrapper for setGradient(double[] grad);
    public void setGrad(double[] grad){
        setGradient(grad);
    }

    /**
    * Set the entire gradient data stored in this tensor.
    * @param data the gradient data to set to this tensor.
    */
    public void setGradient(double[] grad) {
        if (m_gradEnabled && grad.length == m_data.length)
            this.m_grad = grad;
    }

    /**
     * Zeros the gradient in this tensor.
     */
    public void zeroGrad() {
        if (m_gradEnabled)
            m_grad = new double[m_grad.length];
    }

    /**
     * Enables/disables gradients in this tensor.
     * @param g If true, enables gradients. If false, disables gradients.
     */
    public void withGrad(boolean g){
        m_gradEnabled = g;
        if (m_gradEnabled){
            m_grad = new double[m_data.length];
            m_backward = () -> {};
        } else {
            if (m_grad != null) m_grad = null;
        }
    }

    /**
     * @return string representation of tensor.
     * Note: this does not print out the actual values in tensor form. Use matcha.utils.Tensors.toString(Tensor) for this.
     */
    @Override
    public String toString() {
        return Tensors.toString(this);
    }

    public String info() {
        if (m_gradEnabled){
            if (m_gradFn != GradFunctions.None) return "Tensor(shape: " + formatShape() + ", gradFn=<" + m_gradFn + ">)";
            else return "Tensor(shape: " + formatShape() + ", gradEnabled=true)";
        } else
            return "Tensor(shape: " + formatShape() + ")";
    }

    /**
     * @return string representation of the shape of the tensor.
     */
    public String formatShape(){
        StringBuilder s = new StringBuilder("(");
        for (int i = 0;  i < m_shape.length; i++){
            s.append(m_shape[i]);
            if (i != m_shape.length - 1) s.append(", ");
        }
        s.append(")");
        return s.toString();
    }

    // --------------------------
    //    OTHER/PRIVATE METHODS
    // -------------------------

    /**
     * Remaps/reorders a integer array according to the mapping specified by axes.
     * @param nums, the array to reorder.
     * @param axes, a permutation of 0,1,...,N-1, where N is the length of the input array.
     * @return the input array remapped according to the axes specification.
     * 
     * EX:
     * remap([1,4,2,5], [2,0,1,3]) -> [2,1,4,5]
     * 
     */
    private int[] remap(int[] nums, int[] axes){
        if (axes.length != nums.length) throw new IllegalArgumentException("Error: axes must be between 0 and N-1.");
        int[] remappedNums = new int[nums.length];

        for(int i = 0; i < axes.length; i++){
            if (axes[i] >= nums.length) throw new IllegalArgumentException("Error: axes must be between 0 and N-1.");
            remappedNums[i] = nums[axes[i]];
        }

        return remappedNums;
    }

    /**
     * Removes an index from an array.
     * @param inArray an array of length N which to remove an index from.
     * @param idx, the index to remove.
     * @return an array of length N-1 with the element at index idx removed.
     */
    private int[] removeIdx(int[] inArray, int idx){
        if (idx > inArray.length-1 || idx < 0) throw new IndexOutOfBoundsException("Error: axis is out of bounds for the given shape.");

        int[] outArray = new int[inArray.length-1];
        int outIdx = 0;
        for(int i = 0; i < inArray.length && outIdx < outArray.length; i++){
            if (i == idx) i++;
            outArray[outIdx] = inArray[i];
            outIdx++;
        }
        
        return outArray;
    }

    /**
     * NOTE: It is recommended to use the built in TensorIterator instead.
     * Gets all indices in the tensor.
     * @return A list of all coordinates in the tensor.
     */
    public List<int[]> getAllIndices(){
        LinkedList<int[]> indexList = new LinkedList<>();
        insertAllIndices(indexList, 0, new int[m_shape.length]);
        return indexList;
    }

    /**
     * Helper function for the getAllIndices() function.
     * @param indexList the list of coordinates to return.
     * @param idx the current dimension to iterate over.
     * @param indices the current (incomplete) coordinate.
     */
    private void insertAllIndices(List<int[]> indexList, int idx, int[] indices){
        if (idx == m_shape.length-1){
            for(int i=0; i< m_shape[idx]; i++){
                int[] tempIndex = indices.clone();
                tempIndex[idx] = i;
                indexList.add(tempIndex);
            }
        } else {
            for(int i=0; i < m_shape[idx]; i++){
                int[] tempIndex = indices.clone();
                tempIndex[idx] = i;
                insertAllIndices(indexList, idx+1, tempIndex);
            }
        }
    }

    /**
     * Gets the index in the data array of this tensor of the element at tensor index (i, j, k, ...) depending on its data representation.
     * @param idxs the element index in the tensor.
     * @return the data index of this element.
     */
    public int storageIndex(int... idxs){
        return storageIndex(m_shape, idxs, dataLayout);
    }

    /**
     * Gets the index in the data array of the element at tensor index (i, j, k, ...) depending on its data representation.
     * @param length the number of elements in the tensor.
     * @param shape the shape of the tensor.
     * @param idxs the element index in the tensor.
     * @param layout the data representation used.
     * @return the data index of this element.
     */
    protected int storageIndex(int[] shape, int[] idxs, DataRepresentation layout){
        switch (layout) {
        case ROW_MAJOR: 
        default:
            return LinAlg.rmo(shape, idxs);
        }
    }

    /**
     * Run backpropagation starting from this tensor with default (1.0) gradients.
     */
    public void backward(){
        double[] gradient = new double[m_grad.length];
        for (int i = 0; i < m_grad.length; i++) gradient[i] = 1.0;

        backward(gradient);
    }

    /**
     * Run backpropagation starting from this tensor with specified gradients.
     * @param gradient the gradient to set to this tensor.
     */
    public void backward(double[] gradient){
        if (!m_gradEnabled)
            throw new IllegalCallerException("Error: calling backprop on non grad-enabled Tensor.");
        if (gradient.length != m_grad.length)
            throw new IllegalArgumentException("Error: Shape mismatch. Gradient passed contains " + gradient.length + " elements but the tensor has " + m_grad.length + " elements.");
        
        m_grad = gradient; // set gradients

        List<Tensor> ordering = new ArrayList<>();
        buildTopo(this, new HashSet<>(), ordering);
        Collections.reverse(ordering);

        for(Tensor val : ordering){
            if (val.m_gradEnabled)
                val.m_backward.pass();
        }
    }

    /**
     * Increments each element in this tensor by its corresponding gradient * step size.
     * @param step_size the amount to scale each gradient by.
     */
    public void step(double step_size) {
        for(int i = 0; i < m_data.length; i++)
            m_data[i] += step_size*m_grad[i];
    }

    /**
     * Builds a topological ordering of Tensors.
     * @param parent current node parent.
     * @param visited set of visited nodes/tensots.
     * @param ordering topological ordering to return.
     */
    private void buildTopo(Tensor parent, Set<Tensor> visited, List<Tensor> ordering) {
        if (!visited.contains(parent)) {
            visited.add(parent);
            if (parent.m_prev != null) {
                for(Tensor child : parent.m_prev){
                    buildTopo(child, visited, ordering);
                }
            }
            ordering.add(parent);
        }
    }

    /**
     * @return TensorIterator of this tensor
     */
    @Override
    public TensorIterator iterator() {
        return new TensorIterator();
    }

    /**
     * @param start starting location for thie Tensor.
     * @param axis the axis to iterate over.
     * @return AxisIterator for this tensor at location start along the specified axis.
     */
    public AxisIterator iterator(int[] start, int axis){
        return new AxisIterator(start, axis);
    }

    /**
     * AxisIterator class, useful for many functionals.
     * Iterates across the Tensor along the direction of it_axis, starting at a specified location.
     * This behaviour is useful for iterating over samples in batches, and across feature dimensions.
     */
    protected class AxisIterator implements Iterator<Double>{
        protected int[] it_pos;
        protected double[] it_data;
        protected int it_axis;
        protected int it_iter;

        AxisIterator(int[] start, int axis){
            it_pos = start;
            it_axis = axis;
            it_data = m_data;
            it_iter = 0;
        }

        @Override
        public boolean hasNext(){
            return it_pos[it_axis] < m_shape[it_axis];
        }

        @Override
        public Double next(){
            double it_val = it_data[storageIndex(it_pos)];
            it_pos[it_axis]++;
            it_iter++;
            return it_val;
        }

        public int iter(){ return it_iter; }
    }

    /**
     * Iterator class for this Tensor. Iterates across Tensor elements in increasing order.
     * Ex: For a 2x3x2 Tensor:
     * [0, 0, 0] -> [0, 0, 1] -> [0, 1, 0] -> [0, 1, 1] -> [0, 2, 0] -> [0, 2, 1] -> [1, 0, 0] -> [1, 0, 1] -> [1, 1, 0] -> [1, 1, 1] -> [1, 2, 0] -> [1, 2, 1].
     */
    public class TensorIterator implements Iterator<Double>{
        protected int[] it_pos;
        protected int it_iter; // current iteration 
        protected int it_curAxis;

        TensorIterator(){
            it_pos = new int[m_shape.length];
            it_pos[m_shape.length-1] = -1;
            it_iter = -1;
            it_curAxis = m_shape.length - 1;
        }

        @Override
        public boolean hasNext(){
            for(int i = 0; i < m_shape.length; i++){
                if (it_pos[i] < m_shape[i]-1) return true;
            }
            return false;
        }

        @Override
        public Double next(){

            if (it_pos[it_curAxis] == m_shape[it_curAxis]-1){
                while(it_pos[it_curAxis] == m_shape[it_curAxis] - 1) {it_curAxis -= 1;}
                it_pos[it_curAxis] += 1;
                while(it_curAxis != m_shape.length-1){
                    it_pos[++it_curAxis] = 0;
                }
            } else{
                it_pos[it_curAxis] += 1;
            }

            it_iter++;

            double it_val = m_data[storageIndex(it_pos)];

            return it_val;
        }

        public int[] index(){ return it_pos; }

        public int iter(){ return it_iter; }

    }
}
