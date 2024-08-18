package matcha.engine;

import matcha.engine.threads.matMulThread;
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
public class Tensor{
    private int[] m_shape; // shape of tensor
    // data is stored as a 1-d array in memory, with shapes being row-major indexed. See https://pytorch.org/docs/stable/storage.html
    private double[] m_data;
    private boolean m_gradEnabled;
    private double[] m_grad;

    private List<Tensor> m_prev;
    private Backward m_backward;
    public DataRepresentation dataLayout = DataRepresentation.ROW_MAJOR;

    /**
     * Tensor constructor.
     * @param shape the shape of the tensor (e.g. [3, 5, 2] denotes a 3x5x2 tensor).
     * @param data the data held by the tensor, zero-initialized if null.
     * @param gradEnabled boolean indicating if auto differentiation is enabled. False by default.
     */
    public Tensor(int[] shape, double[] data, boolean gradEnabled){
        // if shape is null, tensor become a scalars (1x1)
        if (shape.length == 0) shape = new int[] {1, 1};
        
        // shape dimensions must be greater than 0
        for (int d : shape) if (d < 1) throw new IllegalArgumentException("Error: dimensions must be > 0!");
        
        // the number of elements in data must be consistent with dimension of shape
        int numElements = 1;
        for (int i = 0; i < shape.length; i++)
            numElements *= shape[i];
        if (data != null && numElements != data.length) 
            throw new IllegalArgumentException("Error: number of elements specified by dimensions (" + numElements +  ") are inconsistent with the length of data provided (" + data.length + ").");

        // single-element shapes (shape = [d]) become column vectors (shape' = [d, 1]).
        if (shape.length == 1){
            int[] colVec = new int[2];
            colVec[0] = shape[0];
            colVec[1] = 1;
            shape = colVec;
        }

        this.m_shape = shape;
        this.m_data = (data != null) ? data : new double[numElements];
        withGrad(gradEnabled);
    }

    // Creates a zero-initialized tensor of a specified shape.
    public Tensor(int[] shape){
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

    // Private constructor used for auto differentiation and chain-ruling.
    private Tensor(int[] shape, double[] data, boolean gradEnabled, List<Tensor> children){
        this(shape, data, gradEnabled);
        this.m_prev = children;
    }

    // --------------------------
    //    UNARY OPERATIONS
    // -------------------------

    /**
     * Multiplies a tensor by a scalar x.
     * Unary operation: xA = B, where A and B are tensors of the same shape and x is a scalar.
     * @param sc_x, x in xA = B, where A is the tensor being called and x is a scalar.
     * @return B in xA = B.
     */
    public Tensor mul(double sc_x){
        double[] dOut = Arrays.stream(m_data).map(x -> sc_x * x).toArray();
        Tensor t_C;

        if (m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);

            t_C = new Tensor(m_shape, dOut, m_gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < m_grad.length; i++){
                    this.m_grad[i] += sc_x * t_C.m_grad[i];
                }
            };
            t_C.m_backward = back;
        } else {
            t_C = new Tensor(m_shape, dOut);
        }

        return t_C;
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
            Backward back = () -> {
                for(int i = 0; i < m_grad.length; i++){
                    this.m_grad[i] += sc_x * Math.pow(this.m_data[i], sc_x-1) * t_B.m_grad[i];
                }
            };
            t_B.m_backward = back;
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
    
        } else {
            t_B = new Tensor(m_shape, dOut);
        }
    
        return t_B;
    }

    /**
     * Calculates the hyperbolic tangent (tanh) of all elements in the tensor.
     * Unary Operation: tanh(A) = B, where A and B are tensors of the same shape and tanh is the hyperbolic tangent function.
     * @return t_B, B in tanh(A) = B, where A is the tensor being called.
     */
    public Tensor tanh() {
        double[] dOut = Arrays.stream(m_data).map(x -> Math.tanh(x)).toArray();
    
        Tensor t_B;
    
        if (m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);
    
            t_B = new Tensor(m_shape, dOut, m_gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < m_grad.length; i++){
                    this.m_grad[i] += (1-(t_B.m_data[i]*t_B.m_data[i])) * t_B.m_grad[i];
                }
            };
            t_B.m_backward = back;
    
        } else {
            t_B = new Tensor(m_shape, dOut);
        }
    
        return t_B;
    }

    /**
     * Returns the Rectified Linear Unit (ReLU) activation of this tensor.
     * Unary Operation: relu(A) = B, where A and B are tensors of the same shape.
     * @return B in relu(A) = B, where relu(A) denotes the operation max(0, A) and A is the tensor being called.
     */
    public Tensor relu() {
        double[] dOut = Arrays.stream(m_data).map(x -> Math.max(x, 0)).toArray();
    
        Tensor t_B;
    
        if (m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);
    
            t_B = new Tensor(m_shape, dOut, m_gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < m_grad.length; i++){
                    this.m_grad[i] += ((this.m_data[i] > 0) ? 1 : 0) * t_B.m_grad[i];
                }
            };
            t_B.m_backward = back;
    
        } else {
            t_B = new Tensor(m_shape, dOut);
        }
    
        return t_B;
    }

    public Tensor softmax(int axis){
        double[] maxData = fillDataAlong(getIndicesAlong(axis), getMaxAlong(axis), axis);
        double[] dataOut = m_data.clone();
        for(int i = 0; i < dataOut.length; i++) dataOut[i] -= maxData[i];
        dataOut = Arrays.stream(dataOut).map(x -> Math.exp(x)).toArray();
        double[] expSums = fillDataAlong(getIndicesAlong(axis), getSumsAlong(axis, dataOut), axis);
        for(int i = 0; i < dataOut.length; i++) dataOut[i] /= expSums[i];
        
        Tensor t_B;
        if (m_gradEnabled) {
            List<Tensor> children = new ArrayList<>();
            children.add(this);

            t_B = new Tensor(m_shape, dataOut, m_gradEnabled, children);
            Backward back = () -> {};
            t_B.m_backward = back;
        } else {
            t_B = new Tensor(m_shape, dataOut);
        }

        return t_B;
    }

    public Tensor max(int axis){
        throw new UnsupportedOperationException();
    }

    public Tensor min(int axis){
        throw new UnsupportedOperationException();
    }

    public Tensor sum(int axis){
        throw new UnsupportedOperationException();
    }

    public Tensor prod(int axis){
        throw new UnsupportedOperationException();
    }

    private int[] removeDim(int[] shapeIn, int axis){
        if (axis > shapeIn.length-1 || axis < 0) throw new IndexOutOfBoundsException("Error: axis is out of bounds for the given shape.");

        int[] shapeOut = new int[shapeIn.length-1];
        int outIdx = 0;
        for(int i = 0; i < shapeIn.length && outIdx < shapeOut.length; i++){
            if (i == axis) i++;
            shapeOut[outIdx] = shapeIn[i];
            outIdx++;
        }
        
        return shapeOut;
    }

    private double[] fillDataAlong(List<int[]> partitions, double[] fillData, int axis){
        if (partitions.size() != fillData.length) throw new IllegalArgumentException("Error: number of elements in data and partition should be equal.");

        double[] data = new double[m_data.length];
        for(int i = 0; i < partitions.size(); i++){
            TensorIterator it = new TensorIterator(partitions.get(i), axis);
            while (it.hasNext()){
                data[storageIndex(it.m_pos)] = fillData[i];
                it.next();
            }
        }

        return data;
    }

    private double[] getSumsAlong(int axis){
        List<int[]> indices = getIndicesAlong(axis);
        double[] sums = new double[indices.size()];
        for(int i = 0; i < sums.length; i++){
            double sum = 0;
            int[] idx = indices.get(i);
            TensorIterator it = new TensorIterator(idx.clone(), axis);
            while(it.hasNext()) sum += it.next();
            sums[i] = sum;
        }

        return sums;
    }

    private double[] getSumsAlong(int axis, double[] data){
        List<int[]> indices = getIndicesAlong(axis);
        double[] sums = new double[indices.size()];
        for(int i = 0; i < sums.length; i++){
            double sum = 0;
            int[] idx = indices.get(i);
            TensorIterator it = new TensorIterator(idx.clone(), axis);
            it.it_data = data;
            while(it.hasNext()) sum += it.next();
            sums[i] = sum;
        }

        return sums;
    }

    private double[] getMaxAlong(int axis){
        List<int[]> indices = getIndicesAlong(axis);
        double[] maxes = new double[indices.size()];
        for(int i = 0; i < maxes.length; i++){
            double max = Integer.MIN_VALUE;
            int[] idx = indices.get(i);
            TensorIterator it = new TensorIterator(idx.clone(), axis);
            while(it.hasNext()) max = Math.max(max, it.next());
            maxes[i] = max;
        }
        
        return maxes;
    }

    private List<int[]> getIndicesAlong(int axis){
        if (axis >= m_shape.length) throw new IllegalArgumentException("Error: axis " + axis + " out of bounds for shape " + formatShape() + ".");
        LinkedList<int[]> indexList = new LinkedList<>();
        insertIndicesAlong(indexList, axis, 0, new int[m_shape.length]);
        return indexList;
    }

    private void insertIndicesAlong(List<int[]> indexList, int alongAxis, int idx, int[] indices){
        if (idx == m_shape.length-1){
            if(alongAxis == idx){
                indexList.add(indices);
            } else{
                for(int i=0; i< m_shape[idx]; i++){
                    int[] tempIndex = indices.clone();
                    tempIndex[idx] = i;
                    indexList.add(tempIndex);
                }
            }
        } else if (idx == alongAxis) {
            insertIndicesAlong(indexList, alongAxis, idx+1, indices);
        } else {
            for(int i=0; i < m_shape[idx]; i++){
                int[] tempIndex = indices.clone();
                tempIndex[idx] = i;
                insertIndicesAlong(indexList, alongAxis, idx+1, tempIndex);
            }
        }
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
                for(int i = 0; i < m_grad.length; i++){
                    if (this.m_gradEnabled) this.m_grad[i] += 1.0 * t_C.m_grad[i];
                    if (t_B.m_gradEnabled) t_B.m_grad[i] += 1.0 * t_C.m_grad[i];
                }
            };
            t_C.m_backward = back;

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
                for(int i = 0; i < m_grad.length; i++){
                    if (this.m_gradEnabled) this.m_grad[i] += t_B.m_data[i] * t_C.m_grad[i];
                    if (t_B.m_gradEnabled) t_B.m_grad[i] += this.m_data[i] * t_C.m_grad[i];
                }
            };
            t_C.m_backward = back;
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
    
            t_C = new Tensor(m_shape, dOut, m_gradEnabled, children);
            Backward back = () -> {
                for(int i = 0; i < m_grad.length; i++){
                    this.m_grad[i] += t_B.m_data[i] * Math.pow(this.m_data[i], t_B.m_data[i]-1) * t_C.m_grad[i];
                    t_B.m_grad[i] += Math.pow(this.m_data[i], t_B.m_data[i]) * Math.log(t_B.m_data[i]) * t_C.m_grad[i];
                }
            };
            t_C.m_backward = back;
    
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
            
            int[] shapeOut = new int[] {this.m_shape[0], t_B.m_shape[1]};
    
            matMulThread mm = new matMulThread(this, t_B, shapeOut);
            double dataOut[] = mm.matMul();
    
    
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
                                this.m_grad[storageIndex(new int[]{r, k})] += t_B.m_data[storageIndex(t_B.m_data.length, t_B.m_shape, new int[]{k, c}, t_B.dataLayout)] * t_C.m_grad[storageIndex(dataOut.length, shapeOut, new int[]{r, c}, t_C.dataLayout)];
                                t_B.m_grad[storageIndex(t_B.m_data.length, t_B.m_shape,new int[]{k, c}, t_B.dataLayout)] += this.m_data[storageIndex(new int[]{r, k})] * t_C.m_grad[storageIndex(dataOut.length, shapeOut, new int[]{r, c}, t_C.dataLayout)];
                            }
                        }
                    }
                };
    
                t_C.m_backward = back;
    
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
                        dataOut[storageIndex(dataOut.length, shapeOut, new int[]{r, c}, layoutOut)] += m_data[storageIndex(new int[]{r, k})] * t_B.m_data[storageIndex(t_B.m_data.length, t_B.m_shape, new int[]{k, c}, t_B.dataLayout)];
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
                                this.m_grad[storageIndex(new int[]{r, k})] += t_B.m_data[storageIndex(t_B.m_data.length, t_B.m_shape, new int[]{k, c}, t_B.dataLayout)] * t_C.m_grad[storageIndex(dataOut.length, shapeOut, new int[]{r, c}, t_C.dataLayout)];
                                t_B.m_grad[storageIndex(t_B.m_data.length, t_B.m_shape, new int[]{k, c}, t_B.dataLayout)] += this.m_data[storageIndex(new int[]{r, k})] * t_C.m_grad[storageIndex(dataOut.length, shapeOut, new int[]{r, c}, t_C.dataLayout)];
                            }
                        }
                    }
                };
    
                t_C.m_backward = back;
    
            } else {
                t_C = new Tensor(shapeOut, dataOut);
            }
    
            return t_C;
        }
        else throw new UnsupportedOperationException("Error: N-d (N>2) tensor multiplication not supported yet!");
    }

    private int storageIndex(int[] idxs){
        switch (dataLayout) {
        case ROW_MAJOR: 
        default:
            return LinAlg.rmo(m_data.length, m_shape, idxs);
        }
    }

    private int storageIndex(int length, int[] shape, int[] idxs, DataRepresentation layout){
        switch (layout) {
        case ROW_MAJOR: 
        default:
            return LinAlg.rmo(length, shape, idxs);
        }
    }

    public void backward(){
        if (!m_gradEnabled)
            throw new IllegalCallerException("Error: calling backprop on non grad-enabled Tensor.");

        List<Tensor> ordering = new ArrayList<>();
        buildTopo(this, new HashSet<>(), ordering);
        Collections.reverse(ordering);

        for (int i = 0; i < m_grad.length; i++){
            m_grad[i] = 1.0;
        }

        for(Tensor val : ordering){
            if (!val.m_gradEnabled)
                System.out.println("Warning: some tensors encountered in backprop have gradients disabled.");
            else
            val.m_backward.pass();
        }
    }

    public void step(double step_size) {
        for(int i = 0; i < m_data.length; i++)
            m_data[i] += step_size*m_grad[i];
    }

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

    public double[] data() {
        return m_data;
    }

    public double[] grad() {
        return m_grad;
    }

    public boolean gradEnabled() {
        return m_gradEnabled;
    }

    public int[] shape() {
        return m_shape;
    }

    public double get(int[] idxs) {
        return m_data[storageIndex(idxs)];
    }

    public void set(int[] idxs, double x) {
        m_data[storageIndex(idxs)] = x;
    }

    public void setGradient(double[] grad) {
        if (m_gradEnabled && grad.length == m_data.length)
            this.m_grad = grad;
    }

    public void zeroGrad() {
        if (m_gradEnabled)
            m_grad = new double[m_grad.length];
    }

    public void withGrad(boolean g){
        m_gradEnabled = g;
        if (m_gradEnabled){
            m_grad = new double[m_data.length];
            m_backward = () -> {};
        } else {
            if (m_grad != null) m_grad = null;
        }
    }

    @Override
    public String toString() {
        return "Tensor(shape: " + formatShape() + ", grad_enabled=" + m_gradEnabled + ")";
    }

    public String formatShape(){
        StringBuilder s = new StringBuilder("(");
        for (int i = 0;  i < m_shape.length; i++){
            s.append(m_shape[i]);
            if (i != m_shape.length - 1) s.append(", ");
        }
        s.append(")");
        return s.toString();
    }

    public Iterator<Double> iterator(int[] start, int axis){
        return new TensorIterator(start, axis);
    }

    private class TensorIterator implements Iterator<Double>{
        private int[] m_pos;
        private double[] it_data;
        private int m_axis;

        TensorIterator(int[] start, int axis){
            m_pos = start;
            m_axis = axis;
            it_data = m_data;
        }

        @Override
        public boolean hasNext(){
            return m_pos[m_axis] < m_shape[m_axis];
        }

        @Override
        public Double next(){
            double n_val = it_data[storageIndex(m_pos)];
            m_pos[m_axis]++;
            return n_val;
        }
    }
}
