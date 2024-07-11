package matcha.legacy.nn;

import java.util.Arrays;
import java.util.List;
import matcha.engine.Value;

/**
 * Abstract class for a network module, where T is the type returned.
 */
public abstract class vModule<T> {
    
    private List<String> activations;

    /**
     * Performs a forward pass of data through a network module.
     * @param x, the input data
     * @return the input data with the respecticve transformation(s) applied
     * @throws Exception, if there is a mismatch in input dims
     */
    abstract T forward(Value[] x) throws Exception;

    public T forward(Double[] x) throws Exception{
        Value[] x_vals = Arrays.stream(x).map(o -> new Value(o)).toArray(Value[]::new);

        return forward(x_vals);
    }

    public T forward(Value x) throws Exception{
        return forward(new Value[]{x});
    }

    public T forward(double[] x) throws Exception{
        Double[] x_vals = Arrays.stream(x).mapToObj(o -> Double.valueOf(o)).toArray(Double[]::new);
        return forward(x_vals);
    }
    
    abstract List<Value> parameters();

}
