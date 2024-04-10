package matcha.nn;

import java.util.Arrays;
import java.util.List;
import matcha.engine.Value;

/**
 * Abstract class for a network module, where T is the type returned.
 */
public abstract class Module<T> {
    
    /**
     * Performs a forward pass of data through a network module.
     * @param x, the input data
     * @return the input data with the respecticve transformation(s) applied
     * @throws Exception, if there is a mismatch in input dims
     */
    abstract T forward(Value[] x) throws Exception;

    public T forward(double[] x) throws Exception{
        Value[] x_vals = Arrays.stream(x).mapToObj(o -> new Value(o)).toArray(Value[]::new);

        return forward(x_vals);
    }
    
    abstract List<Value> parameters();

}
