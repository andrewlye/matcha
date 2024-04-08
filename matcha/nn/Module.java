package matcha.nn;

import java.util.List;
import matcha.engine.Value;

/**
 * Abstract class for a network module, where T is the type returned.
 */
public abstract class Module<T> {
    
    abstract T forward(Value[] x) throws Exception;

    public T forward(double[] x) throws Exception{
        Value[] x_vals = new Value[x.length];
        for(int i = 0; i < x.length; i++){ x_vals[i] = new Value(x[i]); }

        return forward(x_vals);
    }
    
    abstract List<Value> parameters();

}
