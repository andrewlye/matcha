package matcha.nn;

import java.util.Arrays;

import matcha.engine.Value;

public abstract class Loss<T>{
    abstract T loss(Value[] outputs, Value[] targets);

    public T loss(Value[] outputs, double[] targets){
        Value[] targs = Arrays.stream(targets).mapToObj(o -> new Value(o)).toArray(Value[]::new);
        return loss(outputs, targs);
    }

}
