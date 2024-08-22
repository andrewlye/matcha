package matcha.edu.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.edu.engine.Value;

public class vReLU extends vModule<Value[]> {

    @Override
    Value[] forward(Value[] x) throws Exception {
        Value[] out = new Value[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i].relu();
        }

        return out;
    }

    @Override
    List<Value> parameters() {
        return new ArrayList<>();
    }

    @Override
    public String toString() {
        return "ReLU()";
    }

}
