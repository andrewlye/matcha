package matcha.edu.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Value;

public class vTanh extends vModule<Value[]> {
    public vTanh() {

    }

    @Override
    public Value[] forward(Value[] x) throws Exception {
        Value[] out = new Value[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i].tanh();
        }

        return out;
    }

    @Override
    public List<Value> parameters() {
        return new ArrayList<>();
    }

    @Override
    public String toString() {
        return "Tanh()";
    }
}
