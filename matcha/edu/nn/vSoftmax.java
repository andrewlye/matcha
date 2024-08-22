package matcha.edu.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.edu.engine.Value;

public class vSoftmax extends vModule<Value[]> {
    @Override
    Value[] forward(Value[] x) throws Exception {
        Value[] out = new Value[x.length];
        double norm = 0.0;

        for (int i = 0; i < x.length; i++) {
            out[i] = x[i].exp();
            norm += out[i].data();
        }
        for (int i = 0; i < x.length; i++) {
            out[i] = out[i].div(norm);
        }

        return out;
    }

    @Override
    List<Value> parameters() {
        return new ArrayList<>();
    }

    @Override
    public String toString() {
        return "Softmax()";
    }
}
