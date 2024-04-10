package matcha.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Value;

public class ReLU extends Module<Value[]>{

    @Override
    Value[] forward(Value[] x) throws Exception {
        Value[] out = new Value[x.length];
        for(int i = 0; i < x.length; i++){
            out[i] = x[i].relu();
        }

        return out;
    }

    @Override
    List<Value> parameters() {
        return new ArrayList<>();
    }

    @Override
    public String toString(){
        return "ReLU()";
    }
    
}
