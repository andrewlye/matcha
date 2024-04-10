package matcha.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Value;

public class Tanh extends Module<Value[]>{
    public Tanh(){

    }

    @Override
    public Value[] forward(Value[] x) throws Exception {
        Value[] out = new Value[x.length];
        for(int i = 0; i < x.length; i++){
            out[i] = x[i].tanh();
        }

        return out;
    }

    @Override
    public List<Value> parameters() {
        return new ArrayList<>();
    }

    @Override
    public String toString(){
        return "Tanh()";
    }
}
