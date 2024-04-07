package matcha.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import matcha.engine.Value;

public class MLP {
    private List<Linear> layers;
    
    public MLP(int in_channels, List<Integer> hidden_channels){
        List<Integer> sizes = new ArrayList<>(hidden_channels);
        sizes.add(0, in_channels);

        layers = new ArrayList<>(sizes.size()-1);
        for(int i = 0; i < sizes.size() - 1; i++){
            layers.add(new Linear(sizes.get(i), sizes.get(i+1)));
        }
    }

    public Value[] forward(Value[] x) throws Exception{
        Value[] prev = x;
        Value[] next = null;
        for(Linear layer : layers){
            next = layer.pass(prev);
            prev = next;
        }

        return next;
    }

    public Value[] forward(double[] x) throws Exception{
        Value[] x_vals = new Value[x.length];
        for(int i = 0; i < x.length; i++){ x_vals[i] = new Value(x[i]); }
        return forward(x_vals);
    }

    public List<Value> parameters(){
        List<Value> params = new ArrayList<>();
        for(Linear layer : layers){
            for(Value param : layer.parameters()){
                params.add(param);
            }
        }

        return params;
    }

    public String toString(){
        String model_desc = "MLP(\n";
        for(Linear layer : layers){
            model_desc += "   " + layer.toString() + "\n";
        }
        model_desc += ")";

        return model_desc;
    }
}
