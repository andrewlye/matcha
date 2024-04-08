package matcha.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Value;

public class Linear extends Module<Value[]>{
    private List<Neuron> neurons;
    private int in_features;
    private int out_features;

    public Linear(int in_features, int out_features){
        this.in_features = in_features;
        this.out_features = out_features;
        neurons = new ArrayList<>(out_features);

        for(int i = 0; i < out_features; i++){
            neurons.add(new Neuron(in_features));
        }
    }

    public Value[] forward(Value[] x) throws Exception{
        Value[] outs = new Value[out_features];
        for(int i = 0; i < neurons.size(); i++){
            outs[i] = neurons.get(i).forward(x);
        }

        return outs;
    }

    public List<Value> parameters(){
        List<Value> params = new ArrayList<>();
        for(Neuron neuron : neurons){
            for(Value param : neuron.parameters()){
                params.add(param);
            }
        }

        return params;
    }

    public List<Neuron> getNeurons(){
        return neurons;
    }

    public String toString(){
        return "Linear(in_features=" + in_features + ", out_features=" + out_features + ")";
    }
}
