package matcha.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import matcha.engine.Value;

public class MLP extends Module<Value[]>{
    private List<Linear> layers;
    private List<String> activations;
    
    public MLP(int in_channels, List<Integer> hidden_channels){
        List<Integer> sizes = new ArrayList<>(hidden_channels);
        sizes.add(0, in_channels);

        layers = new ArrayList<>(sizes.size()-1);
        for(int i = 0; i < sizes.size() - 1; i++){
            layers.add(new Linear(sizes.get(i), sizes.get(i+1)));
        }
    }

    public MLP(int in_channels, List<Integer> hidden_channels, List<String> activations) throws Exception{
        List<Integer> sizes = new ArrayList<>(hidden_channels);
        sizes.add(0, in_channels);

        if (activations.size() != sizes.size()-1){
            throw new Exception("Warning: activations must be the same in length as the number of layers!");
        } else{
            this.activations = activations;
        }

        layers = new ArrayList<>(sizes.size()-1);
        for(int i = 0; i < sizes.size() - 1; i++){
            layers.add(new Linear(sizes.get(i), sizes.get(i+1), activations.get(i)));
        }

        System.out.println(layers.size());

    }

    @Override
    public Value[] forward(Value[] x) throws Exception{
        Value[] prev = x;
        Value[] next = null;
        for(Linear layer : layers){
            next = layer.forward(prev);
            prev = next;
        }

        return next;
    }

    @Override
    public List<Value> parameters(){
        List<Value> params = new ArrayList<>();
        for(Linear layer : layers){
            for(Value param : layer.parameters()){
                params.add(param);
            }
        }

        return params;
    }

    public List<List<Neuron>> getNeurons(){
        List<List<Neuron>> out = new ArrayList<>(layers.size());
        for(Linear layer : layers){
            out.add(layer.getNeurons());
        }

        return out;
    }

    public List<Neuron> getNeurons(int layer){
        return layers.get(layer).getNeurons();
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
