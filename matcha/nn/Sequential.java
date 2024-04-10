package matcha.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Value;

public class Sequential extends Module<Value[]>{
    List<Module<Value[]>> layers;

    public Sequential(List<Module<Value[]>> layers){
        this.layers = layers;
    }

    @Override
    public Value[] forward(Value[] x) throws Exception {
        Value[] prev = x;
        Value[] next = null;
        for(Module<Value[]> layer : layers){
            next = layer.forward(prev);
            prev = next;
        }

        return next;
    }

    @Override
    public List<Value> parameters() {
       List<Value> params = new ArrayList<>();
        for(Module<Value[]> layer : layers){
            for(Value param : layer.parameters()){
                params.add(param);
            }
        }

        return params;
    }

    public List<List<Neuron>> getNeurons(){
        List<List<Neuron>> out = new ArrayList<>(layers.size());
        for(Module<Value[]> layer : layers){
            if(layer instanceof Linear)
                out.add(((Linear) layer).getNeurons());
        }

        return out;
    }

    public List<Module<Value[]>> getLayers(){
       return layers;
    }

    public List<Neuron> getNeurons(int layer){
        if (layers.get(layer) instanceof Linear){
            return ((Linear) layers.get(layer)).getNeurons();
        }
        else return null;
    }

    public String toString(){
        String model_desc = "Sequential(\n";
        for(Module<Value[]> layer : layers){
            model_desc += "   " + layer.toString() + "\n";
        }
        model_desc += ")";

        return model_desc;
    }
}
