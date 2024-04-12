package matcha.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import matcha.engine.Value;

/**
 * A multi-layer perceptron (MLP) module.
 */
public class MLP extends Module<Value[]>{
    private List<Module<Value[]>> layers;

    /**
     * @param in_channels Number of channels of the input
     * @param hidden_channels List of hidden channel dimensions
     * @param activations List of inter-layer activations
     * @throws Exception
     */
    public MLP(int in_channels, List<Integer> hidden_channels, List<String> activations) throws Exception{
        List<Integer> sizes = new ArrayList<>(hidden_channels);
        sizes.add(0, in_channels);

        if (activations.size() != sizes.size()-1){
            throw new Exception("Warning: activations must be the same in length as the number of layers!");
        }

        layers = new ArrayList<>(sizes.size()-1);
        for(int i = 0; i < sizes.size() - 1; i++){
            layers.add(new Linear(sizes.get(i), sizes.get(i+1)));
            if (activations.get(i).toLowerCase().equals("tanh")){
                layers.add(new Tanh());
            } else if (activations.get(i).toLowerCase().equals("relu")){
                layers.add(new ReLU());
            }
        }

        System.out.println(layers.size());

    }

    @Override
    public Value[] forward(Value[] x) throws Exception{
        Value[] prev = x;
        Value[] next = null;
        for(Module<Value[]> layer : layers){
            next = layer.forward(prev);
            prev = next;
        }

        return next;
    }

    @Override
    public List<Value> parameters(){
        List<Value> params = new ArrayList<>();
        for(Module<Value[]> layer : layers){
            for(Value param : layer.parameters()){
                params.add(param);
            }
        }

        return params;
    }

    /**
     * @return All neurons in the network's non-activation layers
     */
    public List<List<Neuron>> getNeurons(){
        List<List<Neuron>> out = new ArrayList<>(layers.size());
        for(Module<Value[]> layer : layers){
            if(layer instanceof Linear)
                out.add(((Linear) layer).getNeurons());
        }

        return out;
    }

    /**
     * @param layer, the layer to retrieve neurons from
     * @return All neurons in the specified layer of the network, if applicable
     */
    public List<Neuron> getNeurons(int layer){
        if (layers.get(layer) instanceof Linear){
            return ((Linear) layers.get(layer)).getNeurons();
        }
        else return null;
    }

    /**
     * @return All network layers
     */
    public List<Module<Value[]>> getLayers(){
        return layers;
     }

    public String toString(){
        String model_desc = "MLP(\n";
        for(Module<Value[]> layer : layers){
            model_desc += "   " + layer.toString() + "\n";
        }
        model_desc += ")";

        return model_desc;
    }
}
