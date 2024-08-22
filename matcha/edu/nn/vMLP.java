package matcha.edu.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import matcha.edu.engine.Value;

/**
 * A multi-layer perceptron (MLP) module.
 */
public class vMLP extends vModule<Value[]> {
    private List<vModule<Value[]>> layers;

    /**
     * @param in_channels     Number of channels of the input
     * @param hidden_channels List of hidden channel dimensions
     * @param activations     List of inter-layer activations
     * @throws Exception
     */
    public vMLP(int in_channels, List<Integer> hidden_channels, List<String> activations) throws Exception {
        List<Integer> sizes = new ArrayList<>(hidden_channels);
        sizes.add(0, in_channels);

        if (activations.size() != sizes.size() - 1) {
            throw new Exception("Warning: activations must be the same in length as the number of layers!");
        }

        layers = new ArrayList<>(sizes.size() - 1);
        for (int i = 0; i < sizes.size() - 1; i++) {
            layers.add(new vLinear(sizes.get(i), sizes.get(i + 1)));
            if (activations.get(i).toLowerCase().equals("tanh")) {
                layers.add(new vTanh());
            } else if (activations.get(i).toLowerCase().equals("relu")) {
                layers.add(new vReLU());
            }
        }

        System.out.println(layers.size());

    }

    @Override
    public Value[] forward(Value[] x) throws Exception {
        Value[] prev = x;
        Value[] next = null;
        for (vModule<Value[]> layer : layers) {
            next = layer.forward(prev);
            prev = next;
        }

        return next;
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (vModule<Value[]> layer : layers) {
            for (Value param : layer.parameters()) {
                params.add(param);
            }
        }

        return params;
    }

    /**
     * @return All neurons in the network's non-activation layers
     */
    public List<List<vNeuron>> getNeurons() {
        List<List<vNeuron>> out = new ArrayList<>(layers.size());
        for (vModule<Value[]> layer : layers) {
            if (layer instanceof vLinear)
                out.add(((vLinear) layer).getNeurons());
        }

        return out;
    }

    /**
     * @param layer, the layer to retrieve neurons from
     * @return All neurons in the specified layer of the network, if applicable
     */
    public List<vNeuron> getNeurons(int layer) {
        if (layers.get(layer) instanceof vLinear) {
            return ((vLinear) layers.get(layer)).getNeurons();
        } else
            return null;
    }

    /**
     * @return All network layers
     */
    public List<vModule<Value[]>> getLayers() {
        return layers;
    }

    public String toString() {
        String model_desc = "MLP(\n";
        for (vModule<Value[]> layer : layers) {
            model_desc += "   " + layer.toString() + "\n";
        }
        model_desc += ")";

        return model_desc;
    }
}
