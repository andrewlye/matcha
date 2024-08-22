package matcha.edu.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.edu.engine.Value;

/**
 * A sequential container for matcha.legacy.nn Modules
 */
public class vSequential extends vModule<Value[]> {
    List<vModule<Value[]>> layers;

    /**
     * Modules are added in the order they are passed to the constructor.
     * 
     * @param layers A list of matcha.legacy.nn.Module classes
     */
    public vSequential(List<vModule<Value[]>> layers) {
        this.layers = layers;
    }

    // Forwards are computed in the order the of the module list passed to the
    // constructor.
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
        String model_desc = "Sequential(\n";
        for (vModule<Value[]> layer : layers) {
            model_desc += "   " + layer.toString() + "\n";
        }
        model_desc += ")";

        return model_desc;
    }
}
