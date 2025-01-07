package matcha.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import matcha.engine.Tensor;

public class Sequential implements Module {
    List<Module> layers;

    public Sequential(Module... layers) {
        this.layers = Arrays.asList(layers);
    }

    public Sequential(List<Module> layers){
        this.layers = layers;
    }

    @Override
    public Tensor forward(Tensor x){
        Tensor prev = x;
        Tensor next = null;
        for(Module layer : layers){
            next = layer.forward(prev);
            prev = next;
        }

        return next;
    }

    @Override
    public List<Tensor> parameters(){
        List<Tensor> params = new ArrayList<>();
        for(Module layer : layers){
            if (layer.parameters() != null){
                for (Tensor param : layer.parameters())
                    params.add(param);
            }
        }
                
        return params;
    }

    public String toString(){
        StringBuilder sb = new StringBuilder("Sequential(\n");
        for(Module layer : layers)
            sb.append("   ").append(layer.toString()).append('\n');
        sb.append(")");

        return sb.toString();
    }
}
