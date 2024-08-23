package matcha.nn;

import java.util.List;

import matcha.engine.Tensor;
import matcha.engine.FN_Activations;

public class ReLU implements Module{

    @Override
    public Tensor forward(Tensor x){
        return FN_Activations.relu(x);
    }

    @Override
    public List<Tensor> parameters(){
        return null;
    }


    @Override
    public String toString(){
        return "ReLU()";
    }
}
