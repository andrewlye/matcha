package matcha.nn;

import java.util.List;

import matcha.engine.Tensor;
import matcha.engine.FN_Activations;;

public class Tanh extends Module{

    @Override
    Tensor forward(Tensor x){
        return FN_Activations.tanh(x);
    }


    @Override
    public String toString(){
        return "Tanh()";
    }

    @Override
    public List<Tensor> parameters(){
        return null;
    }
}
