package matcha.nn;

import java.util.List;

import matcha.engine.Tensor;

public class ReLU extends Module{

    @Override
    Tensor forward(Tensor x){
        return x.relu();
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
