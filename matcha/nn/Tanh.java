package matcha.nn;

import java.util.List;

import matcha.engine.Tensor;

public class Tanh extends Module{

    @Override
    Tensor forward(Tensor x){
        return x.tanh();
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
