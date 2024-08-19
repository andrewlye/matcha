package matcha.nn;

import java.util.List;

import matcha.engine.Tensor;
import matcha.engine.FN_Activations;;

public class Softmax extends Module{

    int m_axis;

    public Softmax(int axis){
        m_axis = axis;
    }

    @Override
    public Tensor forward(Tensor x){
        return FN_Activations.softmax(x, m_axis);
    }

    @Override
    public List<Tensor> parameters(){
        return null;
    }

    @Override
    public String toString(){
        return "Softmax(axis=" + m_axis + ')';
    }
    
}
