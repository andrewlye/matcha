package matcha.nn;

import java.util.List;

import matcha.engine.Tensor;

public class Softmax extends Module{

    int m_axis;

    public Softmax(int axis){
        m_axis = axis;
    }

    @Override
    public Tensor forward(Tensor x){
        return x.softmax(m_axis);
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
