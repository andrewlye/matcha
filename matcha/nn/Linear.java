package matcha.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Tensor;
import matcha.nn.Initializer;

public class Linear implements Module{

    private Tensor m_weights;
    private Tensor m_bias;
    private int m_inFeatures;
    private int m_outFeatures;
    private boolean m_hasBias;

    public Linear(int in_features, int out_features){ this(in_features, out_features, true, true); }

    public Linear(int in_features, int out_features, boolean bias){ this(in_features, out_features, bias, true); }

    public Linear(int in_features, int out_features, boolean bias, boolean gradEnabled){
        m_inFeatures = in_features;
        m_outFeatures = out_features;
        m_hasBias = bias;
        
        m_weights = new Tensor(new int[]{in_features, out_features}, gradEnabled);
        if (bias){ 
            m_bias = new Tensor(new int[]{1, out_features}, gradEnabled);
            Initializer.uniform(m_bias, -Math.sqrt(1.0 / in_features), Math.sqrt(1.0 / in_features));
        }

        Initializer.uniform(m_weights, -Math.sqrt(1.0 / in_features), Math.sqrt(1.0 / in_features));
    }

    @Override
    public Tensor forward(Tensor x){
        return (m_hasBias) ? x.matmul(m_weights).addBias(m_bias) : x.matmul(m_weights);
    }

    @Override
    public List<Tensor> parameters(){
        List<Tensor> wandb = new ArrayList<>();
        wandb.add(m_weights);
        if (m_hasBias) wandb.add(m_bias);
        return wandb;
    }

    public String toString(){
        return "Linear(in_features=" + m_inFeatures + ", out_features=" + m_outFeatures + ", bias=" + m_hasBias + ")";
    }
}
