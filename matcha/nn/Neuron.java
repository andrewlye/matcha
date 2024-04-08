package matcha.nn;

import matcha.engine.Value;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * A single neuron, takes an input of the same dimension as its number of weights and computes its dot product + bias run through a nonlinear activation.
 */
public class Neuron extends Module<Value>{
    Value[] weights;
    Value bias;
    
    public Neuron(int n_in){
        weights = new Value[n_in];
        Random r = new Random();
        for(int i = 0; i < weights.length; i++){
            weights[i] = new Value(r.nextDouble());
        }
        bias = new Value(r.nextDouble());
    }

    /**
     * 
     * @param x
     * @return
     * @throws Exception
     */
    public Value forward(Value[] x) throws Exception{
        if (x.length != weights.length){
            throw new Exception("Warning: input dimensions must match!");
        }
        Value out = new Value(0.0);
        for(int i = 0; i < x.length; i++){ 
            out = out.add(weights[i].mul(x[i])); 
        }
        out = out.add(bias);
        return out.tanh();
    }

    /**
     * Returns references to all of the parameters 
     * @return
     */
    public List<Value> parameters(){
        List<Value> params = new ArrayList<>(Arrays.asList(weights));
        params.add(bias);

        return params;
    }

    public String toString(){
        return "Neuron(data=" + Arrays.toString(weights) + ", bias=" + bias.toString() + ")";
    }
}
