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
    private Value[] weights;
    private Value bias;
    private String activation = "none";
    
    public Neuron(int n_in){
        weights = new Value[n_in];
        Random r = new Random();
        for(int i = 0; i < weights.length; i++){
            weights[i] = new Value(r.nextDouble());
        }
        bias = new Value(r.nextDouble());
    }

    public Neuron(int n_in, String activation) throws Exception{
        if (!((activation.equals("relu")) || activation.equals("tanh") || activation.equals("none"))){
            throw new Exception("Warning: activation function must be of the following: 'relu', 'tanh', 'none'.");
        }

        weights = new Value[n_in];
        Random r = new Random();
        for(int i = 0; i < weights.length; i++){
            weights[i] = new Value(r.nextDouble());
        }
        bias = new Value(r.nextDouble());
        this.activation = activation;
    }

    /**
     * 
     * @param x
     * @return
     * @throws Exception
     */
    @Override
    public Value forward(Value[] x) throws Exception{
        if (x.length != weights.length){
            throw new Exception("Warning: input dimensions must match!");
        }
        Value out = new Value(0.0);
        for(int i = 0; i < x.length; i++){ 
            out = out.add(weights[i].mul(x[i])); 
        }
        out = out.add(bias);

        if (activation.equals("relu")){
            return out.relu();
        } else if (activation.equals("tanh")){
            return out.tanh();
        } else{
            return out;
        }
    }

    @Override
    public List<Value> parameters(){
        List<Value> params = new ArrayList<>(Arrays.asList(weights));
        params.add(bias);

        return params;
    }

    public String getActivation(){
        return activation;
    }

    public String toString(){
        return "Neuron(data=" + Arrays.toString(weights) + ", bias=" + bias.toString() + ")";
    }
}
