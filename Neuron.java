import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Neuron {
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

    public Value pass(Value[] x) throws Exception{
        if (x.length != weights.length){
            throw new Exception("Warning: input dimensions must match!");
        }
        double sum = 0;
        for(int i = 0; i < x.length; i++){ sum += weights[i].data()*x[i].data(); }
        Value out = new Value(sum);
        return out.tanh();
    }

    public Value pass(double[] x) throws Exception{
        Value[] x_vals = new Value[x.length];
        for(int i = 0; i < x.length; i++){ x_vals[i] = new Value(x[i]); }
        return pass(x_vals);
    }

    public List<Value> parameters(){
        List<Value> params = new ArrayList<>(Arrays.asList(weights));
        params.add(bias);

        return params;
    }



    public String toString(){
        return "Neuron(data=" + Arrays.toString(weights) + ", bias=" + bias.toString() + ")";
    }
}
