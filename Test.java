import matcha.engine.Value;
import matcha.nn.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

public class Test {
    public static void main(String[] args) throws Exception{
        Neuron n = new Neuron(4);
        double[] x = new double[]{-0.8, -0.6, -0.1, 1.2};
        Value out = n.pass(x);
        System.out.println(out.data());
        out.backward();

        for(Value param: n.parameters()){
            param.increment(param.data() + -1.0*param.grad());
        }

        out = n.pass(x);
        System.out.println(out.data());
    }
}
