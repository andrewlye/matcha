package matcha.nn;

import matcha.engine.Tensor;

import java.util.Random;

public class Initializer {
    static Random rand = new Random();

    public static void uniform(Tensor t){
        uniform(t, 0., 1.);
    }

    public static void uniform(Tensor t, double start, double end){
        double[] data_i = t.data();
        for(int i = 0; i < data_i.length; i++){
            data_i[i] = rand.nextDouble() * (end - start) + start;
        }
    }

    public static void seed(long seed){
        rand = new Random(seed);
    }
}
