package matcha.utils.math;

import matcha.engine.*;;

public class LinAlg {
    public static double[][] eye(int dim){
        double[][] I = new double[dim][dim];
        
        for(int i = 0; i < dim; i++){
            for(int j = 0; j < dim; j++){
                if (i == j) I[i][j] = 1.0;
            }
        }

        return I;
    }

    public static double[][] diagFlat(Value[] v){
        double[][] A = new double[v.length][v.length];
        
        for(int i = 0; i < v.length; i++){
            for(int j = 0; j < v.length; j++){
                if (i == j) A[i][j] = v[i].data();
            }
        }

        return A;
    }

}   
