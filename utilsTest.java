
import java.util.Arrays;
import matcha.utils.math.LinAlg;

public class utilsTest {
    public static void main(String[] args){
        double[][] I = LinAlg.eye(5);
        for(double[] row : I){
            System.out.println(Arrays.toString(row));
        }
    }
}
