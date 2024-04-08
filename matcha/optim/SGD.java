package matcha.optim;

import matcha.engine.Value;
import java.util.List;

public class SGD extends Optimization{
    private List<Value> params;
    private double lr;

    public SGD(List<Value> params, double lr){
        super(params);
        this.params = params;
        this.lr = lr;
    }

    public void step(){
        for(Value param : params){
            param.step(-lr);
        }
    }

}
