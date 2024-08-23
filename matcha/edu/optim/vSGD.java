package matcha.edu.optim;

import java.util.List;

import matcha.edu.engine.Value;

public class vSGD extends Optimization{
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
