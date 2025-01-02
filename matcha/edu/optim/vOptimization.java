package matcha.edu.optim;

import java.util.List;

import matcha.edu.engine.Value;

public abstract class vOptimization {
    private List<Value> params;

    public vOptimization(List<Value> params){
        this.params = params;
    }
    
    public void zeroGrad(){
        for(Value param : params){;
            param.setGradient(0.0);
        }
    }

    public abstract void step();
}
