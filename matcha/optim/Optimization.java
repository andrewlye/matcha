package matcha.optim;

import java.util.List;
import matcha.engine.Value;

public abstract class Optimization {
    private List<Value> params;

    public Optimization(List<Value> params){
        this.params = params;
    }
    
    public void zeroGrad(){
        for(Value param : params){;
            param.setGradient(0.0);
        }
    }
}
