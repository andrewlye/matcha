package matcha.nn;

import matcha.engine.Value;

public class MSELoss extends Loss<Value>{
    public MSELoss(){

    }

    public Value loss(Value[] outputs, Value[] targets){
        Value loss = new Value(0.0);
        for(int i = 0; i < targets.length; i++){
            loss = loss.add((outputs[i].sub(targets[i])).pow(2));
        }

        return loss;
    }
}
