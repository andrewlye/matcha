package matcha.nn;

import matcha.engine.Tensor;

public abstract class Loss {
    
    abstract Tensor loss(Tensor input, Tensor target);
}
