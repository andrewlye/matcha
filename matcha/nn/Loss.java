package matcha.nn;

import matcha.engine.Tensor;

public interface Loss {
    public Tensor loss(Tensor input, Tensor target);
}
