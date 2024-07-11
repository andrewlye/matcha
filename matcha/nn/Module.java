package matcha.nn;

import matcha.engine.Tensor;

public abstract class Module {

    abstract Tensor forward(Tensor x) throws Exception;
}
