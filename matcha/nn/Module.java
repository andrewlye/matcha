package matcha.nn;

import java.util.List;

import matcha.engine.Tensor;

public abstract class Module {

    abstract Tensor forward(Tensor x);

    abstract List<Tensor> parameters();
}
