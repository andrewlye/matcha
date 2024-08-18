package matcha.nn;

import java.util.List;

import matcha.engine.Tensor;

public abstract class Module {

    abstract Tensor forward(Tensor x) throws Exception;

    abstract List<Tensor> parameters();
}
