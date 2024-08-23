package matcha.nn;

import java.util.List;

import matcha.engine.Tensor;

public interface Module {

    public Tensor forward(Tensor x);

    public List<Tensor> parameters();
}
