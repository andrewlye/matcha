package matcha.optim;

import java.util.List;
import matcha.engine.Tensor;

public abstract class Optimization {
  protected List<Tensor> m_params;

  public Optimization(List<Tensor> params) { m_params = params; }

  public void zeroGrad(){
    for(Tensor t : m_params) t.setGrad(new double[t.size()]);
  }

  public abstract void step();
}
