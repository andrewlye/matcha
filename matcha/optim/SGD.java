package matcha.optim;

import java.util.List;

import matcha.engine.Tensor;

public class SGD extends Optimization{
  public double lr;

  public SGD(List<Tensor> params, double lr){
    super(params);
    this.lr = lr;
  }

  public void step(){
    for(Tensor t : m_params) t.step(-lr);
  }
}
