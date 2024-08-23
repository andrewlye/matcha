package matcha.optim;

import java.util.List;

import matcha.engine.Tensor;

public class SGD extends Optimization{
  private double m_lr;

  public SGD(List<Tensor> params, double lr){
    super(params);
    m_lr = lr;
  }

  public void step(){
    for(Tensor t : m_params) t.step(-m_lr);
  }
  
  public void setLR(double lr){ m_lr = lr; }
  public double lr() { return m_lr; }

}
