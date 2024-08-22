package matcha.datasets.toy;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Tensor;
import matcha.nn.Initializer;

public class SineWave{
  double m_A;
  double m_f;
  double m_phi;

  public SineWave(double A, double f, double phi){
    this.m_A = A;
    this.m_f = f;
    this.m_phi = phi;
  }

  public SineWave() { this(1, 1, 0); }
  public SineWave(double A) { this(A, 1, 0); }
  public SineWave(double A, double f) { this(A, f, 0); }

  public List<Tensor> gen(double xStart, double xEnd, int n){ return gen(xStart, xEnd, 0, 0.1, n); }

  public List<Tensor> gen(double xStart, double xEnd, double loc, double scale, int n){
    ArrayList<Tensor> data = new ArrayList<>(2);
    Tensor t_X = new Tensor(new int[]{1,n});
    Tensor t_y = new Tensor(new int[]{1,n});

    for(int i = 0; i < n; i++){
      double x_i = Initializer.rand.nextDouble() * (xEnd - xStart) + xStart;
      t_X.set(new int[]{0, i}, x_i); 
      double noise = Initializer.rand.nextGaussian() * scale + loc;
      t_y.set(new int[]{0, i}, Math.sin(x_i) + noise);
    }
  
    data.add(t_X);
    data.add(t_y);

    return data;
  }

}
