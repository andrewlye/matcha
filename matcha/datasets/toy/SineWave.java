package matcha.datasets.toy;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import matcha.datasets.Dataset;
import matcha.engine.Tensor;
import matcha.nn.Initializer;

/**
 * Generates batched or singular (x,y) pair samples from a sine wave of specified amplitude, frequency, phase, domain, and noise.
 * @author andrew ye
 */
public class SineWave extends Dataset{
  double m_A; // amplitutde
  double m_f; // frequency
  double m_phase; // phase shift

  private Map<String, Number> m_config; // configs

  private List<Tensor> m_data;
  private List<Tensor> m_target;

  /**
   * Initializes a sin wave dataset of the form y(x) = Asin(fx+phase)
   * @param A
   * @param f
   * @param phase
   * @param config see init() for valid configurations.
   */
  public SineWave(double A, double f, double phase, Map<String, Number> config){
    this.m_A = A;
    this.m_f = f;
    this.m_phase = phase;
    this.m_config = init();
    updateConfig(config);

    m_data = new ArrayList<>();
    m_target = new ArrayList<>();

    for(int i = 0; i < (int) ((int) m_config.get("n_samples") / (int) m_config.get("batch_size")); i++){
      gen((int) m_config.get("batch_size"));
    }
    if (m_data.size() * (int) m_config.get("batch_size") < (int) m_config.get("n_samples"))
      gen((int) m_config.get("n_samples") - m_data.size() * (int) m_config.get("batch_size"));
  }

  public SineWave() { this(1, 1, 0, new HashMap<String, Number>()); }
  public SineWave(double A) { this(A, 1, 0,  new HashMap<String, Number>()); }
  public SineWave(double A, double f) { this(A, f, 0,  new HashMap<String, Number>()); }
  public SineWave(double A, double f, double phase) { this(A, f, phase,  new HashMap<String, Number>()); }

  /**
   * Generates n random samples from the sine wave.
   * @param n
   */
  private void gen(int n){
    Tensor t_X = new Tensor(new int[]{n,1});
    Tensor t_y = new Tensor(new int[]{n,1});

    for(int i = 0; i < n; i++){
      double x_i = Initializer.rand.nextDouble() * ((double) m_config.get("x_end") - (double) m_config.get("x_start")) + (double) m_config.get("x_start");
      t_X.set(new int[]{i, 0}, x_i); 
      double noise = Initializer.rand.nextGaussian() * (double) m_config.get("scale") + (double) m_config.get("loc");
      t_y.set(new int[]{i, 0}, m_A * Math.sin(m_f * x_i + m_phase) + noise);
    }
  
    m_data.add(t_X);
    m_target.add(t_y);
  }

  private Map<String, Number> init(){
    Map<String, Number> configs = new HashMap<>();
    configs.put("x_start", -Math.PI); // domain start
    configs.put("x_end", Math.PI); // domain end
    configs.put("loc", 0.0); // mean for addedgaussian noise
    configs.put("scale", 0.1*m_A); // std dev for added gaussian noise 
    configs.put("n_samples", 500); // number of samples ot generate
    configs.put("batch_size", 4); // batch size per samples such that each sample is a batch_size*1 tensor.

    return configs;
  }

  public void updateConfig(Map<String, Number> config){
    for(String key : config.keySet()){
        if (!m_config.containsKey(key)) throw new IllegalArgumentException("Error: " + key + " is an invalid configuration for this component.");
        if (m_config.get(key).getClass() != config.get(key).getClass()) throw new IllegalArgumentException("Error: " + key + " expects a type of " + m_config.get(key).getClass() + " but was provided a " + config.get(key).getClass() + '.');
        m_config.replace(key, config.get(key));
    }
  }

  @Override
  public List<Tensor> get(int i){
    List<Tensor> sample = new ArrayList<>();
    sample.add(m_data.get(i));
    sample.add(m_target.get(i));

    return sample;
  }

  @Override
  public int size(){
    return m_data.size();
  }

  public Object getConfig(String s){ return m_config.get(s); }

  public List<Tensor> data() { return m_data; }
  public List<Tensor> target() { return m_target; }
  public Map<String, Number> config(){ return m_config; }

}
