package matcha.datasets;

import java.util.List;

import matcha.engine.Tensor;

/**
 * All datasets should implement the dataset interface for train test modularity.
 */
public interface Dataset {
  
  /**
   * Get the ith sample in the dataset
   * @param i
   * @return
   */
  public List<Tensor> get(int i);

  /**
   * @return the number of samples in the dataset.
   */
  public int size();
  
}
