package matcha.datasets;

import java.util.Iterator;
import java.util.List;

import matcha.engine.Tensor;

/**
 * All datasets should implement the dataset interface for train test modularity.
 */
public abstract class Dataset implements Iterable<List<Tensor>>{
  /**
   * Get the ith sample in the dataset
   * @param i
   * @return
   */
  public abstract List<Tensor> get(int i);

  /**
   * @return the number of samples in the dataset.
   */
  public abstract int size();

  @Override
  public Iterator<List<Tensor>> iterator() {
    return new Iterator<List<Tensor>>() {
      private int idx = 0;
      @Override
      public boolean hasNext() {
        return idx < size();
      }

      @Override
      public List<Tensor> next() {
        return get(idx++);
      }
    };
  }
}
