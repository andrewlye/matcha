package matcha.nn;

import matcha.engine.FN_Loss;
import matcha.engine.Tensor;

public class MSELoss implements Loss {

    @Override
    public Tensor loss(Tensor input, Tensor target){ return loss(input, target, "mean"); }

    public Tensor loss(Tensor input, Tensor target, String reduction) {
       return FN_Loss.mse_loss(input, target, reduction);
    }
    
}
