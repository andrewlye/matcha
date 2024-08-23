package matcha.nn;

import matcha.engine.FN_Loss;
import matcha.engine.Tensor;

public class LogLoss implements Loss {

    @Override
    public Tensor loss(Tensor input, Tensor target) {
        return FN_Loss.log_loss(input, target);
    }

    public Tensor loss(Tensor input, Tensor target, String reduction) {
        return FN_Loss.log_loss(input, target, reduction);
    }
}
