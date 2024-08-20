package matcha.nn;

import matcha.engine.FN_Loss;
import matcha.engine.Tensor;

public class LogLoss extends Loss {

    @Override
    Tensor loss(Tensor input, Tensor target) {
        return FN_Loss.log_loss(input, target);
    }

    Tensor loss(Tensor input, Tensor target, String reduction) {
        return FN_Loss.log_loss(input, target, reduction);
    }
}
