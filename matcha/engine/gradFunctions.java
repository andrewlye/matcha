package matcha.engine;

public enum GradFunctions {
    None,
    ScalarMulBackward,
    ScalarPowBackward,
    ExpBackward,
    AddBackward,
    MulBackward,
    PowBackward,
    MatrixMultiplyBackward,
    TanhBackward,
    ReLUBackward,
    SoftmaxBackward,
    CrossEntropyBackward,
    MeanLogLossBackward,
    SumLogLossBackward,
    LogLossBackward,
}
