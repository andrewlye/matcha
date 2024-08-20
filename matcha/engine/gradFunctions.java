package matcha.engine;

public enum GradFunctions {
    None,
    ScalarMulBackward,
    ScalarPowBackward,
    ExpBackward,
    AddBackward,
    AddBiasBackward,
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
