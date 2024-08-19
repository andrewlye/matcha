package matcha.engine;

public enum gradFunctions {
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
}
