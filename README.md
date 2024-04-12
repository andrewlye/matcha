# matcha
*Neural networks in Java.*

## Overview
A standalone simple yet powerful neural networks library and autograd engine built on Java that combines torch-like API with standard library features. Inspired from [PyTorch](https://pytorch.org/) and [micrograd](https://github.com/karpathy/micrograd).
- `matcha.engine` Contains the main code for autograd and backpropagation.
- `matcha.nn` Contains neural network modules from neurons to MLPs, as well as loss functions.
- `matcha.optim` Contains various optimization algorithms, such as Stochastic Gradient Descent (SGD) and Adam.

## Example
```
import matcha.engine.*;
import matcha.nn.*;
import matcha.optim.*;
import java.util.ArrayList;
import java.util.List;

public static void main(String[] args) throws Exception{
    List<matcha.nn.Module<Value[]>> layers = new ArrayList<>();
    layers.add(new Linear(3,4));
    layers.add(new ReLU());
    layers.add(new Linear(4,4));
    layers.add(new Tanh());
    layers.add(new Linear(4,1));
    layers.add(new Tanh());
    
    Sequential nn = new Sequential(layers);

    System.out.println(nn);

    double[][] Xs = new double[4][3];
    Xs[0] = new double[]{2.0, 3.0, -1.0};
    Xs[1] = new double[]{3.0, -1.0, 0.5};
    Xs[2] = new double[]{0.5, 1.0, 1.0};
    Xs[3] = new double[]{1.0, 1.0, -1.0};

    double[] Ys = new double[]{1.0, -1.0, -1.0, 1.0};

    for(int i = 1; i <= 200; i++){
        Value[] outputs = new Value[4];
        for(int j = 0; j < Xs.length; j++){
            outputs[j] = nn.forward(Xs[j])[0];
        }
        MSELoss loss_func = new MSELoss();
        Value loss = loss_func.loss(outputs, Ys);
        SGD optim = new SGD(nn.parameters(), 0.1);

        if(i % 10 == 0)
            System.out.println("epoch " + i + ", loss: " + loss.data());

        optim.zeroGrad();
        loss.backward();
        optim.step();

    }

    ArrayList<Value> outs = new ArrayList<>();
    for(int i=0; i < Xs.length; i++){
        outs.add(nn.forward(Xs[i])[0]);
    }
    System.out.print("[ ");
    for(Value out : outs){
        System.out.print(out.data() + " ");
    }
    System.out.println("]");
}
```
