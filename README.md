* Currently under major redevelopment for tensor operations.

# matcha
*Neural networks in Java.*

## Overview
A standalone, simple yet powerful neural networks library and autograd engine built on Java's standard library. Inspired from [PyTorch](https://pytorch.org/) and Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).
- `matcha.engine` Main code for autograd and backpropagation.
- `matcha.nn` Contains neural network modules from individual neurons to MLPs, as well as loss functions.
- `matcha.optim` Various optimization algorithms, such as Stochastic Gradient Descent (SGD) and Adam.

## Example
```Java
import matcha.engine.*;
import matcha.legacy.nn.*;
import matcha.optim.*;
import java.util.ArrayList;
import java.util.List;

public class Example{
    public static void main(String[] args) throws Exception{
        // constructing a simple network
        List<matcha.legacy.nn.vModule<Value[]>> layers = new ArrayList<>();
        layers.add(new vLinear(3,4));
        layers.add(new vReLU());
        layers.add(new vLinear(4,4));
        layers.add(new vTanh());
        layers.add(new vLinear(4,1));
        layers.add(new vTanh());
        
        vSequential nn = new vSequential(layers);

        // prints network information, such as layers and dimensions
        System.out.println(nn);

        // example input data
        double[][] Xs = new double[4][3];
        Xs[0] = new double[]{2.0, 3.0, -1.0};
        Xs[1] = new double[]{3.0, -1.0, 0.5};
        Xs[2] = new double[]{0.5, 1.0, 1.0};
        Xs[3] = new double[]{1.0, 1.0, -1.0};

        // example target values for each input
        double[] Ys = new double[]{1.0, -1.0, -1.0, 1.0};

        // check newly trained outputs
        ArrayList<Value> outs = new ArrayList<>();
        for(int i=0; i < Xs.length; i++){
            outs.add(nn.forward(Xs[i])[0]);
        }

        System.out.println();
        System.out.print("Initial outputs: [ ");
        for(Value out : outs){
            System.out.print(out.data() + " ");
        }
        System.out.println("]");
        System.out.print("Target outputs: [ ");
        for(double y : Ys){
            System.out.print(y + " ");
        }
        System.out.println("]");
        System.out.println();

        // training loop
        for(int i = 1; i <= 100; i++){
            Value[] outputs = new Value[4];
            for(int j = 0; j < Xs.length; j++){
                outputs[j] = nn.forward(Xs[j])[0];
            }

            vMSELoss loss_func = new vMSELoss(); // Mean Squared Error (MSE) loss function
            Value loss = loss_func.loss(outputs, Ys);
            SGD optim = new SGD(nn.parameters(), 0.1); // SGD optimizer

            if(i % 20 == 0 || i == 1)
                System.out.println("iter: " + i + ", loss: " + loss.data());

            // backpropagation and optimization
            optim.zeroGrad();
            loss.backward();
            optim.step();

        }

        // check newly trained outputs
        outs = new ArrayList<>();
        for(int i=0; i < Xs.length; i++){
            outs.add(nn.forward(Xs[i])[0]);
        }

        System.out.println();
        System.out.print("Final outputs: [ ");
        for(Value out : outs){
            System.out.print(out.data() + " ");
        }
        System.out.println("]");
        System.out.print("Target outputs: [ ");
        for(double y : Ys){
            System.out.print(y + " ");
        }
        System.out.println("]");
    }
}
```
```
Sequential(
   Linear(in_features=3, out_features=4)
   ReLU()
   Linear(in_features=4, out_features=4)
   Tanh()
   Linear(in_features=4, out_features=1)
   Tanh()
)

Initial outputs: [ 0.9559388226690206 0.9341103352135249 0.9555142334237039 0.9542248511948822 ]
Target outputs: [ 1.0 -1.0 -1.0 1.0 ]

iter: 1, loss: 7.568855457498393
iter: 20, loss: 0.047512204519041826
iter: 40, loss: 0.019849683020984978
iter: 60, loss: 0.012462493276612124
iter: 80, loss: 0.009058484820287009
iter: 100, loss: 0.007105392643990915

Final outputs: [ 0.9604232420857443 -0.9597892868883113 -0.9564328883841315 0.9558627194661695 ]
Target outputs: [ 1.0 -1.0 -1.0 1.0 ]
```
