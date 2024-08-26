# matcha
*Neural networks in Java.*

## Overview
A standalone, simple yet powerful neural networks library and tensor engine built on Java's standard library featuring torch-like API, multithreading, auto differentiation, and more. Inspired by [PyTorch](https://pytorch.org/) and Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).
- `matcha.engine` Main code for tensor operations, functions, threads, and differentiation.
- `matcha.nn` Contains various useful neural network modules and loss functions.
- `matcha.optim` Contains various optimization algorithms.
- `matcha.dataset` Pre-built and ready to train tensor datasets.
- `matcha.utils` Plotting, tensor visualizations, and more!
- `matcha.edu` A scalar version of matcha useful for understanding fundamental neural networks concepts (wip).


## Example
Here is a simple example of a Multi-Layer Perceptron built in matcha learning to fit the `SineWave` toy dataset.



### Code
```Java
import matcha.datasets.toy.SineWave;
import matcha.engine.Tensor;
import matcha.nn.*;
import matcha.nn.Module;
import matcha.optim.*;
import matcha.utils.*;
import matcha.utils.math.LinAlg;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

public class Example{
    public static void main(String[] args) throws Exception{
        // constructing a simple network
        List<Module> layers = new ArrayList<>();
        layers.add(new Linear(1, 16));
        layers.add(new ReLU());
        layers.add(new Linear(16, 16));
        layers.add(new ReLU());
        layers.add(new Linear(16, 1));

        Sequential model = new Sequential(layers);

        // prints model information, such as layers and dimensions
        System.out.println(model);
```
```
        Sequential(
        Linear(in_features=1, out_features=16, bias=true)
        ReLU()
        Linear(in_features=16, out_features=16, bias=true)
        ReLU()
        Linear(in_features=16, out_features=1, bias=true)
        )
```
```Java
        // initialize sine dataset
        SineWave sineData = new SineWave();
        // train model for 20 epochs
        int epochs = 20;

        // optimizer and loss function
        Optimization optim = new SGD(model.parameters(), 0.05); // 0.05 learning rate
        Loss lossFn = new MSELoss();    

        // train loop
        for(int i = 0; i < epochs; i++){
            double lossSum = 0; // running loss
            for(List<Tensor> batch : sineData){
                // each batch is a list of an input and target tensor
                Tensor inputs = batch.get(0);
                Tensor targets = batch.get(1);

                // zero gradients for every batch!!
                optim.zeroGrad();

                // make batch predictions
                Tensor outputs = model.forward(inputs);

                // calculate loss and compute gradients
                Tensor loss = lossFn.loss(outputs, targets);
                loss.backward();

                // apply gradients
                optim.step();

                lossSum += loss.data()[0];
            }
            if (i % 5 == 4 || i == 0)
                System.out.println("Epoch: " + (i + 1) + ", average loss: " + (lossSum / 4.0));
        }
```
```
        Epoch: 1, average loss: 5.252128635146163
        Epoch: 5, average loss: 0.597272164197843
        Epoch: 10, average loss: 0.47446241197420985
        Epoch: 15, average loss: 0.453805811609508
        Epoch: 20, average loss: 0.43520514472577043
```
```Java
        // initialize new figure
        jPlot plt = new jPlot();

        // uncomment to plot dataset samples
        // for(List<Tensor> batch : sineData){
        //     plt.scatter(batch.get(0), batch.get(1));
        // }


        // generate and plot true sine values on the figure
        double[] xs = LinAlg.arange(-Math.PI, Math.PI, 0.01);
        plt.plot(xs, Arrays.stream(xs).map(x -> Math.sin(x)).toArray());

        // change color of scatter markers using a configuration mapping
        Map<String, Object> plotConfig = new HashMap<>();
        plotConfig.put("fill_color", "#55a630");

        // plot post-train predictions across x axis on the figure
        for(double x : LinAlg.arange(-Math.PI, Math.PI, 0.05)){
            Tensor t_X = new Tensor(new int[]{1, 1}, new double[]{x});
            Tensor t_Y = model.forward(t_X);
            plt.scatter(t_X, t_Y, plotConfig);
        }

        // display figure
        plt.show();
    }
}
```
```
See figures above for plot outputs.
```
