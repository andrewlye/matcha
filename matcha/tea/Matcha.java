package matcha.tea;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;

import matcha.datasets.Dataset;
import matcha.engine.Tensor;
import matcha.nn.Loss;
import matcha.nn.Module;
import matcha.optim.Optimization;
import matcha.optim.SGD;
import matcha.utils.jPlot;
import matcha.utils.math.LinAlg;

/** WORK IN PROGRESS! Some methods may be written very poorly.
 * General purpose trainer class to wrap around a model, dataset, etc.
 */
public class Matcha {
    public Module model;
    public Optimization optimizer;
    public Loss loss;
    public Dataset trainData, testData;
    private HashMap<String, String> configs;
    public static final int DEFAULT_EPOCHS = 10;
    public static final double DEFAULT_LR = 0.001;
    public static final int BAR_LENGTH = 200;

    private ArrayList<Double> lossCurve = null;

    public Matcha(Module model) { this(model, null, null, null); }
    public Matcha(Module model, Loss loss) { this(model, null, null, loss); }
    public Matcha(Module model, Dataset trainData, Loss loss) { this(model, trainData, null, loss); }
    public Matcha(Module model, Dataset trainData, Dataset testData, Loss loss){
        this.model = model;
        this.trainData = trainData;
        this.testData = testData;
        this.loss = loss;
        this.optimizer = new SGD(model.parameters(), DEFAULT_LR);
        this.configs = new HashMap<>();
        init();
    }

    public void config(String... c){
        for (String s : c) {
            String[] toks = s.split("=");
            if (toks.length != 2 || !configs.containsKey(toks[0])) throw new IllegalArgumentException("Error: config option " + s + " is invalid.");
            configs.put(toks[0], toks[1]);
        }
    }

    public void brew() { brew(DEFAULT_EPOCHS); }

    public void brew(int epochs){
        if (model == null) throw new IllegalStateException("Error: please specify a model.");
        if (optimizer == null) throw new IllegalStateException("Error: please specify an optimizer.");
        if (loss == null) throw new IllegalStateException("Error: please specify a loss function.");
        if (trainData == null) throw new IllegalStateException("Error: please specify a training dataset.");

        double inc = (double) BAR_LENGTH / epochs;
        lossCurve = new ArrayList<>();
        long startTime = System.nanoTime();
        System.out.println("TRAINING LOG:");
        System.out.println("-------------");
        for (int i = 0; i <= epochs; i++) {
            double runningLoss = 0.;
            if (i > 0) {
                for (List<Tensor> sample : trainData) {
                    Tensor X = sample.get(0), y = sample.get(1);
                    optimizer.zeroGrad();
                    Tensor outputs = model.forward(X);
                    Tensor tLoss = loss.loss(outputs, y);
                    tLoss.backward();
                    optimizer.step();
                    runningLoss += tLoss.sum();
                }
                lossCurve.add(runningLoss / trainData.size());
            }
			

            boolean print = i > 0 && ((i % Integer.parseInt(configs.get("print_every"))) == 0 || i == epochs);
            StringBuilder sb = new StringBuilder("|");
            String info = "";
            for (int j = 0; j < BAR_LENGTH; j += 4) sb.append((j <= (inc * i)) ? '█' : ' ');
            if (print)
                info = ", loss/epoch: " + String.format(configs.get("loss_format"), (runningLoss / trainData.size()));
            System.out.print(
                sb.toString() 
                + "|" + '(' + (int) (((double) i / epochs) * 100) + "%)" 
                + " Epoch: " + i 
                + info 
                + ((print) ? '\n' : '\r')
            );
		}
        
        long endTime = System.nanoTime();
        long minutes = TimeUnit.MINUTES.convert(endTime - startTime, TimeUnit.NANOSECONDS);
        long ms = TimeUnit.MILLISECONDS.convert(endTime - startTime, TimeUnit.NANOSECONDS) - TimeUnit.MILLISECONDS.convert(minutes, TimeUnit.MILLISECONDS);
        
        System.out.println("Time elapsed: " + minutes + " minutes and " + String.format("%.2f", (double) ms / 1000) + " seconds.");
    };

    public void sip(Dataset testData){
        if (model == null) throw new IllegalStateException("Error: please specify a model.");
        if (loss == null) throw new IllegalStateException("Error: please specify a loss function.");
        if (!configs.get("report_test_acc").equals("true") && !configs.get("report_test_acc").equals("false"))
            throw new IllegalArgumentException(
                "Error: config \"report_test_acc\" must be \"true\" or \"false\" but found \"" + configs.get("report_test_acc") + "\"."
            );

        double runningLoss = 0.;
        for (var sample : testData) {
			Tensor X = sample.get(0), y = sample.get(1);
			Tensor outputs = model.forward(X);
			Tensor tLoss = loss.loss(outputs, y);
            runningLoss += tLoss.sum();
		}
        
        if (!configs.get("test_loss").equals("total") && !configs.get("test_loss").equals("average"))
            throw new IllegalArgumentException(
                "Error: config \"test_loss\" must be \"average\" or \"total\" but found \"" + configs.get("test_loss") + "\"."
                );
        
        System.out.println(
                ((configs.get("test_loss").equals("average")) ? "average loss: " : "total loss: ")
                + String.format(
                    configs.get("loss_format"), 
                    ((configs.get("test_loss").equals("average")) ? (runningLoss / trainData.size()) : runningLoss)
                )
        );

    };

    public ArrayList<Double> lossCurve() {
        return lossCurve;
    }

    public void plot() {
        if (lossCurve == null) throw new IllegalStateException("Error: calling plot() without any prior training run.");
        jPlot plt = new jPlot();
        double[] xs = LinAlg.arange(0, lossCurve.size()-1, 1);
        plt.plot(xs, lossCurve.stream().mapToDouble(x -> x).toArray());
        plt.show();
    }

    private void init() {
        configs.put("loss_format", "%4.4f");
        configs.put("print_every", "5");
        configs.put("test_loss", "average");
        configs.put("report_test_acc", "false");
    }
}
