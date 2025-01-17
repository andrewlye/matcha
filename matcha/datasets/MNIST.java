package matcha.datasets;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import javax.swing.JComponent;
import javax.swing.JFrame;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;

import matcha.engine.Tensor;

public class MNIST extends Dataset {
    private List<List<Tensor>> samples;
    private int batchSize;

    public static final int DISPLAY_WIDTH = 300;
    public static final int DISPLAY_HEIGHT = 300;
    public static final int IMG_LENGTH = 28;

    public MNIST(String cache) {
        this(cache, -1, 1, true);
    }

    public MNIST(String cache, int nSamples) {
        this(cache, nSamples, 1, true);
    }

    public MNIST(String cache, int nSamples, int batchSize) {
        this(cache, nSamples, batchSize, true);
    }

    public MNIST(String cache, int nSamples, int batchSize, boolean shuffle){
        if (batchSize <= 0) throw new IllegalArgumentException("Error: batch size must be > 0.");
        this.batchSize = batchSize;

        samples = new ArrayList<>();
        FileReader fr = null;
        try {
            fr = new FileReader(cache);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        String line;
        BufferedReader br = new BufferedReader(fr);
        try {
            line = br.readLine();
            ArrayList<ArrayList<double[]>> data = new ArrayList<>();
            while((line = br.readLine()) != null) {
                    double[] XData = new double[784], yData = new double[1];
                    String[] tokens = line.split(",");
                    yData[0] = Double.parseDouble(tokens[0]);
                    for (int j = 1; j < tokens.length; j++) XData[j-1] = Double.parseDouble(tokens[j]);
                    var sample = new ArrayList<double[]>();
                    sample.add(XData); sample.add(yData);
                    data.add(sample);
            }
            br.close();

            if (shuffle) Collections.shuffle(data);
            
            nSamples = (nSamples > 0) ? nSamples : data.size();
            for (int i = 0; i < nSamples; i += batchSize) {
                ArrayList<Double> XBatchData = new ArrayList<>(), yBatchData = new ArrayList<>();
                for (int j = 0; j < batchSize && i + j < nSamples; j++) {
                    var sample = data.get(i + j);
                    for (double d : sample.get(0)) XBatchData.add(d);
                    yBatchData.add(sample.get(1)[0]);
                }
                var sample = new ArrayList<Tensor>();
                if (batchSize == 1) {
                    sample.add(new Tensor(new int[]{28, 28}, XBatchData.stream().mapToDouble(x -> x).toArray()));
                    sample.add(new Tensor(new int[]{1}, yBatchData.stream().mapToDouble(x -> x).toArray()));
                } else {
                    sample.add(new Tensor(new int[]{yBatchData.size(), 28, 28}, XBatchData.stream().mapToDouble(x -> x).toArray()));
                    sample.add(new Tensor(new int[]{yBatchData.size(), 1}, yBatchData.stream().mapToDouble(x -> x).toArray()));
                }
                
                samples.add(sample);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void reshapeX(int... shape) {
        reshape(true, shape);
    }

    public void reshapeY(int... shape) {
        reshape(false, shape);
    }

    private void reshape(boolean isX, int... shape) {
        int k = (isX) ? 0 : 1;
        if (batchSize == 1) {
            for (var sample : samples) sample.get(k).reshape(shape);
            return;
        }

        for (var sample : samples) {
            int[] bShape = new int[shape.length+1];
            bShape[0] = sample.get(k).shape()[0];
            for (int i = 0; i < shape.length; i++) {
                bShape[i+1] = shape[i];
            }
            sample.get(k).reshape(bShape);
        }
    }

    @Override
    public List<Tensor> get(int i) {
        return samples.get(i);
    }

    @Override
    public int size() {
        return samples.size();
    }

    public void show(Tensor X) { show(X, 25); }

    public void show(Tensor X, int pixelSize) {
        if (pixelSize <= 0) throw new IllegalArgumentException("Error: pixel size must be >0.");
        if (X.size() != IMG_LENGTH * IMG_LENGTH) throw new IllegalArgumentException("Error: input tensor must be a single 784-element tensor.");
        JFrame frame = new JFrame();
        frame.setSize(DISPLAY_WIDTH, DISPLAY_HEIGHT);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); 
        frame.add(new Digit(X, pixelSize));
        frame.setVisible(true);
    }
    
}

class Digit extends JComponent {
    private int WIDTH;
    private Tensor X;
    public Digit(Tensor X, int width) {
        this.X = X;
        WIDTH = width;
    }
    
    public void paint(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setColor(Color.BLUE);
        for (int i = 0; i < MNIST.IMG_LENGTH; i++) {
            for (int j = 0; j < MNIST.IMG_LENGTH; j++) {
                int intensity = (int) X.get(i, j);
                Color c = new Color(0, 0, 0, 255 - intensity);
                g2d.setColor(c);
                g2d.fillRect(j * WIDTH, i * WIDTH, WIDTH, WIDTH);
            }
        }
    }
}
