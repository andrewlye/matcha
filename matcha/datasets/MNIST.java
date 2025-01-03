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
            
            if (shuffle) Collections.shuffle(samples);
            nSamples = (nSamples > 0) ? nSamples : data.size();
            for (int i = 0; i < nSamples; i += batchSize) {
                ArrayList<Double> XBatchData = new ArrayList<>(), yBatchData = new ArrayList<>();
                for (int j = 0; j < batchSize && i + j < nSamples; j++) {
                    var sample = data.get(i + j);
                    for (double d : sample.get(0)) XBatchData.add(d);
                    yBatchData.add(sample.get(1)[0]);
                }
                var sample = new ArrayList<Tensor>();
                sample.add(new Tensor(new int[]{yBatchData.size(), 28, 28}, XBatchData.stream().mapToDouble(x -> x).toArray()));
                sample.add(new Tensor(new int[]{yBatchData.size(), 1}, yBatchData.stream().mapToDouble(x -> x).toArray()));
                samples.add(sample);
            }
        } catch (IOException e) {
            e.printStackTrace();
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

    public void show(Tensor X) {
        JFrame frame = new JFrame();
        frame.setSize(300, 300);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); 
        frame.add(new Digit(X));
        frame.setVisible(true);
    }
    
}

class Digit extends JComponent {
    public static final int WIDTH = 30;
    private Tensor X;
    public Digit(Tensor X) {
        this.X = X;
    }
    public void paint(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setColor(Color.BLUE);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int intensity = (int) X.get(new int[]{i, j});
                Color c = new Color(0, 0, 0, 255 - intensity);
                g2d.setColor(c);
                g2d.fillRect(j * WIDTH, i * WIDTH, WIDTH, WIDTH);
            }
        }
    }
}
