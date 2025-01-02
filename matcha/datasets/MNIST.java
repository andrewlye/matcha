package matcha.datasets;

import java.io.BufferedReader;
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
    public MNIST(String cacheDir) throws IOException {
        this(cacheDir, -1, true);
    }

    public MNIST(String cacheDir, int nSamples) throws IOException {
        this(cacheDir, nSamples, true);
    }

    public MNIST(String cacheDir, int nSamples, boolean shuffle) throws IOException {
        samples = new ArrayList<>();
        
        String line;
        BufferedReader br = new BufferedReader(new FileReader(cacheDir));
        line = br.readLine();
        int i = 0;
        while((line = br.readLine()) != null && (nSamples > 0 && i < nSamples)) {
				String[] tokens = line.split(",");
				double yVal = Double.parseDouble(tokens[0]);
				String[] Xtokens = Arrays.copyOfRange(tokens, 1, tokens.length);
				Tensor y = new Tensor(new int[]{1}, new double[]{yVal});
				Tensor X = new Tensor(new int[]{Xtokens.length}, Arrays.stream(Xtokens).mapToDouble(x -> Double.parseDouble(x)).toArray());
				X.reshape(28, 28);
                var sample = new ArrayList<Tensor>();
                sample.add(X); sample.add(y);
                samples.add(sample);
                i++;
		}
        br.close();
        if (shuffle) Collections.shuffle(samples);
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
