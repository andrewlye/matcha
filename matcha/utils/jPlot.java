package matcha.utils;

import javax.swing.JComponent;
import javax.swing.JFrame;
import java.util.Queue;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import matcha.utils.DefaultPlotConfig;
import matcha.utils.jPlotComponent.*;

/**
 * Simple plotter class used for visualizations of data.
 */
public class jPlot {
    private JFrame frame;
    private Queue<JComponent> m_components; // figure layers are stored as a FIFO ordering of components


    private int m_width; // window width
    private int m_height; // window height

    // start and end positions (in px) of the x and y axis
    private int m_xStartpx;
    private int m_xEndpx;
    private int m_yStartpx;
    private int m_yEndpx;

    // start and end positions of the x and y axis
    private double m_xStart = DefaultPlotConfig.START_X;
    private double m_xEnd = DefaultPlotConfig.END_X;
    private double m_yStart = DefaultPlotConfig.START_Y;
    private double m_yEnd = DefaultPlotConfig.END_Y;

    // number of ticks to draw per axis
    private int m_xTicks = DefaultPlotConfig.X_TICKS;
    private int m_yTicks = DefaultPlotConfig.Y_TICKS;

    // default window initialization
    public jPlot(){ this(DefaultPlotConfig.WIDTH, DefaultPlotConfig.HEIGHT); }

    public jPlot(int width, int height){
        m_width = width;
        m_height = height;
        frame = new JFrame();
        frame.setSize(width, height);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);  
        m_components = new LinkedList<>();
        init();
        // we always add an axis as the first component to our figure
        m_components.add(new Axis(this, DefaultPlotConfig.XLABEL_OFFSET, DefaultPlotConfig.YLABEL_OFFSET, DefaultPlotConfig.TICK_SIZE));
    }

    /**
     * Plots y versus x as lines.
     * @param xs x values.
     * @param ys y values.
     * @param configs dict of configs to provide 
     * param -> type : default
     * [
     * "color" -> java.awt.Color : BLACK. Line color.
     * "stroke" -> float : DefaultPlotConfig.LINE_STROKE. Line width.
     * ].
     */
    public void plot(double[] xs, double[] ys, Map<String, Object> configs){
        if (xs.length != ys.length) throw new IllegalArgumentException("Error: x and y arrays should be the same length.");
        resize(xs, ys);
        m_components.add(new LinePlot(this, xs, ys, configs));
    }

    // default plot
    public void plot(double[] xs, double[] ys) { plot(xs, ys, new HashMap<String, Object>()); }

    /**
     * Plots y versus x as dots/markers.
     * @param xs x values.
     * @param ys y values.
     * @param configs configs to provide 
     * param -> type : default
     * [
     *  "fill" -> boolean : true. If true fills in markers with fill_color.
     *  "fill_color" -> java.awt.Color : DefaultPlotConfig.MARKER_FILL. Fill color.
     *  "outline" -> boolean : true. If true, draws outline around markers.
     *  "outline_color" -> java.awt.Color : BLACK. Outline color.
     *  "marker_size" -> int : DefaultPlotConfig.MARKER_SIZE. Size of markers to draw.
     * ]
     */
    public void scatter(double[] xs, double[] ys, Map<String, Object> configs){
        if (xs.length != ys.length) throw new IllegalArgumentException("Error: x and y arrays should be the same length.");
        resize(xs, ys);
        
        m_components.add(new Scatter(this, xs, ys, configs));
    }

    // default scatter
    public void scatter(double[] xs, double[] ys) { scatter(xs, ys, new HashMap<String, Object>()); }

    /**
     * Constructs and shows the plot/figure.
     */
    public void show(){
        frame.add(new PlotComponentAggregator(m_components));
        frame.setVisible(true);  
        //frame.setResizable(false);
    }

    /**
     * Auto-resizes boundaries of figure given x and y data.
     */
    private void resize(double[] xs, double[] ys){
        m_xStart = Math.min(m_xStart, Arrays.stream(xs).min().orElse(Double.MAX_VALUE));
        m_xEnd = Math.max(m_xEnd, Arrays.stream(xs).max().orElse(Double.MIN_VALUE));
        m_yStart = Math.min(m_yStart, Arrays.stream(ys).min().orElse(Double.MAX_VALUE));
        m_yEnd = Math.max(m_yEnd, Arrays.stream(ys).max().orElse(Double.MIN_VALUE));
    }

    // some initialization stuff for margins.
    private void init(){
        float xMargin = 0.1f;
        float yMargin = 0.1f;
        m_xStartpx = (int) (m_width * xMargin);
        m_xEndpx = m_width - m_xStartpx;
        m_yStartpx = (int) (m_height * yMargin);
        m_yEndpx = m_height - m_yStartpx;
    }

    /**
     * Returns the x px coordinate of the x value on the plot.
     * @param xVal to value to convert to px.
     * @return the x px position of the input value.
     */
    public int xToPX(double xVal){
        return (int) ( (xVal - m_xStart) / (m_xEnd - m_xStart) * (m_xEndpx - m_xStartpx) ) + m_xStartpx;
    }

    /**
     * Returns the y px coordinate of the y value on the plot.
     * @param xVal to value to convert to px.
     * @return the y px position of the input value.
     */
    public int yToPX(double yVal){
        return m_yEndpx - (int) ( (yVal - m_yStart) / (m_yEnd - m_yStart) * (m_yEndpx - m_yStartpx));
    }

    // Getters

    public int xStartPX() { return m_xStartpx; }
    public int xEndPX() { return m_xEndpx; }
    public int yStartPX() { return m_yStartpx; }
    public int yEndPX() { return m_yEndpx; }
    
    public double xStart() { return m_xStart; }
    public double xEnd() { return m_xEnd; }
    public double yStart() { return m_yStart; }
    public double yEnd() { return m_yEnd; }

    public int xTicks() { return m_xTicks; }
    public int yTicks() { return m_yTicks; }
}
