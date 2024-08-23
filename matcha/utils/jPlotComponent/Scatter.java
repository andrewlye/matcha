package matcha.utils.jPlotComponent;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.HashMap;
import java.util.Map;

import java.awt.Color;

import matcha.utils.DefaultPlotConfig;
import matcha.utils.jPlot;

/**
 * Scatter plot component. Draws a parametrized scatter on top of the figure.
 */
public class Scatter extends PlotComponent{
    private jPlot m_plt;
    private double[] m_x;
    private double[] m_y;

    public Scatter(jPlot plt, double[] xs, double ys[], Map<String, ?> config){
        m_plt = plt;
        m_x = xs;
        m_y = ys;

        updateConfig(config);
    }

    private void scatter(Graphics g){
        Graphics2D g2d = (Graphics2D) g;
        int mSize = (int) m_config.get("marker_size");
        if ((boolean) m_config.get("fill")){
            g2d.setColor(Color.decode((String) m_config.get("fill_color")));
            for(int i = 0; i < m_x.length; i++) g2d.fillOval(m_plt.xToPX(m_x[i]) - mSize/2, m_plt.yToPX(m_y[i]) - mSize / 2, mSize, mSize);
        } 

        if ((boolean) m_config.get("outline")){
            g2d.setColor(Color.decode((String) m_config.get("outline_color")));
            for(int i = 0; i < m_x.length; i++){
                g2d.drawOval(m_plt.xToPX(m_x[i]) - mSize/2, m_plt.yToPX(m_y[i]) - mSize / 2, mSize, mSize);
            }
        }
    }

    @Override
    public Map<String, Object> init(){
        Map<String, Object> config = new HashMap<String, Object>();
        config.put("fill", true);
        config.put("fill_color", DefaultPlotConfig.MARKER_FILL);
        config.put("outline", true);
        config.put("outline_color", "#000000");
        config.put("marker_size", DefaultPlotConfig.MARKER_SIZE);
        
        return config;
    }

    @Override
    public void paint(Graphics g){
        scatter(g);
    }
}

