package matcha.utils.jPlotComponent;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.awt.BasicStroke;
import java.awt.Color;

import matcha.utils.DefaultPlotConfig;
import matcha.utils.jPlot;

public class LinePlot extends PlotComponent{
    private jPlot m_plt;
    private double[] m_x;
    private double[] m_y;

    public LinePlot(jPlot plt, double[] xs, double ys[], Map<String, Object> config){
        init();
        m_plt = plt;
        ArrayList<double[]> xyPairs = new ArrayList<>();
        for(int i = 0; i < xs.length; i++) xyPairs.add(new double[]{ xs[i], ys[i]});
        Collections.sort(xyPairs, (a,b) -> Double.compare(a[0], b[0]));
        for(int i = 0; i < xyPairs.size(); i++){
            xs[i] = xyPairs.get(i)[0];
            ys[i] = xyPairs.get(i)[1];
        }
        m_x = xs;
        m_y = ys;

        updateConfig(config);
    }

    private void scatter(Graphics g){
        Graphics2D g2d = (Graphics2D) g;

        g2d.setColor((Color) m_config.get("color"));
        g2d.setStroke(new BasicStroke((float) m_config.get("stroke")));


        for(int i = 0; i < m_x.length-1; i++) g2d.drawLine(m_plt.xToPX(m_x[i]), m_plt.yToPX(m_y[i]), m_plt.xToPX(m_x[i+1]), m_plt.yToPX(m_y[i+1]));
    }

    @Override
    public Map<String, Object> init(){
        Map<String, Object> config = new HashMap<String, Object>();
        config.put("color", Color.BLACK);
        config.put("stroke", DefaultPlotConfig.LINE_STROKE);
        
        return config;
    }

    @Override
    public void paint(Graphics g){
        scatter(g);
    }
}
