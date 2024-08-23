package matcha.utils.jPlotComponent;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;

import javax.swing.JComponent;
import java.awt.FontMetrics;

import matcha.utils.DefaultPlotConfig;
import matcha.utils.jPlot;
import matcha.utils.math.LinAlg;

/**
 * Axis component for jPlot figures. Draws x and y axis, ticks, and labels.
 */
public class Axis extends PlotComponent{
    private jPlot m_plt;

    public Axis(jPlot plt, Map<String, ?> config){
        m_plt = plt;
        updateConfig(config);
    }

    public void drawAxis(Graphics g){
        Graphics2D g2d = (Graphics2D) g;
        int xShift = (int) ((0 - m_plt.xStart()) / (m_plt.xEnd() - m_plt.xStart()) * (m_plt.xEndPX() - m_plt.xStartPX()));
        int yShift = (int) ((0 - m_plt.yStart()) / (m_plt.yEnd() - m_plt.yStart()) * (m_plt.yEndPX() - m_plt.yStartPX()));
        xShift = Math.max(xShift, 0);
        yShift = Math.max(yShift, 0);
        // vertical
        g2d.drawLine(m_plt.xStartPX() + xShift, m_plt.yStartPX(), m_plt.xStartPX() + xShift, m_plt.yEndPX());
        // horizontal
        g2d.drawLine(m_plt.xStartPX(), m_plt.yEndPX() - yShift, m_plt.xEndPX(), m_plt.yEndPX() - yShift);
    }

    public void drawTicks(Graphics g){
        Graphics2D g2d = (Graphics2D) g;
        FontMetrics fm = g2d.getFontMetrics();
        int xShift = (int) ((0 - m_plt.xStart()) / (m_plt.xEnd() - m_plt.xStart()) * (m_plt.xEndPX() - m_plt.xStartPX()));
        int yShift = (int) ((0 - m_plt.yStart()) / (m_plt.yEnd() - m_plt.yStart()) * (m_plt.yEndPX() - m_plt.yStartPX()));
        xShift = Math.max(xShift, 0);
        yShift = Math.max(yShift, 0);

        int[] xTicks = Arrays.stream(LinAlg.arange(m_plt.xStartPX(), m_plt.xEndPX(), (m_plt.xEndPX() - m_plt.xStartPX()) / m_plt.xTicks())).mapToInt(x -> (int) x).toArray();
        for(int i = 0; i < xTicks.length; i++){
            g2d.drawLine(xTicks[i], m_plt.yEndPX() - yShift + (int) m_config.get("tick_size"), xTicks[i], m_plt.yEndPX() - yShift - (int) m_config.get("tick_size"));
            double currentX = m_plt.xStart() + (m_plt.xEnd() - m_plt.xStart()) * i / m_plt.xTicks();
            String str_currentX = String.format("%.2f", currentX);
            g2d.drawString(str_currentX, xTicks[i] - fm.stringWidth(str_currentX) / 2, m_plt.yEndPX() - yShift + (int) m_config.get("xlabel_offset"));
        }
        
        int[] yTicks = Arrays.stream(LinAlg.arange(m_plt.yStartPX(), m_plt.yEndPX(), (m_plt.yEndPX() - m_plt.yStartPX()) / m_plt.yTicks())).mapToInt(x -> (int) x).toArray();
        for(int i = 0; i < yTicks.length; i++){
            g2d.drawLine(m_plt.xStartPX() + xShift + (int) m_config.get("tick_size"), yTicks[i], m_plt.xStartPX() + xShift - (int) m_config.get("tick_size"), yTicks[i]);
            double currentY = m_plt.yStart() + (m_plt.yEnd() - m_plt.yStart()) * (yTicks.length - 1 - i) / m_plt.yTicks();
            String str_currentY = String.format("%.2f", currentY);
            g2d.drawString(str_currentY, m_plt.xStartPX() + xShift - (int) m_config.get("ylabel_offset"), yTicks[i] - fm.getHeight() / 2 + fm.getAscent());
        }
    }

    @Override
    public void paint(Graphics g){
        drawAxis(g);
        drawTicks(g);
    }

    @Override
    public Map<String, Object> init(){
        Map<String, Object> config = new HashMap<>();
        config.put("tick_size", DefaultPlotConfig.TICK_SIZE);
        config.put("xlabel_offset", DefaultPlotConfig.XLABEL_OFFSET);
        config.put("ylabel_offset", DefaultPlotConfig.YLABEL_OFFSET);

        return config;
    }
}
