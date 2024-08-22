package matcha.utils.jPlotComponent;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.Arrays;

import javax.swing.JComponent;
import java.awt.FontMetrics;

import matcha.utils.jPlot;
import matcha.utils.math.LinAlg;

public class Axis extends JComponent{
    private jPlot m_plt;
    private int m_xLabelOffset;
    private int m_yLabelOffset;
    private int m_tickSize;

    public Axis(jPlot plt, int xLabelOffset, int yLabelOffset, int tickSize){
        m_plt = plt;
        m_xLabelOffset = xLabelOffset;
        m_yLabelOffset = yLabelOffset;
        m_tickSize = tickSize;
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
            g2d.drawLine(xTicks[i], m_plt.yEndPX() - yShift + m_tickSize, xTicks[i], m_plt.yEndPX() - yShift - m_tickSize);
            double currentX = m_plt.xStart() + (m_plt.xEnd() - m_plt.xStart()) * i / m_plt.xTicks();
            String str_currentX = String.format("%.2f", currentX);
            g2d.drawString(str_currentX, xTicks[i] - fm.stringWidth(str_currentX) / 2, m_plt.yEndPX() - yShift + m_xLabelOffset);
        }
        
        int[] yTicks = Arrays.stream(LinAlg.arange(m_plt.yStartPX(), m_plt.yEndPX(), (m_plt.yEndPX() - m_plt.yStartPX()) / m_plt.yTicks())).mapToInt(x -> (int) x).toArray();
        for(int i = 0; i < yTicks.length; i++){
            g2d.drawLine(m_plt.xStartPX() + xShift + m_tickSize, yTicks[i], m_plt.xStartPX() + xShift - m_tickSize, yTicks[i]);
            double currentY = m_plt.yStart() + (m_plt.yEnd() - m_plt.yStart()) * (yTicks.length - 1 - i) / m_plt.yTicks();
            String str_currentY = String.format("%.2f", currentY);
            g2d.drawString(str_currentY, m_plt.xStartPX() + xShift - m_yLabelOffset, yTicks[i] - fm.getHeight() / 2 + fm.getAscent());
        }
    }

    public void paint(Graphics g){
        drawAxis(g);
        drawTicks(g);
    }
}
