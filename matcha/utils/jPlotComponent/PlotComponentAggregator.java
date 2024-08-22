package matcha.utils.jPlotComponent;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.Queue;

import javax.swing.JComponent;

public class PlotComponentAggregator extends JComponent{
    Queue<JComponent> m_layers;

    public PlotComponentAggregator(Queue<JComponent> layers){
        m_layers = layers;
    }

    public void paint(Graphics g){
        while(!m_layers.isEmpty()){
            m_layers.poll().paint(g);
        }
    } 

}
