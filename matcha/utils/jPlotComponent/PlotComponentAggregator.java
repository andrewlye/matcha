package matcha.utils.jPlotComponent;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.Queue;

import javax.swing.JComponent;

/**
 * Final JComponent added to the JFrame. Paints all PlotComponent layers.
 */
public class PlotComponentAggregator extends JComponent{
    Queue<PlotComponent> m_layers;

    public PlotComponentAggregator(Queue<PlotComponent> layers){
        m_layers = layers;
    }

    public void paint(Graphics g){
        while(!m_layers.isEmpty()){
            m_layers.poll().paint(g);
        }
    } 

}
