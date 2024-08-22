package matcha.utils.jPlotComponent;

import java.awt.Graphics;
import java.util.Map;

import javax.swing.JComponent;

public abstract class PlotComponent extends JComponent {
    Map<String, Object> m_config;

    public PlotComponent(){
        m_config = init();
    }

    public void updateConfig(Map<String, Object> config){
        for(String key : config.keySet()){
            if (!m_config.containsKey(key)) throw new IllegalArgumentException("Error: " + key + " is an invalid configuration for this component.");
            if (m_config.get(key).getClass() != config.get(key).getClass()) throw new IllegalArgumentException("Error: " + key + " expects a type of " + m_config.get(key).getClass() + " but was provided a " + config.get(key).getClass() + '.');
            m_config.replace(key, config.get(key));
        }
    }

    public abstract Map<String, Object> init();
    public abstract void paint(Graphics g);
}
