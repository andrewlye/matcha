package matcha.nn;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Tensor;
import matcha.nn.Initializer;

public class Linear extends Module{

    private Tensor wandb;
    private int in_features;
    private int out_features;

    public Linear(int in_features, int out_features){
        this.in_features = in_features;
        this.out_features = out_features;
        
        try {
            wandb = new Tensor(new int[]{in_features, out_features}, true);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Initializer.uniform(wandb);
    }

    public Linear(int in_features, int out_features, boolean gradEnabled){
        this.in_features = in_features;
        this.out_features = out_features;
        
        try {
            wandb = new Tensor(new int[]{in_features, out_features}, gradEnabled);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Initializer.uniform(wandb);
    }

    @Override
    public Tensor forward(Tensor x) throws Exception {
        return x.matmul(wandb, false);
    }

    @Override
    public List<Tensor> parameters(){
        List<Tensor> params = new ArrayList<>();
        params.add(wandb);
        return params;
    }

    public String toString(){
        return "Linear(in_features=" + in_features + ", out_features=" + out_features + ")";
    }
}
