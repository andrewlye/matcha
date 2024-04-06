import java.util.ArrayList;
import java.util.List;

public class Value{
    private double data;
    private List<Value> prev; // stores composite values for backprop
    private double grad;
    private Backward backward; // stores derivative function for backprop

    public Value(double data){
        this.data = data;
        this.grad = 0.0;
    }

    public Value(int data){
        this.data = data;
        this.grad = 0.0;
    }

    public Value(float data){
        this.data = data;
        this.grad = 0.0;
    }

    public Value(double data, List<Value> children){
        this.data = data;
        this.prev = children;
        this.grad = 0.0;
    }

    public Value(double data, List<Value> children, Backward backward){
        this.data = data;
        this.prev = children;
        this.grad = 0.0;
        this.backward = backward;
    }

    public Value add(Value x){
        List<Value> children = new ArrayList<>();
        children.add(this);
        children.add(x);   

        Value out = new Value(data + x.data(), children);
        Backward back = () -> {this.grad += 1.0 * out.grad; x.grad += 1.0 * out.grad;};
        out.backward = back;

        return out;
    }

    public Value mul(Value x){
        List<Value> children = new ArrayList<>();
        children.add(this);
        children.add(x);

        Value out = new Value(data + x.data(), children);
        Backward back = () -> {this.grad += x.data * out.grad; x.grad += this.data * out.grad;};
        out.backward = back;

        return out;
    }

    public double data(){
        return data;
    }

    public double grad(){
        return grad;
    }

    public void setGradient(double grad){
        this.grad = grad;
    }

    public void backward(){
        this.backward.pass();
    }

    public List<Value> children(){
        return prev;
    }

    @Override
    public String toString(){
        return data + "";
    }
}
