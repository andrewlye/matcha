package matcha.engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
public class Value{
    private double data;
    private List<Value> prev; // stores composite values for backprop
    private double grad;
    private Backward backward; // stores derivative function for backprop

    public Value(double data){
        this.data = data;
        this.grad = 0.0;
        this.backward = () -> {};
    }

    public Value(int data){
        this.data = data;
        this.grad = 0.0;
        this.backward = () -> {};
    }

    public Value(float data){
        this.data = data;
        this.grad = 0.0;
        this.backward = () -> {};
    }

    public Value(double data, List<Value> children){
        this.data = data;
        this.prev = children;
        this.grad = 0.0;
        this.backward = () -> {};
    }

    public Value add(Value x){
        List<Value> children = new ArrayList<>();
        children.add(this);
        children.add(x);   

        Value out = new Value(data + x.data, children);
        Backward back = () -> {this.grad += 1.0 * out.grad; x.grad += 1.0 * out.grad;};
        out.backward = back;

        return out;
    }

    public Value mul(Value x){
        List<Value> children = new ArrayList<>();
        children.add(this);
        children.add(x);

        Value out = new Value(data * x.data, children);
        Backward back = () -> {this.grad += x.data * out.grad; x.grad += this.data * out.grad;};
        out.backward = back;

        return out;
    }

    public Value sub(Value x){
        return this.add(x.mul(new Value(-1)));
    }

    public Value pow(double x){
        List<Value> children = new ArrayList<>();
        children.add(this);

        Value out = new Value(Math.pow(data, x), children);
        Backward back = () -> {this.grad += x*Math.pow(data, x-1) * out.grad;};
        out.backward = back;

        return out;
    }

    public Value exp(){
        List<Value> children = new ArrayList<>();
        children.add(this);

        Value out = new Value(Math.exp(data), children);
        Backward back = () -> {this.grad += out.data * out.grad;};
        out.backward = back;

        return out;
    }

    public Value tanh(){
        List<Value> children = new ArrayList<>();
        children.add(this);

        double tanh = Math.tanh(data);
        Value out = new Value(tanh, children);
        Backward back = () -> {this.grad = (1-(out.data*out.data)) * out.grad;};
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

    public void increment(double x){
        this.data += x;
    }

    public void backward(){
        List<Value> ordering = new ArrayList<>();
        buildTopo(this, new HashSet<>(), ordering);
        Collections.reverse(ordering);
        
        this.grad = 1.0;
        for(Value val : ordering){
            val.backward.pass();
        }
    }

    public void buildTopo(Value parent, Set<Value> visited, List<Value> ordering){
        if (!visited.contains(parent)){
            visited.add(parent);
            if (parent.prev != null){
                for(Value child : parent.prev){
                    buildTopo(child, visited, ordering);
                }
            }
            ordering.add(parent);
        }
    }

    public List<Value> children(){
        return prev;
    }

    public void setChildren(List<Value> prev){
        this.prev = prev;
    }

    @Override
    public String toString(){
        return "Value(data=" + data + ", grad=" + grad + ")";
    }
}
