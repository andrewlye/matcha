package matcha.engine.threads;

import java.util.Arrays;
import java.util.concurrent.atomic.DoubleAdder;

import matcha.engine.DataRepresentation;
import matcha.engine.Tensor;
import matcha.utils.math.LinAlg;

public class matMulThread {
    static int MAX_THREADS = 8;
    DoubleAdder[] syncData;
    int[] shape;
    Tensor t_a;
    Tensor t_b;
    int step;
    char part;

    class Worker extends Thread {
        int i;
        int rStart, rEnd, cStart, cEnd, kStart, kEnd;

        Worker(int i){
            this.i = i;

            rStart = cStart = kStart = 0;
            rEnd = t_a.shape()[0];
            cEnd = t_b.shape()[1];
            kEnd = t_b.shape()[0];

            if (part == 'r'){
                rStart = i * step;
                rEnd = Math.min(i * step + step, t_a.shape()[0]);
            } else if (part == 'c') {
                cStart = i * step;
                cEnd = Math.min(i * step + step, t_b.shape()[1]);
            } else {
                kStart = i * step;
                kEnd = Math.min(i * step + step, t_b.shape()[0]);
            }

            //System.out.println("Worker " + i + " rStart: " + rStart + " rEnd: " + rEnd + " cStart: " + cStart + " cEnd: " + cEnd + " kStart: " + kStart + " kEnd: " + kEnd);
        }

        public void run(){
            for(int r = rStart; r < rEnd; r++){
                for(int c = cStart; c < cEnd; c++){
                    for(int k = kStart; k < kEnd; k++){
                        double x_i = t_a.data()[storageIndex(t_a.shape(), new int[]{r, k}, t_a.dataLayout)] * t_b.data()[storageIndex(t_b.shape(), new int[]{k, c}, t_b.dataLayout)];
                        syncData[storageIndex(shape, new int[]{r, c}, t_a.dataLayout)].add(x_i);
                    }
                }
            }
        }

        private int storageIndex(int[] shape, int[] idxs, DataRepresentation layout){
            switch (layout) {
            case ROW_MAJOR: 
            default:
                return LinAlg.rmo(shape, idxs);
            }
        }
    }

    public matMulThread(Tensor t_a, Tensor t_b, int[] shape){
        syncData = new DoubleAdder[t_a.shape()[0] * t_b.shape()[1]];
        for(int i = 0; i < syncData.length; i++) syncData[i] = new DoubleAdder();

        this.shape = shape;
        this.t_a = t_a;
        this.t_b = t_b;
        
        // thread along the largest dimension for largest increase in efficiency.
        if (t_a.shape()[0] >= t_b.shape()[1] && t_a.shape()[0] >= t_b.shape()[0]){
            part = 'r';
            step = (int) Math.ceil((double) t_a.shape()[0] / MAX_THREADS);
        }
        else if (t_b.shape()[1] >= t_a.shape()[0] && t_b.shape()[1] >= t_b.shape()[0]){
            part = 'c';
            step = (int) Math.ceil((double) t_b.shape()[1] / MAX_THREADS);
        }
        else{
            part = 'k';
            step = (int) Math.ceil((double) t_b.shape()[0] / MAX_THREADS);
        }
    }

    public double[] matMul(){
        Thread threads[] = new Thread[MAX_THREADS];
        for (int i = 0; i < MAX_THREADS; i++){
            threads[i] = new Thread(new Worker(i));
            threads[i].start();
        }

        for(int i = 0; i < MAX_THREADS; i++){
            try {
                threads[i].join();
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        return Arrays.stream(syncData).mapToDouble(x -> x.sum()).toArray();
    }
}
