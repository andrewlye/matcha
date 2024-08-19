package matcha.engine.threads;

import matcha.engine.DataRepresentation;
import matcha.engine.Tensor;
import matcha.utils.math.LinAlg;

public class matMulThread {
    static int MAX_THREADS = 4;
    double[] data;
    int[] shape;
    Tensor t_a;
    Tensor t_b;
    int step;
    char part;
    private DataRepresentation m_dataLayoutA;
    private DataRepresentation m_dataLayoutB;

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
        }

        public void run(){
            for(int r = rStart; r < rEnd; r++){
                for(int c = cStart; c < cEnd; c++){
                    for(int k = kStart; k < kEnd; k++){
                        data[storageIndex(shape, new int[]{r, c}, m_dataLayoutA)] += t_a.data()[storageIndex(t_a.shape(), new int[]{r, k}, m_dataLayoutA)] * t_b.data()[storageIndex(t_b.shape(), new int[]{k, c}, m_dataLayoutB)];
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
        data = new double[t_a.shape()[0] * t_b.shape()[1]];
        this.shape = shape;
        this.t_a = t_a;
        this.t_b = t_b;
        this.m_dataLayoutA = t_a.dataLayout;
        this.m_dataLayoutB = t_b.dataLayout;
        
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

        return data;
    }
}
