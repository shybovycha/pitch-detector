/**
 * Created by shybovycha on 01/12/15.
 */
public class WindowFn {
    public static double gauss(double n, double frameSize) {
        final double Q = 0.5;

        double a = (frameSize - 1) / 2;
        double t = (n - a) / (Q * a);

        t = t * t;

        return Math.exp(-t / 2);
    }

    public static double cosine(double n, double frameSize) {
        return 0.54 - 0.46 * Math.cos((2.0 * Math.PI * n) / (frameSize - 1.0));
    }
}
