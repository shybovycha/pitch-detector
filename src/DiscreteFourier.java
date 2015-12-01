import java.util.HashMap;
import java.util.Map;

/**
 * Created by shybovycha on 01/12/15.
 */
public class DiscreteFourier {
    public static Map<Double, Double> transform(double[] x, int n, double T, double h, double sampleRate) {
        Map<Double, Double> res = new HashMap<>();

        double f[] = new double[n / 2];

        for (int j = 0; j < n / 2; j++) {
            double firstSum = 0;
            double secondSum = 0;

            for (int k = 0; k < n; k++) {
                double coeff = ((2 * Math.PI) / n) * (j * k);

                firstSum +=  x[k] * Math.cos(coeff);
                secondSum += x[k] * Math.sin(coeff);
            }

            f[j] = Math.abs(Math.sqrt(Math.pow(firstSum, 2) + Math.pow(secondSum, 2)));

            double amplitude = 2 * f[j]/n;
            double frequency = j * h / T * sampleRate;
            // double phase = Math.atan2(secondSum, firstSum);

            res.put(frequency, amplitude);
        }

        return res;
    }
}
