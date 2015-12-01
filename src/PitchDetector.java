import javax.sound.sampled.AudioFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by shybovycha on 01/12/15.
 */
public class PitchDetector {
    private double[] frameData;
    private int frameSize;
    private AudioFormat format;
    private Map<Double, Double> spectrum;

    PitchDetector(byte[] frameData, int dataLength, AudioFormat format) {
        this.frameSize = dataLength;
        this.format = format;

        convertByteDataToEndian(frameData);
    }

    public String pitch() {
        double T = frameSize / format.getFrameRate(); // sample length (in seconds)
        int n = (int) (T * format.getSampleRate()) / 2; // number of equidistant points
        double h = T / n; // time between two equidistant points

        applyWindowFn();
        spectrum = DiscreteFourier.transform(frameData, frameSize, T, h, format.getSampleRate());
        applyAntiAliasingFn();

        double F = selectMajorFrequency();

        return PitchNameSelector.findClosest(F);
    }

    private void convertByteDataToEndian(byte[] data) {
        boolean bigEndian = format.isBigEndian();

        frameData = new double[frameSize]; // value of signal at (i * h) time

        // convert each pair of byte values from the byte array to an Endian value
        for (int i = 0; i < frameSize; i += 2) {
            int b1 = data[i];
            int b2 = data[i + 1];
            if (b1 < 0) b1 += 0x100;
            if (b2 < 0) b2 += 0x100;

            int value;

            // store the data based on the original Endian encoding format
            if (!bigEndian)
                value = (b1 << 8) + b2;
            else
                value = b1 + (b2 << 8);

            frameData[i / 2] = (double) value;
        }
    }

    private void applyWindowFn() {
        for (int i = 0; i < frameSize; i++) {
            frameData[i] *= WindowFn.gauss(i, frameSize);
        }
    }

    private void applyAntiAliasingFn() {
        Map<Double, Double> res = new HashMap<>();

        List<Double> keys = new ArrayList<>(spectrum.keySet());

        for (int i = 0; i < keys.size() - 4; i++) {
            double x0 = keys.get(i + 0);
            double x1 = keys.get(i + 1);

            double y0 = spectrum.get(x0);
            double y1 = spectrum.get(x1);

            double a = (y1 - y0) / (x1 - x0);
            double b = y0 - (a * x0);

            // next pair

            double u0 = keys.get((i + 2) + 0);
            double u1 = keys.get((i + 2) + 1);

            double v0 = spectrum.get(u0);
            double v1 = spectrum.get(u1);

            double c = (v1 - v0) / (u1 - u0);
            double d = v0 - (c * u0);

            // interpolate

            double x = (d - b) / (a - c);
            double y = ((a * d) - (b * c)) / (a - c);

            if (y > y0 && y > y1 && y > v0 && y > v1 && x > x0 && x > x1 && x < u0 && x < u1) {
                res.put(x1, y1);
                res.put(x, y);
            } else {
                res.put(x1, y1);
            }
        }

        spectrum = res;
    }

    double selectMajorFrequency() {
        Double res = null;

        for (Map.Entry<Double, Double> e : spectrum.entrySet()) {
            // drop off all the pitches below C (261.63 Hz)
            if (e.getKey() < 260)
                continue;

            if (res == null || e.getValue() > spectrum.get(res))
                res = e.getKey();
        }

        return (res == null) ? 0.0 : res;
    }
}
