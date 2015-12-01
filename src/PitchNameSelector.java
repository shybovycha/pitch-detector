import java.util.HashMap;
import java.util.Map;

/**
 * Created by shybovycha on 01/12/15.
 */
public class PitchNameSelector {
    private static Map<Double, String> pitches;

    static {
        pitches = new HashMap<>();

        pitches.put(261.63, "C");
        pitches.put(523.25, "C");
        pitches.put(1046.5, "C");
        pitches.put(2093.0, "C");
        pitches.put(4186.0, "C");
        pitches.put(8372.0, "C");
        pitches.put(16744.0, "C");

        pitches.put(293.66, "D");
        pitches.put(587.33, "D");
        pitches.put(1174.7, "D");
        pitches.put(2349.3, "D");
        pitches.put(4698.6, "D");
        pitches.put(9397.3, "D");
        pitches.put(18794.5, "D");

        pitches.put(329.63, "E");
        pitches.put(659.26, "E");
        pitches.put(1318.5, "E");
        pitches.put(2637.0, "E");
        pitches.put(5274.0, "E");
        pitches.put(10548.0, "E");
        pitches.put(21096.2, "E");

        pitches.put(349.23, "F");
        pitches.put(698.46, "F");
        pitches.put(1396.9, "F");
        pitches.put(2793.8, "F");
        pitches.put(5587.7, "F");
        pitches.put(11175.0, "F");
        pitches.put(22350.6, "F");

        pitches.put(392.00, "G");
        pitches.put(783.99, "G");
        pitches.put(1568.0, "G");
        pitches.put(3136.0, "G");
        pitches.put(6271.9, "G");
        pitches.put(12544.0, "G");
        pitches.put(25087.7, "G");

        pitches.put(440.00, "A");
        pitches.put(880.00, "A");
        pitches.put(1760.0, "A");
        pitches.put(3520.0, "A");
        pitches.put(7040.0, "A");
        pitches.put(14080.0, "A");
        pitches.put(28160.0, "A");

        pitches.put(493.88, "H (B)");
        pitches.put(987.77, "H (B)");
        pitches.put(1975.5, "H (B)");
        pitches.put(3951.1, "H (B)");
        pitches.put(7902.1, "H (B)");
        pitches.put(15804.0, "H (B)");
        pitches.put(31608.5, "H (B)");
    }

    public static String findClosest(double frequency) {
        String res = "unknown";
        Double closest = 261.63;

        for (Map.Entry<Double, String> e : pitches.entrySet()) {
            if (Math.abs(e.getKey() - frequency) < Math.abs(frequency - closest)) {
                closest = e.getKey();
                res = e.getValue();
            }
        }

//        System.out.printf("Closest: %s (%.3f <-> %.3f)\n", res, frequency, closest);

        return res;
    }
}
