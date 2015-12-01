import javax.sound.sampled.*;

/**
 * Created by shybovycha on 01/12/15.
 */
public class Main {
    public static void main(String[] args) {
        TargetDataLine line = null;

        float sampleRate = 8000;
        int sampleSizeInBits = 8;
        int channels = 1;
        boolean signed = true;
        boolean bigEndian = true;

        AudioFormat format =  new AudioFormat(sampleRate, sampleSizeInBits, channels, signed, bigEndian);

        DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);

        if (!AudioSystem.isLineSupported(info)) {
            System.out.printf("Linear input is not supported");
            return;
        }

        try {
            line = (TargetDataLine) AudioSystem.getLine(info);
            line.open(format);
            line.start();

            int bufferSize = (int) format.getSampleRate() * format.getFrameSize();
            byte buffer[] = new byte[bufferSize];

            while (true) {
                int count = line.read(buffer, 0, buffer.length);

                if (count > 0) {
                    PitchDetector detector = new PitchDetector(buffer, count, format);

                    String pitch = detector.pitch();

                    System.out.printf("%s\n", pitch);
                }
            }
        } catch (LineUnavailableException e) {
            System.out.printf("Could not open in line");
            e.printStackTrace();
        } catch (Exception e) {
            if (line != null)
                line.close();

            e.printStackTrace();
        }
    }
}
