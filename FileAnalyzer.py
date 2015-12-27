import wave
import sys
import numpy as np
import math


class FileAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.notes = {u'C<sub>6</sub>': 1046.5, u'A<sub>2</sub>': 110, u'G<sub>4</sub>': 392,
                      u'&nbsp;G<sup>#</sup><sub>7</sub>/A<sup>b</sup><sub>7</sub>&nbsp;': 3322.4400000000001,
                      u'E<sub>1</sub>': 41.200000000000003,
                      u'&nbsp;A<sup>#</sup><sub>2</sub>/B<sup>b</sup><sub>2</sub>&nbsp;': 116.54000000000001,
                      u'E<sub>4</sub>': 329.63,
                      u'&nbsp;A<sup>#</sup><sub>7</sub>/B<sup>b</sup><sub>7</sub>&nbsp;': 3729.3099999999999,
                      u'&nbsp;C<sup>#</sup><sub>2</sub>/D<sup>b</sup><sub>2</sub>&nbsp;': 69.299999999999997,
                      u'B<sub>2</sub>': 123.47, u'G<sub>5</sub>': 783.99000000000001, u'B<sub>4</sub>': 493.88,
                      u'D<sub>1</sub>': 36.710000000000001,
                      u'&nbsp;D<sup>#</sup><sub>3</sub>/E<sup>b</sup><sub>3</sub>&nbsp;': 155.56,
                      u'F<sub>7</sub>': 2793.8299999999999,
                      u'&nbsp;G<sup>#</sup><sub>8</sub>/A<sup>b</sup><sub>8</sub>&nbsp;': 6644.8800000000001,
                      u'E<sub>6</sub>': 1318.51, u'G<sub>6</sub>': 1567.98,
                      u'&nbsp;G<sup>#</sup><sub>3</sub>/A<sup>b</sup><sub>3</sub>&nbsp;': 207.65000000000001,
                      u'&nbsp;D<sup>#</sup><sub>1</sub>/E<sup>b</sup><sub>1</sub>&nbsp;': 38.890000000000001,
                      u'F<sub>8</sub>': 5587.6499999999996,
                      u'&nbsp;C<sup>#</sup><sub>6</sub>/D<sup>b</sup><sub>6</sub>&nbsp;': 1108.73,
                      u'&nbsp;A<sup>#</sup><sub>0</sub>/B<sup>b</sup><sub>0</sub>&nbsp;': 29.140000000000001,
                      u'B<sub>3</sub>': 246.94, u'C<sub>5</sub>': 523.25, u'F<sub>2</sub>': 87.310000000000002,
                      u'C<sub>1</sub>': 32.700000000000003,
                      u'&nbsp;F<sup>#</sup><sub>5</sub>/G<sup>b</sup><sub>5</sub>&nbsp;': 739.99000000000001,
                      u'E<sub>5</sub>': 659.25,
                      u'&nbsp;C<sup>#</sup><sub>3</sub>/D<sup>b</sup><sub>3</sub>&nbsp;': 138.59,
                      u'F<sub>5</sub>': 698.46000000000004, u'F<sub>1</sub>': 43.649999999999999,
                      u'B<sub>6</sub>': 1975.53, u'C<sub>8</sub>': 4186.0100000000002,
                      u'&nbsp;D<sup>#</sup><sub>0</sub>/E<sup>b</sup><sub>0</sub>&nbsp;': 19.449999999999999,
                      u'&nbsp;C<sup>#</sup><sub>4</sub>/D<sup>b</sup><sub>4</sub>&nbsp;': 277.18000000000001,
                      u'G<sub>0</sub>': 24.5, u'E<sub>7</sub>': 2637.02, u'A<sub>5</sub>': 880,
                      u'&nbsp;G<sup>#</sup><sub>5</sub>/A<sup>b</sup><sub>5</sub>&nbsp;': 830.61000000000001,
                      u'D<sub>3</sub>': 146.83000000000001, u'D<sub>0</sub>': 18.350000000000001,
                      u'E<sub>0</sub>': 20.600000000000001,
                      u'&nbsp;G<sup>#</sup><sub>0</sub>/A<sup>b</sup><sub>0</sub>&nbsp;': 25.960000000000001,
                      u'F<sub>6</sub>': 1396.9100000000001,
                      u'&nbsp;A<sup>#</sup><sub>3</sub>/B<sup>b</sup><sub>3</sub>&nbsp;': 233.08000000000001,
                      u'&nbsp;F<sup>#</sup><sub>8</sub>/G<sup>b</sup><sub>8</sub>&nbsp;': 5919.9099999999999,
                      u'F<sub>3</sub>': 174.61000000000001, u'A<sub>3</sub>': 220,
                      u'&nbsp;C<sup>#</sup><sub>5</sub>/D<sup>b</sup><sub>5</sub>&nbsp;': 554.37,
                      u'D<sub>2</sub>': 73.420000000000002, u'G<sub>1</sub>': 49,
                      u'&nbsp;F<sup>#</sup><sub>1</sub>/G<sup>b</sup><sub>1</sub>&nbsp;': 46.25,
                      u'&nbsp;D<sup>#</sup><sub>8</sub>/E<sup>b</sup><sub>8</sub>&nbsp;': 4978.0299999999997,
                      u'&nbsp;G<sup>#</sup><sub>1</sub>/A<sup>b</sup><sub>1</sub>&nbsp;': 51.909999999999997,
                      u'G<sub>3</sub>': 196,
                      u'&nbsp;A<sup>#</sup><sub>5</sub>/B<sup>b</sup><sub>5</sub>&nbsp;': 932.33000000000004,
                      u'&nbsp;F<sup>#</sup><sub>6</sub>/G<sup>b</sup><sub>6</sub>&nbsp;': 1479.98,
                      u'B<sub>1</sub>': 61.740000000000002, u'D<sub>7</sub>': 2349.3200000000002,
                      u'&nbsp;F<sup>#</sup><sub>7</sub>/G<sup>b</sup><sub>7</sub>&nbsp;': 2959.96,
                      u'&nbsp;D<sup>#</sup><sub>7</sub>/E<sup>b</sup><sub>7</sub>&nbsp;': 2489.02,
                      u'&nbsp;D<sup>#</sup><sub>6</sub>/E<sup>b</sup><sub>6</sub>&nbsp;': 1244.51,
                      u'B<sub>0</sub>': 30.870000000000001, u'C<sub>2</sub>': 65.409999999999997, u'A<sub>1</sub>': 55,
                      u'&nbsp;A<sup>#</sup><sub>8</sub>/B<sup>b</sup><sub>8</sub>&nbsp;': 7458.6199999999999,
                      u'B<sub>5</sub>': 987.76999999999998,
                      u'&nbsp;D<sup>#</sup><sub>5</sub>/E<sup>b</sup><sub>5</sub>&nbsp;': 622.25,
                      u'&nbsp;G<sup>#</sup><sub>2</sub>/A<sup>b</sup><sub>2</sub>&nbsp;': 103.83,
                      u'D<sub>4</sub>': 293.66000000000003, u'F<sub>0</sub>': 21.829999999999998,
                      u'A<sub>6</sub>': 1760, u'G<sub>7</sub>': 3135.96, u'A<sub>0</sub>': 27.5, u'C<sub>7</sub>': 2093,
                      u'G<sub>8</sub>': 6271.9300000000003, u'E<sub>8</sub>': 5274.04,
                      u'F<sub>4</sub>': 349.23000000000002,
                      u'&nbsp;G<sup>#</sup><sub>6</sub>/A<sup>b</sup><sub>6</sub>&nbsp;': 1661.22,
                      u'&nbsp;G<sup>#</sup><sub>4</sub>/A<sup>b</sup><sub>4</sub>&nbsp;': 415.30000000000001,
                      u'A<sub>7</sub>': 3520,
                      u'&nbsp;C<sup>#</sup><sub>7</sub>/D<sup>b</sup><sub>7</sub>&nbsp;': 2217.46,
                      u'&nbsp;A<sup>#</sup><sub>6</sub>/B<sup>b</sup><sub>6</sub>&nbsp;': 1864.6600000000001,
                      u'D<sub>5</sub>': 587.33000000000004,
                      u'&nbsp;F<sup>#</sup><sub>3</sub>/G<sup>b</sup><sub>3</sub>&nbsp;': 185, u'A<sub>4</sub>': 440,
                      u'C<sub>0</sub>': 16.350000000000001, u'E<sub>3</sub>': 164.81, u'C<sub>4</sub>': 261.63,
                      u'&nbsp;A<sup>#</sup><sub>1</sub>/B<sup>b</sup><sub>1</sub>&nbsp;': 58.270000000000003,
                      u'&nbsp;F<sup>#</sup><sub>2</sub>/G<sup>b</sup><sub>2</sub>&nbsp;': 92.5, u'A<sub>8</sub>': 7040,
                      u'&nbsp;C<sup>#</sup><sub>8</sub>/D<sup>b</sup><sub>8</sub>&nbsp;': 4434.9200000000001,
                      u'&nbsp;F<sup>#</sup><sub>4</sub>/G<sup>b</sup><sub>4</sub>&nbsp;': 369.99000000000001,
                      u'&nbsp;C<sup>#</sup><sub>0</sub>/D<sup>b</sup><sub>0</sub>&nbsp;': 17.32,
                      u'&nbsp;F<sup>#</sup><sub>0</sub>/G<sup>b</sup><sub>0</sub>&nbsp;': 23.120000000000001,
                      u'D<sub>8</sub>': 4698.6300000000001, u'G<sub>2</sub>': 98, u'B<sub>8</sub>': 7902.1300000000001,
                      u'&nbsp;D<sup>#</sup><sub>2</sub>/E<sup>b</sup><sub>2</sub>&nbsp;': 77.780000000000001,
                      u'E<sub>2</sub>': 82.409999999999997,
                      u'&nbsp;D<sup>#</sup><sub>4</sub>/E<sup>b</sup><sub>4</sub>&nbsp;': 311.13,
                      u'D<sub>6</sub>': 1174.6600000000001, u'B<sub>7</sub>': 3951.0700000000002,
                      u'&nbsp;A<sup>#</sup><sub>4</sub>/B<sup>b</sup><sub>4</sub>&nbsp;': 466.16000000000003,
                      u'&nbsp;C<sup>#</sup><sub>1</sub>/D<sup>b</sup><sub>1</sub>&nbsp;': 34.649999999999999,
                      u'C<sub>3</sub>': 130.81}

    def analyze(self):
        wf = wave.open(self.filename, 'rb')

        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        num_frames = wf.getnframes()

        length_in_seconds = num_frames / (frame_rate * 1.0)

        num_iterations = int(length_in_seconds * 2)
        iteration_size = int(num_frames / num_iterations)

        leading_notes = []

        for fragment_num in range(0, num_iterations):
            chunk = iteration_size

            data = wf.readframes(chunk)
            window = np.blackman(len(data) / sample_width)
            in_data = np.array(wave.struct.unpack("%dh" % (len(data) / sample_width), data)) * window
            fft_data = abs(np.fft.rfft(in_data)) ** 2

            majors_indices = fft_data.argsort()[::-1][0:10]
            major_notes = []

            # looking for 10 major notes
            for which in majors_indices:
                fragment = fft_data[which - 1:which + 2:]

                if len(fragment) < 3:
                    heap_freq = which * frame_rate / chunk
                else:
                    y0, y1, y2 = np.log(fragment)
                    x1 = (y2 - y0) * 0.5 / (2 * y1 - y2 - y0)
                    heap_freq = (which + x1) * frame_rate / chunk

                note = self.find_note(heap_freq)
                major_notes.append(note)

            # print "Leading notes: %s\n" % (','.join(set(major_notes)))
            leading_notes.append(set(major_notes))

        return leading_notes

    def find_note(self, freq):
        def reducer(acc, pair):
            if math.sqrt((pair[1] - freq) ** 2) < math.sqrt((acc[1] - freq) ** 2):
                return pair
            else:
                return acc

        candidates = reduce(reducer, self.notes.iteritems())

        return candidates[0]
