# Pitch Detector

## Installation

*Requirements:*

* numpy
* python 2.6+

You can install dependencies with Pip:

    pip install -r requirements.txt

## Running

    python GenreDetector.py

## Results

TBA

## Development

1. Simple multi-layer network, learning on a whole wave data from files - RAM overwhelmed
2. Same network, working with running mean over wave data - RAM overwhelmed
3. Same network, ran on partial wave data - wrong results, huge RAM usage
4. Same network, learning from Fast Fourier Transform coefficients from wave data - RAM OK, wrong results (shorter datasets are filled with zeros)
5. Convolutional network, running on FFT coefficients from wave data - RAM OK, wrong results (`network03.v1.pickle`)
6. Same network, shuffled learning dataset (`network03.v2.pickle`)
7. Same network, shuffled learning dataset (`network03.v3.pickle`, longer datasets are shrinked down to the shortest length)
