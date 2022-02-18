# CHORD TRACKER - SASP PROJECT 2021-2022
## TASK
The task is to implement a chord tracker based on Chromagram time-frequency analysis together with a Hidden Markov Model statistical analysis. 
## DATASET
We decided to use a subset of the famous dataset "The Beatles” developed by Chris Harte (http://isophonics.net/content/reference-annotations-beatles). The subset consists of 42 Beatles tracks for which we have both the audio files (.wav) and the annotation files (.lab). The .lab files essentially are composed by 3 columns: "start", "end" and "chord" where we have the starting and ending point in time of all chords along each song, for example:

A_Day_In_The_Life.lab:

0.000000 1.914761 G
1.914761 3.369980 B:min
3.369980 6.519569 E:min7
6.519569 12.848047 C
12.848047 14.425963 G
....

Each file .lab and .wav has been reviewed by us in order to be sure that audio and annotations are synchronized, since there are many different version of the same song in The Beatles discography.

## PREPROCESSING
We used the Harmonic Percussive Sound Separation (HPSS: https://librosa.org/doc/main/generated/librosa.decompose.hpss.html) technique to split up the harmonic component of the audio file from the percussive one.

<img width="920" alt="HPSS" src="https://user-images.githubusercontent.com/57909529/153031738-65469d96-806a-4093-922d-b496da85c83a.png">

After that, we're ready to perform the Chromagram extraction of our files.

## CHROMAGRAM
Chromagrams are a convenient representation of sound that uses time-frequency analysis of audio files, reworked in order to highlight the notes played (the ones with the higher power spectral density) along time. 

![ChromaFeature](https://user-images.githubusercontent.com/57909529/153032657-83fde971-fc52-4154-b295-b558429e8307.png)

## HIDDEN MARKOV MODEL
Hidden Markov Models are Markov Models where the states cannot be seen directly but only through a representation of them (observable events). In our case, the chromagrams are the observable events that represent the chords (states) that we want to model. 

![HMM](https://user-images.githubusercontent.com/57909529/153033745-95c3ecb4-5812-43c9-b81f-b2da0a17f929.png)

## PIPELINE
The training pipeline has the following structure:

- convert all files in .wav format resampling them at 44.1kHz;
- perform HPSS for each audio;
- extract Chromagram representations and store them;
- associate chromagram to .lab annotations;
- compute the probability state matrix and the initial state matrix from .lab files;
- compute the emission probabilities;
- train hmm model;
- make predictions.

## USED LIBRARIES

- librosa
- scipy
- hmmlearn
- numpy/pandas
- pydub
- sklearn


## FUTURE DEVELOPMENT
We've made a comparison with chromagrams extracted from Librosa and the performances are similar. The accuracy in prediction is not the highest but for multiple causes:

- the dataset is composed by too few songs (42) and all from the Beatles discography so we have no generalization;
- HPSS split harmonic and percussive components: this surely helps but the harmonic remaining part is still a mixture of guitar, bass, voice, organ and other noisy components. With a different separation algorithm such as Spleeter https://github.com/deezer/spleeter) we think that performance could improve;
- machine learning/ deep learning technique could be used, having computational power and a bigger (and well annotated!) dataset;
- the conditioning could be done exploiting music theory of harmony.
