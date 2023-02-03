import * as tf from "@tensorflow/tfjs";
import mel_filters from "./mel_filters.json";
import $ from "jquery";

function exactDiv(x, y) {
    return Math.floor(x / y);
};
function log10(x) {
    let x1 = tf.log(x);
    let x2 = tf.log(10.0);
    return tf.div(x1, x2);
};

const SAMPLE_RATE = 16000
const N_FFT = 400
const N_MELS = 80
const HOP_LENGTH = 160
const CHUNK_LENGTH = 30
const N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  // 480000: number of samples in a chunk
const N_FRAMES = exactDiv(N_SAMPLES, HOP_LENGTH)  // 3000: number of frames in a mel spectrogram input

var audioBuffer;
let tensorAudio;
let logSpec;

export function audio2tensor () {
    let file = $('#audiofile')[0].files[0];
    let reader = new FileReader();

    reader.onload = async function () {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext )();
        const source = audioCtx.createBufferSource();
        let audioArrBuffer = reader.result;
        audioCtx.sampleRate = SAMPLE_RATE;
        audioCtx.decodeAudioData(audioArrBuffer).then(async function (audioBuffer) {
            let offlineCtx = new OfflineAudioContext(audioBuffer.numberOfChannels,
                audioBuffer.duration * SAMPLE_RATE,
                SAMPLE_RATE);
            let offlineSrc = offlineCtx.createBufferSource();
            offlineSrc.buffer = audioBuffer;
            offlineSrc.connect(offlineCtx.destination);
            offlineSrc.start();
            let resampled = await offlineCtx.startRendering();
            let audioArray = new Float32Array(resampled.length);
            resampled.copyFromChannel(audioArray, 0, 0);
            tensorAudio = tf.tensor(audioArray);
            console.log('Audio file has transformed to tensor');
            //console.log('tensorAudio', tensorAudio);
            return tensorAudio;
        });
    };
    reader.readAsArrayBuffer(file);
};

export function logMelSpectrogram(audioTensor=audio2tensor) {
    console.log('LMS is started');
    let stft = tf.signal.stft(tensorAudio, N_FFT, HOP_LENGTH, N_FFT, tf.signal.hannWindow);
    let magnitudes = tf.abs(stft).pow(2).transpose();
    let melSpec = tf.matMul(mel_filters, magnitudes);
    logSpec = tf.minimum(  tf.maximum(melSpec, tf.tensor(1e-10) ), melSpec.max().arraySync() );
    logSpec = log10(logSpec);
    logSpec = tf.maximum(logSpec, tf.sub(logSpec.max(), 8.0).arraySync());
    logSpec = tf.div(tf.add(logSpec, 4.0), 4.0);
    console.log('Log mel spectrogram is calculated');
    logSpec = cutTo30Sec(logSpec);
    console.log(logSpec);
    return logSpec

};

function cutTo30Sec(array=logSpec, length=N_FRAMES){
    let cutSpec;
    let specShape = logSpec.shape[1];
    if (logSpec.shape[1] > N_FRAMES) {
        cutSpec =  logSpec.slice([0,0], [80,N_FRAMES]);
        console.log('First 30 sec of LMS');
        return cutSpec
    } else {
        return logSpec
    };

};

function padOrTrim(array=logSpec, length=N_FRAMES) {
     if (logSpec.shape[1] < N_FRAMES) {
        logSpec =  tf.pad(logSpec, [[0,0],[0,N_FRAMES-logSpec.shape[1]]]);
    };
    if (logSpec.shape[1] > N_FRAMES) {
             logSpec = tf.gather(logSpec, tf.range(0, length, 1, 'int32'), -1);
         };
    return logSpec
};


