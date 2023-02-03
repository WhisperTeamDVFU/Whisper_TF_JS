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
//console.log('N_SAMPLES', N_SAMPLES, 'N_FRAMES', N_FRAMES);
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
    console.log('tensorAudio');
    let stft = tf.signal.stft(tensorAudio, N_FFT, HOP_LENGTH, N_FFT, tf.signal.hannWindow);
    let magnitudes = tf.abs(stft).pow(2).transpose();
    let melSpec = tf.matMul(mel_filters, magnitudes);
    logSpec = tf.minimum(  tf.maximum(melSpec, tf.tensor(1e-10) ), melSpec.max().arraySync() );
    logSpec = log10(logSpec);
    logSpec = tf.maximum(logSpec, tf.sub(logSpec.max(), 8.0).arraySync());
    logSpec = tf.div(tf.add(logSpec, 4.0), 4.0);
    logSpec = padOrTrim(logSpec);
    console.log('Log mel spectrogram is calculated');
    console.log(logSpec);
    return logSpec

};

function padOrTrim(array=logSpec, length=N_SAMPLES) {
    //paddings: It is an array of length R, the rank of the given tensor,
    // where each element is of length 2 of ints ([pad_Before, pad_After]),
    // specifies how much padding should be given along each dimension of the tensor.
    console.log( N_FRAMES, logSpec.shape[1]);
    if (logSpec.shape[1] < N_FRAMES) {
        logSpec =  tf.pad(logSpec, [[0,0],[0,N_FRAMES-logSpec.shape[1]]])
    }
    if (logSpec.shape[1] > N_FRAMES) {

    };
    console.log( N_FRAMES, logSpec.shape[1]);
    console.log('padded');
    return logSpec
};

