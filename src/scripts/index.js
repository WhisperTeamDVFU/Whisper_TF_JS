import $ from "jquery";
import * as tf from "@tensorflow/tfjs";

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

function audio2tensor () {
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
            var tensorAudio = tf.tensor(audioArray);
            console.log('Audio file has transformed to tensor');
            return tensorAudio;
        });
    };
    reader.readAsArrayBuffer(file);
};

$('#read_audio').on('click', logMelSpectrogram(audio2tensor));


function logMelSpectrogram(audioTensor) {
    return audioTensor
};







