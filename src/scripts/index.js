import $ from "jquery";
import * as tf from "@tensorflow/tfjs";
import { Weights } from './Weights';
import mel_filters from './mel_filters.json'

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
            tensorAudio = tf.tensor(audioArray);
            console.log('Audio file has transformed to tensor');
            //console.log('tensorAudio', tensorAudio);
            return tensorAudio;
        });
    };
    reader.readAsArrayBuffer(file);
};
let logSpec;
 function logMelSpectrogram(audioTensor=tensorAudio) {

    console.log('tensorAudio', tensorAudio);
    let stft =  tf.signal.stft(tensorAudio, N_FFT, HOP_LENGTH, N_FFT, tf.signal.hannWindow);
    let magnitudes = tf.abs(stft).pow(2).transpose();
    let melSpec = tf.matMul(mel_filters, magnitudes);
    logSpec = tf.minimum(  tf.maximum(melSpec, tf.tensor(1e-10) ), melSpec.max().arraySync() );
    logSpec = log10(logSpec);
    logSpec = tf.maximum(logSpec, tf.sub(logSpec.max(), 8.0).arraySync());
    logSpec = tf.div(tf.add(logSpec, 4.0), 4.0);
    console.log('LMS', logSpec );
    return logSpec;
};

let IsTensorReady = false;
$('#read_audio').on('click', audio2tensor);
$('#LMS_test').on('click', logMelSpectrogram);



$(function() {
    $("h1").on("click", function() {
        alert("jQuery is working!");
    });

    $('#model_size_select').on("change", function(){
        alert($(this).text());
    });

    $('#input_file_weights').on('change', async function() {
        let file = $(this)[0].files[0];
        console.log(file);
        let data = await file.arrayBuffer();
        console.log(data);

        let weights = new Weights(file.name);

        await weights.init(data);

        console.log(weights);

        weights.get('decoder.positional_embedding').print();
    });



    $('#input_file_config').on('change', async function() {
        let file = $(this)[0].files[0];
        console.log(file);
        var fileread = new FileReader();
        fileread.onload = function(e) {
        var content = e.target.result;
        var intern = JSON.parse(content);
        console.log(intern);
        console.log(intern.n_mels);
        };

        fileread.readAsText(file);
    });



});

