import $ from "jquery";
import { Weights } from './Weights';
import { Whisper } from './Whisper-test';
import tiny_url from 'url:/tiny.hdf5';
import base_url from 'url:/base.hdf5';
import CONFIGS from './Config';

const MODELS_URL = {
    "tiny": tiny_url,
    "base": base_url,
    "small": tiny_url,
    "medium": tiny_url,
    "large": tiny_url
}

let CurrentWeights;
let CurrentCofig;
let CurrentSizeModel = "";

import { audio2tensor, logMelSpectrogram } from './LMS';

let logSpec;
$('#read_audio').on('click', audio2tensor);
$('#LMS_test').on('click', () => logSpec = logMelSpectrogram());

$(function() {

    $('#model_size_select').on("change", async function(){
        CurrentSizeModel = this.value;
        document.getElementById("CurrentSizeModel").innerHTML = CurrentSizeModel;

        let file_weights = await fetch(MODELS_URL[CurrentSizeModel]);
        let data = await file_weights.arrayBuffer();
        let weights = new Weights(CurrentSizeModel);
        await weights.init(data);
        CurrentWeights = weights;
        document.getElementById("CurrentWeights").innerHTML = CurrentWeights.get('decoder.positional_embedding');

        weights.get('decoder.positional_embedding').print();

        CurrentCofig = CONFIGS[CurrentSizeModel];
        document.getElementById("CurrentCofig").innerHTML = [
            CurrentCofig.n_mels,
            CurrentCofig.n_vocab, 
            CurrentCofig.n_audio_ctx,
            CurrentCofig.n_audio_state,
            CurrentCofig.n_audio_head,
            CurrentCofig.n_audio_layer,
            CurrentCofig.n_text_ctx,
            CurrentCofig.n_text_state,
            CurrentCofig.n_text_head,
            CurrentCofig.n_text_layer,
        ];
        console.log(CurrentCofig);
    });

    $('#create_model').on('click', async function() {
        let whisper = new Whisper(CurrentSizeModel);

        await whisper.init(CurrentCofig, CurrentWeights);
    });
});

