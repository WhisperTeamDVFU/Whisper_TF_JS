export class Whisper {
    constructor(name) {
        this.name = name;
    }

    async init(currentConfig, currentWeights) {
        this.dims = currentConfig;   
        this.wheights = currentWeights; 

        let n_mels = this.dims.n_mels;
        let n_audio_ctx = this.dims.n_audio_ctx;
        let n_audio_state = this.dims.n_audio_state;
        let n_audio_head = this.dims.n_audio_head;
        let n_audio_layer = this.dims.n_audio_layer;

        let n_vocab = this.dims.n_vocab;
        let n_text_ctx = this.dims.n_text_ctx;
        let n_text_state = this.dims.n_text_state;
        let n_text_head = this.dims.n_text_head;
        let n_text_layer = this.dims.n_text_layer;

        console.log("Okey")

    }

}