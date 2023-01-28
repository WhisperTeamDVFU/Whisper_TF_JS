import h5wasm from "h5wasm";
import * as tf from '@tensorflow/tfjs';

export class Weights {
    constructor(path) {
        this.file = new h5wasm.File(path, "r");
    }

    get(key) {
        let data = this.file.get(key);
        return tf.tensor(data.value, data.shape);
    }
}