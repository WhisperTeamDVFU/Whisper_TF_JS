import h5wasm from "h5wasm";
import * as tf from '@tensorflow/tfjs';

export class Weights {
    constructor(name) {
        this.name = name;        
    }

    async init(buffer) {
        const { FS } = await h5wasm.ready;

        FS.writeFile(this.name + ".h5", new Uint8Array(buffer));

        this.file = new h5wasm.File(this.name + ".h5", "r");
        this.keys = this.file.keys()
    }

    get(key) {
        let data = this.file.get(key);
        return tf.tensor(data.value, data.shape, 'float32');
    }
}