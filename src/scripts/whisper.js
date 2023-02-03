import * as tf from '@tensorflow/tfjs';


function sinusoids(length, channels, max_timescale=10000) 
{
	tf.util.assert(channels % 2 == 0);
	const log_timescale_increment = Math.log(max_timescale) / (Math.floor(channels / 2) - 1);
	const inv_timescales = tf.exp(tf.range(0, Math.floor(channels / 2)).mul(-log_timescale_increment));
    const scaled_time = tf.expandDims(tf.range(0, length), axis=-1).mul(tf.expandDims(inv_timescales, axis=0));
	return tf.concat([scaled_time.sin(), scaled_time.cos()], 1);
}

class MultiHeadAttention extends tf.layers.Layer {
	constructor(n_state, n_head) {
		super();

		this.n_head = n_head;
		this.query = tf.layers.dense({units: n_state, inputShape: n_state});
		this.key = tf.layers.dense({units: n_state, inputShape: n_state, useBias: false});
		this.value = tf.layers.dense({units: n_state, inputShape: n_state});
		this.out = tf.layers.dense({units: n_state, inputShape: n_state});
	}

	call(x, xa = null, mask = null, kv_cache = null) {
		const q = this.query.apply(x);

		let k, v;
		if (!kv_cache || !xa || !(this.key in kv_cache)) {
			k = this.key.apply(!xa ? x : xa);
			v = this.value.apply(!xa ? x : xa);
		} else {
			k = kv_cache[this.key];
			v = kv_cache[this.value];
		}

		let wv = this.qkv_attention(q, k, v, mask);

		return this.out.apply(wv);
	}

	qkv_attention(q, k, v, mask = null) {
        let [n_batch, n_ctx, n_state] = q.shape;

		let scale = Math.pow(Math.floor(n_state / this.n_head), -0.25);
		q = q.reshape(q.shape.slice(0, 2).concat([this.n_head], [-1])).transpose([0, 2, 1, 3]).mul(scale);
        k = k.reshape(k.shape.slice(0, 2).concat([this.n_head], [-1])).transpose([0, 2, 3, 1]).mul(scale);
        v = v.reshape(v.shape.slice(0, 2).concat([this.n_head], [-1])).transpose([0, 2, 1, 3]);

        let qk = q.matMul(k);
		if (mask) {
			qk = qk.add(mask.slice([0, 0], [n_ctx, n_ctx]));
		}
        qk = qk.cast('float32');

        // let w = tf.softmax(qk, dim=-1);
		let w = tf.layers.softmax({axis: -1}).apply(qk).cast(q.dtype);

		let res = w.matMul(v).transpose([0, 2, 1, 3]);
		let new_shape = res.shape.slice(0, 2).concat([-1]);

		return res.reshape(new_shape);
	}
}

class GELU extends tf.layers.layer {
	call(x) {
		return x.mul(tf.erf(tf.div(x, Math.sqrt(2))).add(tf.scalar(1)).mul(0.5));
	}
}

class ResidualAttentionBlock extends tf.layers.Layer {
	constructor(n_state, n_head, cross_attention=false) {
		super();

		this.attn = new MultiHeadAttention({n_state: n_state, n_head: n_head});
		this.attn_ln = tf.layers.layerNormalization({input_shape: n_state});

		this.cross_attn = cross_attention ? new MultiHeadAttention({n_state: n_state, n_head: n_head}) : null;
		this.cross_attn_ln = cross_attention ? tf.layers.layerNormalization({input_shape: n_state}) : null;

		let n_mlp = n_state * 4;
		this.mlp = tf.sequential({
			layers: [
				tf.layers.dense({units: n_mlp, inputShape: n_state}),
				new GELU(),
				tf.layers.dense({units: n_state, inputShape: n_mlp}),				
			]
		})
		this.mlp_ln = tf.layers.layerNormalization({input_shape: n_state});
	}

	call(x, xa = null, mask = null, kv_cache = null) {
		x = x.add(this.attn.apply({x: self.attn_ln.apply(x), mask: mask, kv_cache: kv_cache}));
		if (this.cross_attn) {
			x = x.add(this.cross_attn.apply({x: this.cross_attn_ln.apply(x), xa: xa, kv_cache: kv_cache}));
		}
		x = x.add(this.mlp.apply(this.mlp_ln.apply(x)));

		return x;
	}
}

class AudioEncoder extends tf.layers.Layer {
	constructor(n_mels, n_ctx, n_state, n_head, n_layer, weights_name, weights_gen) {
		super();

		this.conv1 = tf.layers.conv1d({filters: n_state, kernelSize: 3, padding: 'same'});
		this.conv2 = tf.layers.conv1d({filters: n_state, kernelSize: 3, strides: 2, padding: 'same'});
		this.positional_embedding = sinusoids(n_ctx, n_state);

		this.blocks = [];
		for (let i = 0; i < n_layer; i++) {
			this.blocks.push(new ResidualAttentionBlock({n_state: n_state, n_head: n_head}));
		}
		this.ln_post = tf.layers.layerNormalization({inputShape: n_state});

		this.gelu = new GELU();
	}

	call(x) {
		x = this.gelu.apply(this.conv1.apply(x));
		x = this.gelu.apply(this.conv2.apply(x));
		x = x.transpose([0, 2, 1]);

		tf.util.assert(tf.equal(tf.tensor(x.shape.slice(1)), tf.tensor(this.positional_embedding.shape)), 'incorrect audio shape');
		x = x.add(this.positional_embedding);
		
		for (let block of this.blocks) {
			x = block.apply(x);
		}

		x = this.ln_post.apply(x);
		return x;
	}
}

function Triu_1(arr, n_rows, n_cols) {
	for (let i = 0; i < n_rows; i++) {
		for (let j = 0; j < n_cols; j++) {
			if (i >= j) {
				arr[i][j] = 0;
			}
		}
	}

	return tf.Tensor(arr);
}

class TextDecoder extends tf.layers.Layer {
	constructor(n_vocab, n_ctx, n_state, n_head, n_layer, weights_names, weights_gen) {
		super();

		this.token_embedding = tf.layers.embedding(n_vocab, n_state);
		this.positional_embedding = tf.fill([n_ctx, n_state], null);

		this.blocks = [];
		for (let i = 0; i < n_layer; i++) {
			this.blocks.push(new ResidualAttentionBlock({n_state: n_state, n_head: n_head, cross_attention: true}));
		}
		this.ln = tf.layers.layerNormalization({inputShape: n_state});

		this.mask = Triu_1(tf.fill([n_ctx, n_ctx], -Infinity).arraySync(), n_ctx, n_ctx);
	}

	call(x, xa, kv_cache = null) {
		offset = kv_cache ? Object.values(kv_cache)[0].shape[1] : 0;
		let x = this.token_embedding.apply(x).add(this.positional_embedding.slice([offset], [offset + x.shape[-1]]));
		x = x.cast(xa.dtype);

		for (let block of this.blocks) {
			x = block.apply({x: x, xa: xa, mask: this.mask, kv_cache: kv_cache});
		}

		x = this.ln.apply(x);
		logits = x.matMul(this.token_embedding.getWeights.cast(x.dtype).transpose([0, 1]));

		return logits;
	}
}

class Whisper extends tf.layers.Layer {
	constructor(current_config, current_weights) {
		this.dims = current_config;
		this.weights_gen = current_weights;

		weights_keys = this.weights_gen.key;

		this.weights_encoder = [];
		this.weights_decoder = [];

		for (let key of weights_keys) {
			if(key.includes('encoder')) {
				this.weights_encoder.push(key);
			}

			if(key.includes('decoder')) {
				this.weights_decoder.push(key);
			}
		}

		this.encoder = new AudioEncoder(
			this.dims.n_mels,
			this.dims.n_audio_ctx,
			this.dims.n_audio_state,
			this.dims.n_audio_head,
			this.dims.n_audio_layer,
			this.weights_encoder,
			this.weights_gen
		);

		this.decoder = new TextDecoder(
			this.dims.n_vocab,
			this.dims.n_text_ctx,
			this.dims.n_text_state,
			this.dims.n_text_head,
			this.dims.n_text_layer,
			this.weights_decoder,
			this.weights_gen
		);
	}

	embed_audio(mel) {
		return this.encoder.apply(mel);
	}

	logits(tokens, audio_features) {
		return this.decoder(tokens, audio_features);
	}

	call(mel, tokens) {
		return this.decoder(tokens, this.embed_audio(mel));
	}
}