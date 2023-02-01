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

	forward(x, xa, mask, kv_cache) {
		const q = this.query.apply(x);

		let k, v;
		if (!kv_cache || !xa || !(this.key in kv_cache)) {
			k = this.key.apply(!xa ? x : xa);
			v = this.value.apply(!xa ? x : xa);
		} else {
			k = kv_cache[this.key];
			v = kv_cache[this.value];
		}

		let wv, qk = this.qkv_attention(q, k, v, mask);

		return this.out.apply(wv), qk;
	}

	qkv_attention(q, k, v, mask) {
        let n_batch, n_ctx, n_state;
        n_batch = n_ctx = n_state = q.shape;

		let scale = Math.pow(Math.floor(n_state / this.n_head), -0.25);
		q = q.reshape(q.shape.slice(0, 2).concat([this.n_head], [-1])).transpose([0, 2, 1, 3]).mul(scale);
        k = k.reshape(k.shape.slice(0, 2).concat([this.n_head], [-1])).transpose([0, 2, 3, 1]).mul(scale);
        v = v.reshape(v.shape.slice(0, 2).concat([this.n_head], [-1])).transpose([0, 2, 1, 3]);

        let qk = q.matMul(k);
		if (mask) {
			qk = qk.add(mask.slice([0, 0], [n_ctx, n_ctx]));
		}
        qk = qk.cast('float32');

        w = tf.softmax(qk, dim=-1);

		res = w.matMul(v).transpose([0, 2, 1, 3]);
		new_shape = res.shape.slice(0, 2).concat([-1]);

		return res.reshape(new_shape), qk;
	}
}