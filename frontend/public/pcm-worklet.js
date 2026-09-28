class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._nativeChunkSize = Math.round(0.5 * sampleRate);
    this._carry = [];
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) {
      for (let i = 0; i < channel.length; i++) {
        this._carry.push(channel[i]);
      }
      while (this._carry.length >= this._nativeChunkSize) {
        const slice = this._carry.slice(0, this._nativeChunkSize);
        this._carry = this._carry.slice(this._nativeChunkSize);
        const out = new Int16Array(8000);
        for (let j = 0; j < 8000; j++) {
          const idx = Math.floor((j * this._nativeChunkSize) / 8000);
          const s = Math.max(-1, Math.min(1, slice[idx]));
          out[j] = s < 0 ? s * 32768 : s * 32767;
        }
        this.port.postMessage(out.buffer, [out.buffer]);
      }
    }
    return true;
  }
}

registerProcessor("pcm-worklet-processor", PCMWorkletProcessor);
