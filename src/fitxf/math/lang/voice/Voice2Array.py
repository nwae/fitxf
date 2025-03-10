import logging
import wave
import re
import os
import io
import math
import struct
import numpy as np
import scipy.io.wavfile as wav
from fitxf.math.algo.encoding.Base64 import Base64
from fitxf.math.utils.Logging import Logging


"""
Digital Voice Data Plumbing

Record voice, convert ogg, ogg base 64 bytes to standard PCM (Pulse Code Modulation).
Each sample in PCM either int16 or int32, signed.
"""


class Voice2Array:

    @staticmethod
    def get_pyaudio_type(sample_width):
        import pyaudio
        if sample_width == 2:
            return pyaudio.paInt16
        elif sample_width == 4:
            return pyaudio.paInt32
        else:
            raise Exception('Sample width ' + str(sample_width))

    @staticmethod
    def get_numpy_type(sample_width):
        if sample_width == 2:
            return np.int16
        elif sample_width == 4:
            return np.int32
        else:
            raise Exception('Sample width ' + str(sample_width))

    @staticmethod
    def convert_pcm_float_to_int(x: np.ndarray, to_dtype):
        assert to_dtype in [np.int16]
        return (x * ((2**15)-1)).astype(np.int16)

    @staticmethod
    def get_pack_type(sample_width):
        if sample_width == 2:
            return 'h'
        elif sample_width == 4:
            return 'l'
        else:
            raise Exception('Sample width ' + str(sample_width))

    @staticmethod
    def get_sample_width(x: np.ndarray):
        if x.dtype in [np.int16]:
            return 2
        elif x.dtype in [np.int32]:
            return 4
        else:
            raise Exception('Cannot derive sample width from numpy dtype "' + str(x.dtype) + '"')

    def __init__(
            self,
            chunk: int = 1024,
            logger: Logging = None,
    ):
        self.chunk = chunk
        self.logger = logger if logger is not None else logging.getLogger()

        self.base_64 = Base64(logger=self.logger)
        return

    def normalize_audio_data(self, x: np.ndarray) -> np.ndarray:
        # we need to normalize audio data to range [-1, +1] before play back
        if x.dtype in [np.int16]:
            amplitude = (2**15) - 1
        elif x.dtype in [np.int32]:
            amplitude = (2**31) - 1
        else:
            raise Exception('Data type "' + str(x.dtype) + '"')
        self.logger.info(
            'Amplitude ' + str(amplitude) + ', max ' + str(np.max(x)) + ', min ' + str(np.min(x)) + ': ' + str(x)
        )
        return x / amplitude

    def record_voice(
            self,
            sample_rate: int = 16000,
            sample_width: int = 2,
            channels: int = 1,
            record_secs: float = 10.,
            stop_stddev_thr: float = 0.0,
            save_file_path: str = None,
    ):
        import pyaudio
        pa = pyaudio.PyAudio()
        self.logger.info(
            'Trying to open audio channel for recording, sample rate ' + str(sample_rate)
            + ', sample width ' + str(sample_width) + ', channels ' + str(channels)
        )
        stream = pa.open(
            format = self.get_pyaudio_type(sample_width=sample_width),
            channels = channels,
            rate = sample_rate,
            input = True,
            frames_per_buffer = self.chunk,
        )

        self.logger.info("* recording")

        frames = []
        np_frames = []
        secs_per_chunk = self.chunk / sample_rate
        n_chunks_Xsecs_no_activity = int(2 / secs_per_chunk)
        self.logger.info('Seconds per chunk ' + str(secs_per_chunk))

        history_mean_amplitude = []
        for i in range(0, int(record_secs / secs_per_chunk)):
            # data is of type <bytes>
            # if 1 channels, sample width 2, then each sample will have 2 bytes. So if chunk is 1024,
            # data will be 2048 length.
            # if 2 channels, sample width 2, each sample interleaved as 4 bytes, data will be 4096 length
            # Channel samples are interleaved [s1c1, s1c2, s2c1, s2c2, s3c1, s3c2,... ]
            data = stream.read(self.chunk)
            self.logger.info(
                'Read chunk of ' + str(secs_per_chunk) + ', data type "' + str(type(data))
                + '", bytes length ' + str(len(data))
            )
            frames.append(data)
            # Convert bytes to numpy int16 type
            x = np.frombuffer(data, dtype=self.get_numpy_type(sample_width=sample_width))
            x_mean_amplitude = np.mean(np.abs(x))
            history_mean_amplitude.append(x_mean_amplitude)
            self.logger.info(
                'Chunk #' + str(i+1) + ", length " + str(self.chunk) + ', mean amplitude ' + str(x_mean_amplitude)
                + ', min value ' + str(np.min(x)) + ', max value ' + str(np.max(x))
            )
            np_frames.append(x.tolist())
            if len(history_mean_amplitude) >= n_chunks_Xsecs_no_activity:
                # Check standard deviation of last X second chunks mean amplitudes
                running_std = np.std(np.array(history_mean_amplitude[-n_chunks_Xsecs_no_activity:]))
                self.logger.debug('Running standard deviation ' + str(running_std))
                if stop_stddev_thr > 0:
                    if running_std < stop_stddev_thr:
                        self.logger.info(
                            'Stop recording. Stddev ' + str(running_std) + ' dropped below threshold '
                            + str(stop_stddev_thr)
                        )
                        break

        stream.stop_stream()
        stream.close()
        pa.terminate()
        self.logger.info('Recording done.')

        x = np.array(np_frames).flatten()
        # plt.plot(x)
        # plt.show()

        self.logger.info(
            'Write wav, min/max values ' + str(np.min(x)) + '/' + str(np.max(x))
            + ', data type "' + str(x.dtype) + '", output to file "' + str(save_file_path) + '"'
        )

        if save_file_path is not None:
            self.logger.info('Sample width or int size ' + str(sample_width))
            self.save_wav_to_file(
                save_file_path = save_file_path,
                sample_width = sample_width,
                sample_rate = sample_rate,
                channels = channels,
                # save bytes data of int16
                frames = frames,
            )
        return

    def save_wav_to_file(
            self,
            save_file_path: str,
            sample_width,
            sample_rate: int,
            channels: int,
            frames: list[bytes],
    ):
        USE_SCIPY = False
        if not USE_SCIPY:
            wf = wave.open(save_file_path, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
        else:
            # TODO not working
            raise Exception('TODO not working yet')
            # wav.write(
            #     filename = save_file_path,
            #     rate = sample_rate,
            #     data = x,
            # )
        self.logger.info(
            'Wav successfully saved to file "' + str(save_file_path) + '", sample rate ' + str(sample_rate)
            + ', sample width ' + str(sample_width) + ', channels ' + str(channels)
        )

    def play_voice(self, file_path: str):
        sample_rate, n_channels, x = self.read_audio(file_path_or_base64_str=file_path)
        x = self.normalize_audio_data(x=x)
        self.logger.info('Read audio file "' + str(file_path) + '" as numpy array of shape ' + str(x.shape))
        import sounddevice
        sounddevice.play(data=x, samplerate=sample_rate, blocking=True)
        return x

    def play_voice_stream_from_file(
            self,
            file_path_or_base64_str: str,
    ):
        sample_rate, channels, x = self.read_audio(file_path_or_base64_str=file_path_or_base64_str)
        # we need to normalize audio data to range [-1, +1] before play back
        # x = self.normalize_audio_data(x=x)
        return self.play_voice_stream(
            sample_rate = sample_rate,
            channels = channels,
            x = x,
        )

    def play_voice_stream(
            self,
            sample_rate: int,
            channels: int,
            x: np.ndarray,
    ):
        sample_width = self.get_sample_width(x=x)
        # we need to normalize audio data to range [-1, +1] before play back
        # x = self.normalize_audio_data(x=x)

        import pyaudio
        p = pyaudio.PyAudio()
        stream = p.open(
            # struct.pack later will use 4 byte float
            format = self.get_pyaudio_type(sample_width=sample_width),
            channels = channels,
            rate = sample_rate,
            output = True,
            frames_per_buffer = self.chunk,
        )
        self.logger.info('Read audio file as numpy array of shape ' + str(x.shape) + ': ' + str(x))
        N = self.chunk
        # dt = N / sample_rate
        pack_format = self.get_pack_type(sample_width=sample_width)
        # Loop by chunk
        for i in range(math.ceil(len(x) / N)):
            i_start = i*N
            i_end = min(len(x), (i+1)*N)
            # multiple channels
            if channels > 1:
                # Channel samples are interleaved [s1c1, s1c2, s2c1, s2c2, s3c1, s3c2,...
                data_part = [samp for samp in x[i_start:i_end].flatten()]
                # data_part = []
                # # Channel samples must exist block by block, not interleaved by sample
                # for ch in range(channels):
                #     data_part_channel = [samp[ch] for samp in x[i_start:i_end]]
                #     data_part = data_part + data_part_channel
            else:
                data_part = [samp for samp in x[i_start:i_end]]

            self.logger.debug('Data part #' + str(i) + ', length ' + str(len(data_part)) + ': ' + str(data_part))
            data_part_b = b''.join(struct.pack(pack_format, samp) for samp in data_part)
            self.logger.debug('Data part (b) #' + str(i) + ', length ' + str(len(data_part_b)) + ': ' + str(data_part_b))
            # the stream will know how to play the different parts continuously at the right speed,
            # even if we don't send them at regular intervals
            stream.write(frames = data_part_b)
            self.logger.debug('Done write to stream #' + str(i))
            # demo some lag when random value > 0.8, the sleep time becomes longer than 1 chunk
            # time.sleep(dt * (1 + np.random.rand() - 0.5))

        stream.close()
        p.terminate()
        return

    def read_audio(
            self,
            # can be file path (.wav, .ogg) or ogg base 64
            file_path_or_base64_str: str,
    ):
        audio_format = re.sub(pattern=".*[.]", repl="", string=file_path_or_base64_str).lower()
        if audio_format in ['ogg', 'wav']:
            b64_bytes = None
            file_path = file_path_or_base64_str
        else:
            b64_bytes = self.base_64.decode(s=file_path_or_base64_str)
            file_path = None
            audio_format = 'ogg'

        is_b64 = b64_bytes is not None
        self.logger.info('Is base 64 string "' + str(file_path_or_base64_str) + '": ' + str(is_b64))

        if is_b64:
            sample_rate, n_channels, np_data = self.read_ogg_bytes(ogg_bytes=b64_bytes)
        elif audio_format in ['ogg']:
            sample_rate, n_channels, np_data = self.read_ogg_file(ogg_file_path=file_path)
        elif audio_format in ['wav']:
            sample_rate, n_channels, np_data = self.read_wav(wav_file_path=file_path)
        else:
            raise Exception('Cannot recognize audio file extension "' + str(file_path) + '"')
        return sample_rate, n_channels, np_data

    def read_wav(self, wav_file_path: str):
        sample_rate, data = wav.read(filename=wav_file_path)
        np_wav = np.array(data, dtype=data.dtype)
        n_channels = 1 if (np_wav.ndim <= 1) else np_wav.shape[-1]
        self.logger.info(
            'wav audio file "' + str(wav_file_path) + '", shape ' + str(np_wav.shape) + ', data type ' + str(data.dtype)
            + ', max value ' + str(np.max(np_wav)) + ', min ' + str(np.min(np_wav))
            + ', sample rate ' + str(sample_rate) + ', channels ' + str(n_channels)
        )
        return sample_rate, n_channels, np_wav

    def read_ogg_bytes(self, ogg_bytes: bytes):
        import librosa
        data_normalized, sample_rate = librosa.load(io.BytesIO(ogg_bytes))
        self.logger.info(
            'Read from ogg bytes of length ' + str(len(ogg_bytes)) + ', data type ' + str(type(data_normalized))
            + ', data length ' + str(data_normalized.shape) + ', sample rate ' + str(sample_rate)
            + ', min value ' + str(np.min(data_normalized)) + ', max value ' + str(np.max(data_normalized))
        )
        return self.read_ogg_data(data_normalized=data_normalized, sample_rate=sample_rate)

    def read_ogg_file(self, ogg_file_path: str):
        import librosa
        data_normalized, sample_rate = librosa.load(ogg_file_path)
        self.logger.info(
            'Read from ogg file "' + str(ogg_file_path) + '", data type ' + str(type(data_normalized))
            + ', data length ' + str(data_normalized.shape) + ', sample rate ' + str(sample_rate)
            + ', min value ' + str(np.min(data_normalized)) + ', max value ' + str(np.max(data_normalized))
        )
        return self.read_ogg_data(data_normalized=data_normalized, sample_rate=sample_rate)

    def read_ogg_data(self, data_normalized: np.ndarray, sample_rate: int):
        np_ogg = (data_normalized * ((2**15)-1)).astype(np.int16)
        n_channels = 1 if (np_ogg.ndim <= 1) else np_ogg.shape[-1]
        # metadata = model.sttWithMetadata(int16)
        self.logger.info(
            'ogg audio data shape ' + str(np_ogg.shape) + ', max value ' + str(np.max(np_ogg))
            + ', min ' + str(np.min(np_ogg))
        )
        return sample_rate, n_channels, np_ogg

    def save_b64str_to_ogg(self, s_b64: str, file_path: str):
        audio_bytes = self.base_64.decode(s=s_b64)
        with open(file_path, "wb") as binary_file:
            # Write bytes to file
            binary_file.write(audio_bytes)
        return

    def save_np_audio_to_wav(self, x: np.ndarray, sample_rate, channels, save_file_path):
        self.logger.info(
            'x min value ' + str(np.min(x)) + ', max value ' + str(np.max(x))
        )
        N = self.chunk
        # dt = N / sample_rate
        frames = []

        sample_width = self.get_sample_width(x=x)
        pack_format = self.get_pack_type(sample_width=sample_width)
        for i in range(math.ceil(len(x) / N)):
            i_start = i*N
            i_end = min(len(x), (i+1)*N)
            data_part = x[i_start:i_end]
            data_part_b = b''.join(struct.pack(pack_format, samp) for samp in data_part)
            frames.append(data_part_b)
        self.save_wav_to_file(
            save_file_path = save_file_path,
            sample_width = sample_width,
            sample_rate = sample_rate,
            channels = channels,
            frames = frames,
        )
        return

    def up_down_sample(
            self,
            x: np.ndarray,
            from_sample_rate: int,
            to_sample_rate: int,
    ):
        ratio_to_from = to_sample_rate / from_sample_rate
        new_interval = 1 / ratio_to_from
        new_len = math.floor(len(x) / new_interval)
        self.logger.info(
            'Data length ' + str(len(x)) + ', from rate ' + str(from_sample_rate) + ' to rate ' + str(to_sample_rate)
            + ' new interval ' + str(new_interval) + ', new length ' + str(new_len)
        )
        sample_indexes = np.array([round(v * new_interval) for v in range(new_len)])
        sample_indexes = [i for i in sample_indexes if i < len(x)]
        return x[sample_indexes]


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    v = Voice2Array(logger=lgr)
    f_wav = "sample_5s.wav"

    if not os.path.exists(f_wav):
        v.record_voice(
            sample_rate = 8000,
            sample_width = 2,
            channels = 2,
            record_secs = 5.,
            save_file_path = f_wav,
            stop_stddev_thr = 20.,
        )

    sample_1_s64 = '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'
    for f in [
        sample_1_s64,
        f_wav,
    ]:
        v.play_voice_stream_from_file(file_path_or_base64_str=f)

    # v.play_voice_stream(file_path=f_wav, channels=1)
    # exit(0)
    x = np.arange(100)
    for from_rate, to_rate in [(44100, 8000), (8000, 44100)]:
        x_new = v.up_down_sample(x=x, from_sample_rate=from_rate, to_sample_rate=to_rate)
        lgr.info(x_new)
    exit(0)
