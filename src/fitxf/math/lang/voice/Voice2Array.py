import logging
import pyaudio
import wave
import re
import os
import io
import math
import struct
import librosa
import numpy as np
import sounddevice as sd
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
        sample_rate, x = self.read_audio(file_path_or_base64_str=file_path)
        x = self.normalize_audio_data(x=x)
        self.logger.info('Read audio file "' + str(file_path) + '" as numpy array of shape ' + str(x.shape))
        sd.play(data=x, samplerate=sample_rate, blocking=True)
        return x

    def play_voice_stream(
            self,
            file_path_or_base64_str: str,
            channels: int,
    ):
        sample_rate, x = self.read_audio(file_path_or_base64_str=file_path_or_base64_str)
        sample_width = self.get_sample_width(x=x)
        # we need to normalize audio data to range [-1, +1] before play back
        # x = self.normalize_audio_data(x=x)

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
            sample_rate, np_data = self.read_ogg_bytes(ogg_bytes=b64_bytes)
        elif audio_format in ['ogg']:
            sample_rate, np_data = self.read_ogg_file(ogg_file_path=file_path)
        elif audio_format in ['wav']:
            sample_rate, np_data = self.read_wav(wav_file_path=file_path)
        else:
            raise Exception('Cannot recognize audio file extension "' + str(file_path) + '"')
        return sample_rate, np_data

    def read_wav(self, wav_file_path: str):
        sample_rate, data = wav.read(filename=wav_file_path)
        np_wav = np.array(data, dtype=data.dtype)
        self.logger.info(
            'wav audio file "' + str(wav_file_path) + '", shape ' + str(np_wav.shape) + ', data type ' + str(data.dtype)
            + ', max value ' + str(np.max(np_wav)) + ', min ' + str(np.min(np_wav))
            + ', sample rate ' + str(sample_rate)
        )
        return sample_rate, np_wav

    def read_ogg_bytes(self, ogg_bytes: bytes):
        data_normalized, sample_rate = librosa.load(io.BytesIO(ogg_bytes))
        self.logger.info(
            'Read from ogg bytes of length ' + str(len(ogg_bytes)) + ', data type ' + str(type(data_normalized))
            + ', data length ' + str(data_normalized.shape) + ', sample rate ' + str(sample_rate)
            + ', min value ' + str(np.min(data_normalized)) + ', max value ' + str(np.max(data_normalized))
        )
        return self.read_ogg_data(data_normalized=data_normalized, sample_rate=sample_rate)

    def read_ogg_file(self, ogg_file_path: str):
        data_normalized, sample_rate = librosa.load(ogg_file_path)
        self.logger.info(
            'Read from ogg file "' + str(ogg_file_path) + '", data type ' + str(type(data_normalized))
            + ', data length ' + str(data_normalized.shape) + ', sample rate ' + str(sample_rate)
            + ', min value ' + str(np.min(data_normalized)) + ', max value ' + str(np.max(data_normalized))
        )
        return self.read_ogg_data(data_normalized=data_normalized, sample_rate=sample_rate)

    def read_ogg_data(self, data_normalized: np.ndarray, sample_rate: int):
        np_ogg = (data_normalized * ((2**15)-1)).astype(np.int16)
        # metadata = model.sttWithMetadata(int16)
        self.logger.info(
            'ogg audio data shape ' + str(np_ogg.shape) + ', max value ' + str(np.max(np_ogg))
            + ', min ' + str(np.min(np_ogg))
        )
        return sample_rate, np_ogg

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
    f_wav = "sample_5s_y.wav"

    if not os.path.exists(f_wav):
        v.record_voice(
            sample_rate = 8000,
            sample_width = 2,
            channels = 2,
            record_secs = 5.,
            save_file_path = f_wav,
            stop_stddev_thr = 20.,
        )

    for f, chn in [
        # (f_wav, 2),
    ]:
        v.play_voice_stream(file_path_or_base64_str=f, channels=chn)

    # v.play_voice_stream(file_path=f_wav, channels=1)
    # exit(0)
    x = np.arange(100)
    for from_rate, to_rate in [(44100, 8000), (8000, 44100)]:
        x_new = v.up_down_sample(x=x, from_sample_rate=from_rate, to_sample_rate=to_rate)
        lgr.info(x_new)
    exit(0)
