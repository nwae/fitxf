# -*- coding: utf-8 -*-
import logging
import json
import numpy as np
from base64 import b64encode, b64decode
from fitxf.utils import Logging


class Base64:

    B64_ALLOWED_CHARSET = [chr(x) for x in range(ord('A'), ord('Z')+1)] + \
                          [chr(x) for x in range(ord('a'), ord('z')+1)] + \
                          [chr(x) for x in range(ord('0'), ord('9')+1)] + \
                          ['+', '/', '=']

    def __init__(
            self,
            text_encoding: str = 'utf-8',
            logger: logging.Logger = None,
    ):
        self.text_encoding = text_encoding
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def encode(
            self,
            b,
    ) -> str:
        b = bytes(b, self.text_encoding) if type(b) is str else b
        assert type(b) is bytes
        e_bytes = b64encode(s=b)
        b64_str = e_bytes.decode(encoding=self.text_encoding)
        assert type(b64_str) is str
        return b64_str

    def decode(
            self,
            s: str,
    ) -> bytes:
        d_bytes = b64decode(s.encode(encoding=self.text_encoding))
        assert type(d_bytes) is bytes
        return d_bytes

    def is_base_64_string(
            self,
            s: str,
    ):
        try:
            self.decode(s=s)
            return True
        except Exception as ex:
            return False

    def encode_numpy_array_to_base64_string_multidim(
            self,
            x: np.ndarray,
            # can be string like 'float64', or numpy dtype such as np.float64
            data_type = np.float64,
    ) -> str:
        x_shape = x.shape
        b64_str_flattenned = self.encode_numpy_array_to_base64_string(x=x, data_type=data_type)
        return json.dumps({'shape': list(x_shape), 'b64_str': b64_str_flattenned})

    def decode_base64_string_to_numpy_array_multidim(
            self,
            string: str,
            data_type = np.float64,
    ) -> np.ndarray:
        data = json.loads(string)
        x_shape = data['shape']
        b64_str_flattenned = data['b64_str']
        x_flattenned = self.decode_base64_string_to_numpy_array(s64=b64_str_flattenned, data_type=data_type)
        return x_flattenned.reshape(x_shape)

    # Warning: Encoding numpy array to bytes will flatten it to 1-dimensional
    def encode_numpy_array_to_base64_string(
            self,
            x: np.ndarray,
            # can be string like 'float64', or numpy dtype such as np.float64
            data_type = np.float64,
    ) -> str:
        if x.ndim > 1:
            self.logger.warning(
                'Encoding to base 64 will flatten numpy ndim=' + str(x.ndim) + ' to ndim=1.'
            )
        # Step 1: Convert numpy to float64 bytes
        x_tp = x.astype(dtype=data_type)
        x_tp_bytes = x_tp.tobytes()
        # Step 2 & 3: then convert to base 64 bytes, then to base 64 utf-8 string
        return b64encode(x_tp_bytes).decode('utf-8')

    def decode_base64_string_to_numpy_array(
            self,
            s64: str,
            data_type = np.float64,
            # if provided, will automatically discover dtype and ignore data_type above
            float_vector_len: int = 0,
    ) -> np.ndarray:
        # Step 1: Convert base 64 string to base 64 bytes
        s64_b = s64.encode('utf-8')
        # Step 2: Convert base 64 bytes to actual bytes
        actual_bytes = b64decode(s64_b)
        # Step 3: Finally convert to numpy array from base 64 bytes
        dtype_final = data_type
        if float_vector_len > 0:
            # Auto discover dtype
            nbytes_per_flt = int( len(actual_bytes) / float_vector_len )
            map_ = {2: np.float16, 4: np.float32, 8: np.float64}
            assert nbytes_per_flt in map_.keys(), \
                'Cannot auto discover float length, unusual nbytes per float ' + str(nbytes_per_flt)
            if data_type != map_[nbytes_per_flt]:
                dtype_final = map_[nbytes_per_flt]
                self.logger.debug(
                    'Ignore passed in data type ' + str(data_type) + ', auto discovered type ' + str(dtype_final)
                )
        vector = np.frombuffer(actual_bytes, dtype=dtype_final)
        # return to user the desired data type requested, if auto discovered is different
        if data_type != dtype_final:
            self.logger.debug(
                'Converting numpy vector back to requested data type ' + str(data_type) + ' from auto detected type '
                + str(dtype_final)
            )
            return vector.astype(data_type)
        else:
            return vector


class Base64UnitTest():

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        logger = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
        b64 = Base64(logger=logger)
        invalid_b64_strings = ['no b64', '1231234']
        for s in invalid_b64_strings:
            assert b64.is_base_64_string(s=s) is False, 'String "' + str(s) + '" should not be a base64 string.'

        #
        # Test string encoding/decoding
        #
        tests = (
            ('string to be encoded', 'c3RyaW5nIHRvIGJlIGVuY29kZWQ='),
            ('한국 인터넷 라디오 방송국', '7ZWc6rWtIOyduO2EsOuEtyDrnbzrlJTsmKQg67Cp7Iah6rWt'),
        )
        for s_or_np, expected_b64_str in tests:
            e_str = b64.encode(b=s_or_np)
            d_bytes = b64.decode(s=e_str)
            d_str_or_np = d_bytes.decode(encoding=b64.text_encoding)
            self.logger.info(
                'Original object <<' + str(s_or_np) + '>> encoded to <<' + str(e_str)
                + '>>, decoded back as <<' + str(d_str_or_np) + '>>'
            )
            assert e_str == expected_b64_str, 'Encoded b64 "' + str(e_str) + '" not "' + str(expected_b64_str) + '"'
            assert d_str_or_np == s_or_np, 'Decoded b64 "' + str(d_str_or_np) + '" not "' + str(s_or_np) + '"'

        #
        # Test numpy encoding/decoding of bytes to base 64
        #
        np_tests = (
            (
                np.array([1.23, 4.55, 7.43, 555.42], dtype=np.float64), 4,
                'rkfhehSu8z8zMzMzMzMSQLgehetRuB1Aj8L1KFxbgUA=',
            ),
            (
                np.array([[1.23, 4.55], [7.43, 555.42]], dtype=np.float64), 4,
                'rkfhehSu8z8zMzMzMzMSQLgehetRuB1Aj8L1KFxbgUA=',
            ),
        )
        for i, (x, len_flat, expected_b64_str) in enumerate(np_tests):
            x_flat = x.flatten()
            self.logger.info('x flattened to ' + str(x_flat) + ', dtype ' + str(x_flat.dtype))

            e_str = b64.encode_numpy_array_to_base64_string(x=x, data_type=x.dtype)

            d_np = b64.decode_base64_string_to_numpy_array(s64=e_str, data_type=x.dtype)
            self.logger.info(
                'Test #' + str(i) + ' Original object <<' + str(x) + '>> encoded to <<' + str(e_str)
                + '>>, decoded back as <<' + str(d_np) + '>>'
            )
            assert e_str == expected_b64_str, \
                'Test #' + str(i) + ' Encoded b64 "' + str(e_str) + '" not "' + str(expected_b64_str) + '"'
            assert np.sum((d_np - x_flat) ** 2) < 0.0000000001, \
                'Test #' + str(i) + ' Decoded b64 "' + str(d_np) + '" not "' + str(x) + '"'

            # Test auto discover data type given only vector length
            d_np_auto = b64.decode_base64_string_to_numpy_array(
                # purposely give wrong data type
                s64=e_str, data_type=np.float16, float_vector_len=len_flat,
            )
            self.logger.info(
                'Test #' + str(i) + ' for auto discover dtype. Original object <<' + str(x)
                + '>> encoded to <<' + str(e_str) + '>>, decoded back as <<' + str(d_np) + '>>'
            )
            assert d_np_auto.dtype == np.float16
            err = np.sum((d_np_auto - x_flat) ** 2)
            # put smaller tolerance 0.01 since we are decoding back to np.float16 from np.float64
            assert err < 0.01, 'Decoded b64 "' + str(d_np) + '" not "' + str(x) + '", err ' + str(err)

        # Test multidim
        np_tests = (
            (
                np.array([1.23, 4.55, 7.43, 555.42]),
                {'shape': [4], 'b64_str': 'rkfhehSu8z8zMzMzMzMSQLgehetRuB1Aj8L1KFxbgUA='},
            ),
            (
                np.array([[1.23, 4.55], [7.43, 555.42]]),
                {'shape': [2,2], 'b64_str': 'rkfhehSu8z8zMzMzMzMSQLgehetRuB1Aj8L1KFxbgUA='}
            ),
        )
        for x, expected_b64_json in np_tests:
            e_str = b64.encode_numpy_array_to_base64_string_multidim(x=x, data_type=x.dtype)
            d_np = b64.decode_base64_string_to_numpy_array_multidim(string=e_str, data_type=x.dtype)
            self.logger.info(
                'Original object <<' + str(x) + '>> encoded to <<' + str(e_str)
                + '>>, decoded back as <<' + str(d_np) + '>>'
            )
            assert json.loads(e_str) == expected_b64_json, \
                'Encoded b64 json "' + str(e_str) + '" not "' + str(expected_b64_json) + '"'
            assert np.sum((d_np - x) ** 2) < 0.0000000001, \
                'Decoded b64 "' + str(d_np) + '" not "' + str(x) + '"'

        self.logger.info('B64 TESTS PASSED OK')
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)
    Base64UnitTest(logger=lgr).test()

    s="""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"""
    b = Base64(logger=lgr)
    for dty in [np.float16, np.float32, np.float64]:
        x = b.decode_base64_string_to_numpy_array(s64=s, data_type=dty, float_vector_len=384)
        lgr.info(
            's64 len ' + str(len(s)) + ', shape ' + str(x.shape) + ', dtype ' + str(x.dtype)
            + ', n bytes ' + str(x.nbytes)
        )
        s2 = b.encode_numpy_array_to_base64_string(x=x, data_type=np.float16)
        lgr.info('   s64 len for np.float16 ' + str(len(s2)))
    exit(0)
