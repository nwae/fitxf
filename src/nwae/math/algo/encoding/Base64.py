# -*- coding: utf-8 -*-
import logging
import numpy as np
from base64 import b64encode, b64decode
from nwae.math.utils import Logging


class Base64:

    B64_ALLOWED_CHARSET = [chr(x) for x in range(ord('A'), ord('Z')+1)] + \
                          [chr(x) for x in range(ord('a'), ord('z')+1)] + \
                          [chr(x) for x in range(ord('0'), ord('9')+1)] + \
                          ['+', '/', '=']

    def __init__(
            self,
            text_encoding = 'utf-8',
            logger = None,
    ):
        self.text_encoding = text_encoding
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def encode(
            self,
            b,
    ):
        b = bytes(b, self.text_encoding) if type(b) is str else b
        e_bytes = b64encode(s=b)
        return e_bytes.decode(encoding=self.text_encoding)

    def decode(
            self,
            s,
    ):
        d_bytes = b64decode(s.encode(encoding=self.text_encoding))
        return d_bytes

    #
    # Note: If a string consists only of base64 characters [a-zA-Z0-9+/=], and have length
    # 0 (mod 3) then it will be wrongly recognized as a base64 string.
    #
    def is_base_64_string(
            self,
            s,
    ):
        s = str(s)
        zero_mod3 = (len(s) % 3 == 0)
        is_b64_charset = True
        for char in s:
            if char not in self.B64_ALLOWED_CHARSET:
                is_b64_charset = False
                break

        return zero_mod3 & is_b64_charset

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
    ) -> np.ndarray:
        # Step 1: Convert base 64 string to base 64 bytes
        s64_b = s64.encode('utf-8')
        # Step 2: Convert base 64 bytes to actual bytes
        actual_bytes = b64decode(s64_b)
        # Step 3: Finally convert to numpy array from base 64 bytes
        return np.frombuffer(actual_bytes, dtype=data_type)


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
            (np.array([1.23, 4.55, 7.43, 555.42]), 'rkfhehSu8z8zMzMzMzMSQLgehetRuB1Aj8L1KFxbgUA='),
            (np.array([[1.23, 4.55], [7.43, 555.42]]), 'rkfhehSu8z8zMzMzMzMSQLgehetRuB1Aj8L1KFxbgUA='),
        )
        for x, expected_b64_str in np_tests:
            e_str = b64.encode_numpy_array_to_base64_string(x=x, data_type=x.dtype)
            d_np = b64.decode_base64_string_to_numpy_array(s64=e_str, data_type=x.dtype)
            self.logger.info(
                'Original object <<' + str(x) + '>> encoded to <<' + str(e_str)
                + '>>, decoded back as <<' + str(d_np) + '>>'
            )
            assert e_str == expected_b64_str, 'Encoded b64 "' + str(e_str) + '" not "' + str(expected_b64_str) + '"'
            assert np.sum((d_np - x.flatten()) ** 2) < 0.0000000001, 'Decoded b64 "' + str(d_np) + '" not "' + str(x) + '"'

        print('B64 TESTS PASSED OK')
        return


if __name__ == '__main__':
    Base64UnitTest(logger=Logging.get_default_logger(log_level=logging.INFO, propagate=False)).test()
    exit(0)
