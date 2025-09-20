# -*- coding: utf-8 -*-
import sys


class CmdLine:

    @staticmethod
    def get_cmdline_params(
            pv_default = None
    ):
        pv = {}
        if type(pv_default) is dict:
            for k in pv_default.keys():
                pv[k] = pv_default[k]

        args = sys.argv
        for arg in args:
            arg_split = arg.split('=')
            if len(arg_split) == 2:
                param = arg_split[0].lower()
                value = str(arg_split[1]).strip()
                if value != '':
                    pv[param] = value

        return pv


if __name__ == '__main__':
    print(CmdLine.get_cmdline_params())
    print(CmdLine.get_cmdline_params(pv_default={'a':99, 'c':'Test', 'd':888}))
