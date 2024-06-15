# -*- coding: utf-8 -*-
import flask
import logging
import os
from fitxf.math.utils.CmdLine import CmdLine


#
# In order to make gunicorn work, make sure in your derived class to add this line
#    from ub.api_rest.RestFlaskMultiThread import app as application
# This is because gunicorn will look for the object "application" to call methods to
# start the Flask server
#
app = flask.Flask(__name__)

class RestFlaskMultiThread:

    JSON_ENCODING = 'utf-8'

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        # Flask app
        self.app = app
        # self.app_test.config['DEBUG'] = False
        self.params_cl = self.get_cmdline_params()
        self.port = self.params_cl['port']
        self.logger.info('Port from commend line "' + str(self.port) + '"')
        self.init_rest_urls()
        return

    def get_cmdline_params(self):
        params = CmdLine.get_cmdline_params()
        for p, p_env_var, v_default in [('port', 'PORT', 7777)]:
            if p not in params.keys():
                # not in command line, try env var instead
                v_envvar_or_default = os.environ.get(p_env_var, v_default)
                params[p] = v_envvar_or_default
                self.logger.warning(
                    'Param "' + str(p) + '" not in command line, default to value from env var or default "'
                    + str(v_envvar_or_default) + '"'
                )
        return params

    def init_rest_urls(self):
        raise Exception('Must be implemented by derived class')
        # Example http://localhost:7777/?text=I%20love%20Vader%20Sentiment
        # @self.app_test.route('/', methods=['POST','GET'])
        # def call():
        #     request_json = self.get_request_json(method=flask.request.method)
        #     m = flask.request.method
        #     # Do something
        #     return json.dumps({
        #         'text': text,
        #         'sentiment': res,
        #         'req': str(request_json),
        #     })

    def get_request_json(
            self,
            method = None
    ):
        method = flask.request.method if method is None else method
        req_json = {}
        if method == 'GET':
            for param_name in flask.request.args:
                req_json[param_name] = flask.request.args[param_name]
            return req_json
        else:
            #obj = json.loads(flask.request.json, encoding=AggregateServer.JSON_ENCODING)
            return flask.request.json

    def get_param(self, param_name, method=None):
        method = flask.request.method if method is None else method
        if method == 'GET':
            if param_name in flask.request.args:
                return str(flask.request.args[param_name])
            else:
                return None
        else:
            try:
                val = flask.request.json[param_name]
                return val
            except Exception as ex:
                logging.critical(
                    str(self.__class__) + ': No param name "' + str(param_name) + '" in request: ' + str(ex)
                )
                return None

    def get_params(self, param_list, method=None):
        method = flask.request.method if method is None else method
        data = {}
        for p in param_list:
            data[p] = self.get_param(param_name=p, method=method)
            self.logger.info('Read param "' + str(p) + '": "' + str(data[p]) + '"')

        return data.values()

    def get_remote_ip_from_nginx(
            self,
            key = None,
    ):
        key = 'X-Real-IP' if key is None else key
        x_real_ip = flask.request.headers.get(key, None)
        self.logger.debug('Real IP from key "' + str(key) + '" = "' + str(x_real_ip) + '"')
        return x_real_ip

    def get_remote_port_from_nginx(
            self,
            key = None,
    ):
        key = 'X-Real-Port' if key is None else key
        x_real_port = flask.request.headers.get(key, None)
        self.logger.debug('Real Port from key "' + str(key) + '" = "' + str(x_real_port) + '"')
        return x_real_port

    def run(
            self,
    ):
        try:
            self.app.run(
                host = '0.0.0.0',
                port = self.port
            )
        except Exception as ex:
            self.logger.critical(
                str(self.__class__) + ': Could not start REST Server, got exception: ' + str(ex)
            )

    def stop(
            self
    ):
        self.logger.critical(
            str(self.__class__) + ': Stop aggregate server command received.'
        )
        # TODO Stop server
        func = flask.request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        return


"""
Example class to derive from this RestFlaskMultiThread class.
Then instantiate the class outside of main for gunicorn.
"""
# class ExampleRest(RestFlaskMultiThread):
#     def __init__(self):
#         super().__init__()
#
#     def init_rest_urls(self):
#         # Example http://localhost:7777/?text=yo
#         @self.app.route('/', methods=['POST','GET'])
#         def call():
#             request_json = self.get_request_json(method=flask.request.method)
#             # Do something
#             return json.dumps({
#                 'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 'port': self.port,
#                 'request': str(request_json),
#             })
# __inst = ExampleRest()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # If running from gunicorn, no need to run this (will be started by
    # gunicorn instead with host/port given to gunicorn on the command line)
    #__inst.run()
