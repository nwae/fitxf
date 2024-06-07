import logging
import re
import os


class StringVar:

    PAT_VARNAME = r'[a-zA-Z._:\-]+'

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def replace_vars(
            self,
            value,
            # any function that takes the first argument as the variable name, returns value of variable
            get_var_value_func,
            # minimize use of stressful regex for user, just simple string
            var_string_start = '${',
            var_string_end   = '}',
            # avoid using regex if possible, it is over over-complicated & hard to maintain as always
            use_regex = False,
            # only required if use_regex = True
            varname_pattern  = PAT_VARNAME,
    ):
        if use_regex:
            # Put a bracket on every character
            pat_front = ''.join(['['+c+']' for c in var_string_start])
            pat_back = ''.join(['['+c+']' for c in var_string_end])
            # Final pattern to detect
            pattern = pat_front + r'(' + varname_pattern + ')' + pat_back
            self.logger.debug(
                'Using this pattern "' + str(pattern) + '" for value "' + str(value) + '"'
            )
        else:
            pattern = None

        # Index required if not using regex
        i = 0
        # Keep track of how many variables in the value
        count_var = 0
        # start/end index of variable found
        start, end = 0, 0
        # If not using regex
        mode_look_for_start_var = True

        while True:
            if i > len(value):
                break
            if not use_regex:
                try:
                    if mode_look_for_start_var:
                        if value[i:(i + len(var_string_start))] == var_string_start:
                            # Found start of var
                            mode_look_for_start_var = False
                            # include start variable brackets/etc
                            start = i
                            # fast forward to look for variable name
                            i += len(var_string_start)
                        else:
                            i += 1
                        continue

                    # If arrive here means we are looking for variable ending

                    if value[i:(i + len(var_string_end))] == var_string_end:
                        # Found end of var
                        mode_look_for_start_var = True
                        count_var += 1
                        # include end variable brackets/etc
                        end = i + len(var_string_end)
                        # fast forward to look for new variable
                        i = end + 1
                    else:
                        # keep looking for variable ending
                        i = i + 1
                        continue
                except:
                    # no more start or end to look for
                    break
            else:
                # re.match() will not work with this pattern, only re.search() or re.finditer() will work
                m = re.search(pattern=pattern, string=value)
                if not m:
                    self.logger.debug(
                        'No more variable pattern matches at iteration #' + str(i)
                        + ' for string value "' + str(value) + '"'
                    )
                    break
                count_var += 1
                self.logger.debug(
                    'Found variable #' + str(count_var) + ' from value "' + str(value) + '": ' + str(m)
                )
                start, end = m.span()[0], m.span()[1]

            # Take into account start characters (e.g. "${") and ending characters (e.g. "}")
            start_actual, end_actual = start + len(var_string_start), end - len(var_string_end)

            if start_actual >= end_actual:
                self.logger.error(
                    'Incorrect start ' + str(start_actual) + ' <= end ' + str(end_actual)
                    + ', regex pattern "' + str(pattern)
                    + '", for string "' + str(value) + '"'
                )
                break

            # Maintain simplicity, try not to use regex
            varname = value[start_actual:end_actual]
            # varname_with_brackets = value[start:end]
            # varname = re.sub(pattern=pattern, repl='\\1', string=varname_with_brackets)

            var_value = get_var_value_func(varname)

            self.logger.debug(
                'Var name "' + str(varname) + '" variable value = "' + str(var_value) + '"'
            )

            value = value[:start] + str(var_value) + value[end:]
            self.logger.debug(
                'Updated Value now after replacing variable #' + str(count_var) + ' "' + str(value) + '"'
            )
        return value


class StringVarUnitTest:
    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        sv = StringVar()
        os.environ['XXX'] = 'abc888'

        def example_get_var_value(*args):
            return {
                'FOLDER':       '/usr/local',
                'SECT::FOLDER': '/usr/share',
                'SECRET':       'abc123',
                'ENV_PASSWORD': os.environ['XXX'],
            }[args[0]]

        for string, var_str_start, var_str_end, expected_val in (
                ('1 ${FOLDER} xx',          '${',    '}', '1 /usr/local xx'),
                ('2 ${SECT::FOLDER} xx',    '${',    '}', '2 /usr/share xx'),
                ('3 $ENV{ENV_PASSWORD} xx', '$ENV{', '}', '3 abc888 xx'),
        ):
            for use_re in (True, False,):
                val = sv.replace_vars(
                    value = string,
                    var_string_start = var_str_start,
                    var_string_end = var_str_end,
                    get_var_value_func = example_get_var_value,
                    use_regex = use_re,
                    varname_pattern = StringVar.PAT_VARNAME if use_re else None,
                )
                assert val == expected_val,\
                    'Use regex "' + str(use_re) + '" Value get "' + str(val) + '" not equal to expected value "' \
                    + str(expected_val) + '"'
        print('ALL TESTS PASSED OK')
        return


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    StringVarUnitTest().test()
    exit(0)
