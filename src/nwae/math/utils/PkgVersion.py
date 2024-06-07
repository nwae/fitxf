import logging
import sys
if sys.version_info[0] + sys.version_info[1]/10 >= 3.8:
    import importlib.metadata as mtdata
else:
    import pkg_resources


# Helper class to manage environments & versions
class PkgVersion:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logging.getLogger() if logger is None else logger
        self.python_ge_3_8 = sys.version_info[0] + sys.version_info[1] / 10 >= 3.8
        self.logger.info('Is python version >= 3.8 = "' + str(self.python_ge_3_8) + '"')
        return

    def get_pkg_version_string(self, pkg_name):
        pkg_ver = mtdata.version(pkg_name) if self.python_ge_3_8 else pkg_resources.get_distribution(pkg_name).version
        return pkg_ver

    def convert_version_to_list(
            self,
            version: str,
            maxlen: int = 0,
    ):
        assert maxlen <= 3
        v_list = version.split(sep='.', maxsplit=3)
        if maxlen > 0:
            [v_list.append('') for _ in range(3-len(v_list))]
            v_list = v_list[0:maxlen]
        return v_list

    def check_package_version(
            self,
            pkg_name: str,
            version: str,
            return_version = False,
    ):
        v_cmp = self.convert_version_to_list(version=version)
        pkg_ver = mtdata.version(pkg_name) if self.python_ge_3_8 else pkg_resources.get_distribution(pkg_name).version
        v_cur = self.convert_version_to_list(
            version = pkg_ver,
            # Compare only based on given version, means if actual version is "7.17.9",
            # given version "7", we only compare major "7" against "7"
            maxlen  = len(v_cmp),
        )
        self.logger.debug('Pkg "' + str(pkg_name) + '" version ' + str(v_cur))
        verdict = tuple(v_cmp)==tuple(v_cur)
        return (verdict, v_cur) if return_version else verdict

    def check_python(
            self,
            version: str,
            return_version = False,
    ):
        v_cmp = self.convert_version_to_list(version=version)
        v_cur = self.convert_version_to_list(
            version = '.'.join([str(sys.version_info.major), str(sys.version_info.minor), str(sys.version_info.micro)]),
            maxlen  = len(v_cmp),
        )
        str(sys.version_info.major), str(sys.version_info.minor), str(sys.version_info.micro)
        self.logger.debug('Python current version: ' + str(v_cur))
        verdict = tuple(v_cmp)==tuple(v_cur)
        return (verdict, v_cur) if return_version else verdict


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pkg = 'numpy'
    print(PkgVersion().check_python(version='3', return_version=True))
    print(PkgVersion().check_python(version='3.8', return_version=True))
    print(PkgVersion().check_python(version='3.10', return_version=True))
    print(PkgVersion().check_python(version='3.8.16', return_version=True))

    print(PkgVersion().check_package_version(pkg_name=pkg, version='1.24', return_version=True))
    print(PkgVersion().check_package_version(pkg_name=pkg, version='1.22', return_version=True))
    print(PkgVersion().check_package_version(pkg_name=pkg, version='1.24.0', return_version=True))
    exit(0)
