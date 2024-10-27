import logging
import threading
import os
import time
from datetime import datetime
from fitxf.math.utils.Env import Env
from fitxf.math.utils.Logging import Logging


#
# Instantiate this class instead of your desired class, then call get_singleton()
#
class Singleton:

    # Default global singleton store, if not provided by user
    SINGLETON_STORE_BY_CLASSTYPE = {}
    SINGLETON_LAST_UPDATE_BY_CLASSTYPE = {}

    # A gloabl mutex means you cannot call get singleton inside a get singleton of a same class,
    # means singleton inside singleton of same class not allowed
    MUTEX_SINGLETON_STORE_BY_CLASSTYPE = {}

    def __init__(
            self,
            class_type,
            logger = None,
    ):
        self.class_type = class_type
        self.logger = logger if logger is not None else logging.getLogger()

        # No need to care about race condition here, at most the latter lock object will just overwrite the previous
        # Worst case is that 2 singletons are stored on 2 different storage duee to race condition,
        # of which the earlier one will become irrelevant later
        if not self.class_type in self.SINGLETON_STORE_BY_CLASSTYPE.keys():
            self.SINGLETON_STORE_BY_CLASSTYPE[self.class_type] = {}
            self.SINGLETON_LAST_UPDATE_BY_CLASSTYPE[self.class_type] = {}
            self.logger.info('Created new store for class type "' + str(self.class_type) + '"')

        # No need to care about race condition here, at most the latter lock object will just overwrite the previous
        if not self.class_type in self.MUTEX_SINGLETON_STORE_BY_CLASSTYPE.keys():
            self.MUTEX_SINGLETON_STORE_BY_CLASSTYPE[self.class_type] = threading.Lock()
            self.logger.info('Created new mutex for class type "' + str(self.class_type) + '"')

        self.singleton_store = self.SINGLETON_STORE_BY_CLASSTYPE[self.class_type]
        self.last_update_dict = self.SINGLETON_LAST_UPDATE_BY_CLASSTYPE[self.class_type]
        return

    # def get_derived_key(self, user_key_id):
    #     return str(self.class_type) + '___' + str(user_key_id)

    def get_singleton(
            self,
            # user provided key, if need multiple singletons for the same class.
            # user's responsibility to make sure no naming clashes between IDs of different class types
            key_id,
            *args,
    ):
        try:
            self.logger.info('Trying to acquire singleton mutex for class type ' + str(self.class_type))
            # Use different mutex, so that a singleton of one class can instantiate a singleton of a different class
            self.MUTEX_SINGLETON_STORE_BY_CLASSTYPE[self.class_type].acquire()
            return self.__get_singleton_thread_safe(
                key_id, *args
            )
        finally:
            self.MUTEX_SINGLETON_STORE_BY_CLASSTYPE[self.class_type].release()
            self.logger.info('Singleton mutex released for class type ' + str(self.class_type))

    def __get_singleton_thread_safe(
            self,
            # The reason we need a user provided ID is that sometimes for a given class type,
            # the user may want to return different instance if the __init__() parameters are
            # different. And we let user have that control.
            # user's responsibility to make sure no naming clashes between IDs of different class types
            key_id,
            *args,
    ):
        # Update last accessed time
        self.last_update_dict[key_id] = datetime.now()
        self.logger.info(
            'Last active for singleton "' + str(key_id) + '" set to ' + str(self.last_update_dict[key_id])
        )

        # derived_key = self.get_derived_key(user_key_id=key_id)
        if key_id in self.singleton_store.keys():
            self.logger.info(
                'Returning already available singleton class class type "' + str(self.class_type)
                + '", key id "' + str(key_id) + '": ' + str(self.singleton_store[key_id])
            )
            type_existing = type(self.singleton_store[key_id])
            assert type_existing == self.class_type, \
                'Wrong type for singleton "' + str(type_existing) + '" with "' + str(type(self.class_type)) + '"'
            return self.singleton_store[key_id]
        else:
            self.singleton_store[key_id] = self.class_type(*args)
            self.logger.info(
                'Created new singleton class for class type "' + str(self.class_type)
                + '", key id "' + str(key_id) + '": ' + str(self.singleton_store[key_id])
            )
            return self.singleton_store[key_id]


class SingletonMemoryMgmt:

    def __init__(
            self,
            class_type,
            delete_secs_inactive: float,
            thread_check_every_n_secs:float = 10.,
            logger = None,
    ):
        self.class_type = class_type
        self.delete_secs_inactive = delete_secs_inactive
        self.thread_check_every_n_secs = thread_check_every_n_secs
        self.signal_stop_thread = False
        self.logger = logger if logger is not None else logging.getLogger()

        if self.class_type not in Singleton.SINGLETON_STORE_BY_CLASSTYPE.keys():
            Singleton.SINGLETON_STORE_BY_CLASSTYPE[self.class_type] = {}
            Singleton.SINGLETON_LAST_UPDATE_BY_CLASSTYPE[self.class_type] = {}
            Singleton.MUTEX_SINGLETON_STORE_BY_CLASSTYPE[self.class_type] = threading.Lock()

        self.__store = Singleton.SINGLETON_STORE_BY_CLASSTYPE[self.class_type]
        self.__last_update = Singleton.SINGLETON_LAST_UPDATE_BY_CLASSTYPE[self.class_type]
        self.__mutex = Singleton.MUTEX_SINGLETON_STORE_BY_CLASSTYPE[self.class_type]

        self.memory_mgmt_records = {}
        self.thread = threading.Thread(target=self.run_memory_mgmt_thread)
        self.thread.start()
        self.logger.info('Started memory management thread for Singletons.')
        return

    def update_memory_mgmt_params(
            self,
            delete_secs_inactive,
    ):
        self.delete_secs_inactive = delete_secs_inactive
        self.logger.info('Set delete secs inactive to new value ' + str(self.delete_secs_inactive) + 's')
        return

    def __delete_key_id(self, key_id):
        del self.__store[key_id]
        del self.__last_update[key_id]
        # force garbage collection
        gc.collect()
        # Mutex can't be deleted
        # del self.__mutex[key_id]
        self.logger.info(
            'Deleted key id "' + str(key_id) + '", remaining keys: ' + str(self.__store.keys())
        )
        return

    def clear_all(self):
        try:
            self.__mutex.acquire()
            for key_id in self.__store.keys():
                self.__delete_key_id(key_id=key_id)
        finally:
            self.__mutex.release()

    def run_memory_mgmt_thread(self):
        while True:
            if self.signal_stop_thread:
                self.logger.info(
                    'Memory mgmt for "' + str(self.class_type) + '" stop signal received, ending thread..'
                )
                break
            time.sleep(self.thread_check_every_n_secs)
            try:
                self.__mutex.acquire()
                keys_to_delete = []
                for key_id in self.__store.keys():
                    self.logger.debug('Checking key id "' + str(key_id) + '"')
                    last_active = self.__last_update[key_id]
                    diff = datetime.now() - last_active
                    secs_non_active = round(diff.days * 86400 + diff.seconds + diff.microseconds / 1000000, 4)
                    if secs_non_active > self.delete_secs_inactive:
                        self.logger.info(
                            'Delete inactive for ' + str(secs_non_active) + 's, key id "' + str(key_id)
                            + '", set object for deletion..'
                        )
                        keys_to_delete.append(key_id)
                    else:
                        self.logger.debug(
                            'Keep inactive for ' + str(secs_non_active) + 's, key id "' + str(key_id) + '"'
                        )
                for key_id in keys_to_delete:
                    self.__delete_key_id(key_id=key_id)
            except Exception as ex:
                self.logger.error('Error in memory mgmt thread ' + str(ex))
            finally:
                self.__mutex.release()

    def stop_memory_mgmt(self):
        self.signal_stop_thread = True
        return


class SingletonUnitTest:
    class SampleTestClass:
        def __init__(self, a, b, throw_exception=False, logger=None):
            self.a = a
            self.b = b
            self.logger = logger if logger is not None else logging.getLogger()
            self.address = id(self)
            self.logger.info('Initialized new object ' + str(self) + ' at ' + str(hex(self.address)))
            if throw_exception:
                raise Exception('Flag throw exception set')
            return

    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        # start memory mgmt
        self.deleted_after_secs = 10000.
        self.test_class_type = SingletonUnitTest.SampleTestClass

        self.mm = SingletonMemoryMgmt(
            class_type = self.test_class_type,
            delete_secs_inactive = self.deleted_after_secs,
            thread_check_every_n_secs = 0.2,
            logger = self.logger,
        )
        return

    def test(self):
        store = Singleton.SINGLETON_STORE_BY_CLASSTYPE[self.test_class_type]
        sgt = Singleton(
            class_type = self.test_class_type,
            logger = self.logger,
        )

        for i in range(10):
            user_key = str(i % 3)
            throw_exc = True if user_key == '2' else False

            got_exception = False
            try:
                obj = sgt.get_singleton(user_key, i, i + 1, throw_exc, self.logger)
            except Exception as ex:
                self.logger.error('Error get singleton ' + str(ex))
                obj = None
                got_exception = True

            assert throw_exc == got_exception, \
                'Requested to throw exception = ' + str(throw_exc) + ' but got ' + str(got_exception)
            assert len(store) == len(set(store.values())), 'All in store must be unique: ' + str(store)
            self.logger.info('All in store at #' + str(i) + ': ' + str(store))

            # key = sgt.get_derived_key(user_key_id=user_key)
            if not throw_exc:
                assert id(obj) == id(store[user_key]), \
                    'Object key "' + str(user_key) + '" not same ' + str(id(obj)) + ' vs ' + str(id(store[user_key]))
            else:
                assert user_key not in store.keys(), 'User key "' + str(user_key) + '" should not exist'

        sleep_total = 0.
        id_0_before_deletion = id(store['0'])
        while sleep_total < 3.0:
            obj = sgt.get_singleton('0', 0, 1, False, self.logger)
            assert id_0_before_deletion == id(obj), \
                'Must be same cached object ' + str(id(obj)) + ' as old ' + str(id_0_before_deletion)
            time.sleep(1.0)
            sleep_total += 1.

        # Make memory mgmt delete the objects
        self.mm.update_memory_mgmt_params(delete_secs_inactive=1.)

        # After sleep, memory should have been cleaned up
        time.sleep(1.)
        assert len(store) == 0, 'Store should be cleaned up of memory but got ' + str(store)
        self.mm.stop_memory_mgmt()

        #
        # Test that new object will be created since we already deleted the above
        #
        obj = sgt.get_singleton('0', 0, 1, False, self.logger)
        assert id_0_before_deletion != id(obj), \
            'Must be new object created already, but got same ' + str(id(obj)) + ' as old ' + str(id_0_before_deletion)

        print('SINGLETON TESTS PASSED OK')


if __name__ == '__main__':
    e = Env()
    Env.set_env_vars_from_file(env_filepath=e.REPO_DIR + '/.env.fitxf.math.ut')
    lgr = Logging.get_logger_from_env_var()
    SingletonUnitTest(logger=lgr).test()
    exit(0)
