import logging
import pickle
import os
import threading
import random
import time
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.LockF import LockFile


#
# Use cases:
#   1. RAM scalability
#      If you need to cache a 500M object in memory, you can use this class which will cache it in file
#      and quickly retrieve it when needed only.
#   2. Shared resource between multi-worker/threaded processes running on same server
#      If you have multiple workers and threads that need access to a shared object, this is it.
#
class ObjectPersistence:

    ATOMIC_UPDATE_MODE_ADD = 'add'
    ATOMIC_UPDATE_MODE_REMOVE = 'remove'

    DEFAULT_WAIT_TIME_LOCK_FILE = 2

    def __init__(
            self,
            default_obj,
            obj_file_path,
            lock_file_path,
            logger = None,
    ):
        self.default_obj = default_obj
        # fixed and don't change, these file paths
        self.obj_file_path = obj_file_path
        self.lock_file_path = lock_file_path
        self.logger = logger if logger is not None else logging.getLogger()

        self.lockf = LockFile(
            lock_file_path = self.lock_file_path,
            logger = self.logger,
        )
        for f in [self.obj_file_path, self.lock_file_path]:
            self.logger.info('File exists "' + str(f) + '" = ' + str(os.path.exists(f)))

        # Read once from storage
        self.obj = None
        self.obj = self.read_persistent_object()
        self.logger.info(
            'New object persistence created from "' + str(self.obj_file_path)
            + '", lock file "' + str(self.lock_file_path) + '" as: ' + str(self.obj)
        )
        return

    def __assign_default_object_copy(self):
        try:
            self.obj = self.default_obj.copy()
        except Exception as ex_copy:
            errmsg = 'Failed to assign copy of default object: ' + str(ex_copy) \
                     + '. This will potentially modify default object!'
            self.logger.error(errmsg)
            self.obj = self.default_obj

    #
    # Makes sure that read/write happens in one go
    #
    def atomic_update(
            self,
            # Only dict type supported, will add a new items to dict
            new_items,
            # 'add' or 'remove'
            mode,
            max_wait_time_secs = DEFAULT_WAIT_TIME_LOCK_FILE,
    ):
        if not self.lockf.acquire_file_cache_lock(
                max_wait_time_secs = max_wait_time_secs,
        ):
            self.logger.critical(
                'Atomic update could not serialize to "' + str(self.obj_file_path)
                + '", could not obtain lock to "' + str(self.lock_file_path) + '".'
            )
            return False

        try:
            self.obj = self.__deserialize_object_from_file()
            if self.obj is None:
                self.logger.warning(
                    'Atomic update cannot deserialize from file "' + str(self.obj_file_path) + '"'
                )
                self.__assign_default_object_copy()

            self.logger.debug(
                'Cache object type ' + str(type(self.obj))
            )

            if type(self.obj) is dict:
                if type(new_items) is not dict:
                    self.logger.error(
                        'Atomic updates to dict type must be a dict item! Got item type "'
                        + str(type(new_items)) + '": ' + str(new_items)
                    )
                    return False
                for k in new_items.keys():
                    if mode == self.ATOMIC_UPDATE_MODE_ADD:
                        self.obj[k] = new_items[k]
                        self.logger.info('Atomic update added new item ' + str(new_items))
                    elif mode == self.ATOMIC_UPDATE_MODE_REMOVE:
                        if k in self.obj.keys():
                            del self.obj[k]
                            self.logger.info('Atomic update removed item ' + str(new_items))
                    else:
                        self.logger.error('Atomic update invalid mode "'+ str(mode) + '"!')
                        return False
            elif type(self.obj) is list:
                # In this mode, new items is only ONE item, otherwise you get unexpected results
                if mode == ObjectPersistence.ATOMIC_UPDATE_MODE_ADD:
                    self.obj.append(new_items)
                    self.logger.info('Atomic update added new items ' + str(new_items))
                elif mode == ObjectPersistence.ATOMIC_UPDATE_MODE_REMOVE:
                    if new_items in self.obj:
                        self.obj.remove(new_items)
                        self.logger.error(
                            'Atomic updates removed item "' + str(new_items)
                            + str(type(self.obj)) + '"!'
                        )
            else:
                self.logger.error('Atomic updates not supported for cache type "' + str(type(self.obj)) + '"!')
                return False

            res = self.__serialize_object_to_file(obj=self.obj)
            if not res:
                self.logger.error(
                    'Atomic update new item ' + str(new_items)
                    + ' fail, could not serialize update to file "' + str(self.obj_file_path) + '"'
                )
                return False
        except Exception as ex:
            self.logger.error(
                'Atomic update new item ' + str(new_items)
                + ' fail. Exception update to file "' + str(self.obj_file_path) + '": ' + str(ex)
            )
            return False
        finally:
            self.lockf.release_file_cache_lock()
        return True

    #
    # Wrapper write function to applications
    #
    def update_persistent_object(
            self,
            new_obj,
            max_wait_time_secs = DEFAULT_WAIT_TIME_LOCK_FILE
    ):
        self.obj = new_obj
        res = self.serialize_object_to_file(
            obj = self.obj,
            max_wait_time_secs = max_wait_time_secs,
        )
        if not res:
            self.logger.error(
                'Error writing to file "' + str(self.obj_file_path)
                + '", lock file "' + str(self.lock_file_path) + '" for data: ' + str(self.obj)
            )
        return res

    #
    # Wrapper read function for applications
    #
    def read_persistent_object(
            self,
            max_wait_time_secs = DEFAULT_WAIT_TIME_LOCK_FILE,
    ):
        obj_read = self.deserialize_object_from_file(
            max_wait_time_secs = max_wait_time_secs,
        )
        if obj_read is not None:
            self.obj = obj_read
        else:
            self.logger.warning(
                'None object from file "' + str(self.obj_file_path)
                + '", lock file "' + str(self.lock_file_path) + '". Returning memory object.'
            )

        if type(self.default_obj) != type(self.obj):
            self.logger.warning(
                'Object read from file "' + str(self.obj_file_path)
                + '" of type "' + str(type(self.obj)) + ', different with default obj type "'
                + str(type(self.default_obj)) + '". Setting obj back to default obj.'
            )
            self.__assign_default_object_copy()
            self.logger.info('Assigned default copy to object')

            # Need to write back to correct the object type
            self.update_persistent_object(new_obj=self.obj)
            self.logger.info('Updated persistent object with new object ' + str(self.obj))

        return self.obj

    def serialize_object_to_file(
            self,
            obj,
            max_wait_time_secs = DEFAULT_WAIT_TIME_LOCK_FILE,
    ):
        if not self.lockf.acquire_file_cache_lock(
                max_wait_time_secs = max_wait_time_secs
        ):
            self.logger.critical(
                'Could not serialize to "' + str(self.obj_file_path) + '", could not obtain lock to "'
                + str(self.lock_file_path) + '".'
            )
            return False

        try:
            res = self.__serialize_object_to_file(obj=obj)
            return res
        except Exception as ex:
            self.logger.critical(
                'Exception deserializing/loading object from file "'
                + str(self.obj_file_path) + '". Exception message: ' + str(ex) + '.'
            )
            return False
        finally:
            self.lockf.release_file_cache_lock()

    # lock must already be obtained before calling any functions with "__"
    def __serialize_object_to_file(
            self,
            obj,
    ):
        fhandle = open(
            file = self.obj_file_path,
            mode = 'wb'
        )
        pickle.dump(
            obj      = obj,
            file     = fhandle,
            protocol = pickle.HIGHEST_PROTOCOL
        )
        fhandle.close()
        self.logger.debug(
            'Object "' + str(obj)
            + '" serialized successfully to file "' + str(self.obj_file_path) + '"'
        )
        return True

    def deserialize_object_from_file(
            self,
            max_wait_time_secs = DEFAULT_WAIT_TIME_LOCK_FILE,
    ):
        if not self.lockf.acquire_file_cache_lock(
            max_wait_time_secs = max_wait_time_secs,
        ):
            self.logger.critical(
                'Could not deserialize from "' + str(self.obj_file_path) + '", could not obtain lock to "'
                + str(self.lock_file_path) + '".'
            )
            return None

        try:
            return self.__deserialize_object_from_file()
        except Exception as ex:
            self.logger.critical(
                'Exception deserializing/loading object from file "'
                + str(self.obj_file_path) + '". Exception message: ' + str(ex) + '.'
            )
            return None
        finally:
            self.lockf.release_file_cache_lock()

    # lock must already be obtained before calling any functions with "__"
    def __deserialize_object_from_file(
            self,
    ):
        fhandle = open(
            file = self.obj_file_path,
            mode = 'rb',
        )
        obj = pickle.load(
            file = fhandle,
        )
        fhandle.close()
        self.logger.debug(
            'Object "' + str(obj) + '" deserialized successfully from file "' + str(self.obj_file_path)
            + '" to ' + str(obj) + '.'
        )
        return obj


#
# We do extreme testing on ObjectPersistence, by running hundreds of threads updating
# a single file.
# We then check back if there are any errors.
#
class LoadTest:

    DELETED_KEYS_SET = set()

    def __init__(
            self,
            obj_file_path,
            lock_file_path,
            max_wait_time_secs,
            n_threads,
            count_to,
            logger = None,
    ):
        self.obj_file_path = obj_file_path
        self.lock_file_path = lock_file_path
        self.max_wait_time_secs = max_wait_time_secs
        self.n_threads = n_threads
        self.count_to = count_to
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def target(self, thread_no):
        cache = ObjectPersistence(
            default_obj = {},
            obj_file_path = self.obj_file_path,
            lock_file_path = self.lock_file_path,
            logger = self.logger,
        )
        for i in range(self.count_to):
            # To ensure all values are unique, "count_to" is the mathematical base
            value = self.count_to*thread_no + i
            cache.atomic_update(
                new_items = {value: threading.get_ident()},
                mode = ObjectPersistence.ATOMIC_UPDATE_MODE_ADD
            )
            self.logger.info('Value=' + str(value) + ' +++ ' + str(cache.read_persistent_object()))
            # Delete something at random
            if random.choice([0,1]) == 1:
                obj = cache.read_persistent_object()
                key_choices = list(obj.keys())
                if len(key_choices) > 0:
                    random_key_to_delete = random.choice(key_choices)
                    cache.atomic_update(
                        new_items = {random_key_to_delete: obj[random_key_to_delete]},
                        mode = ObjectPersistence.ATOMIC_UPDATE_MODE_REMOVE
                    )
                    LoadTest.DELETED_KEYS_SET.add(random_key_to_delete)
                    self.logger.info('DELETED ' + str(random_key_to_delete))
            time.sleep(random.uniform(0.005,0.010))
        self.logger.info('***** THREAD ' + str(threading.get_ident()) + ' DONE ' + str(self.count_to) + ' COUNTS')

    def test(self):
        threads_list = []
        n_sum = 0
        for i in range(self.n_threads):
            n_sum += self.count_to
            thr = threading.Thread(
                target = self.target,
                args = [i],
            )
            threads_list.append(thr)
            self.logger.info(str(i) + '. New thread "' + str(threads_list[i].getName()) + '" count ' + str(self.count_to))
        expected_values = []
        for i in range(len(threads_list)):
            for j in range(self.count_to):
                expected_values.append(self.count_to*i + j)
            thr = threads_list[i]
            self.logger.info('Starting thread ' + str(i))
            thr.start()

        for thr in threads_list:
            thr.join()

        cache = ObjectPersistence(
            default_obj = {},
            obj_file_path = self.obj_file_path,
            lock_file_path = self.lock_file_path,
            logger = self.logger,
        )
        self.logger.info('********* Final Object File: ' + str(cache.read_persistent_object()))
        values = list(cache.read_persistent_object().keys())
        self.logger.info('Added Keys: ' + str(values))
        self.logger.info('Deleted Keys: ' + str(LoadTest.DELETED_KEYS_SET))
        self.logger.info('Total Added = ' + str(len(values)))
        self.logger.info('Total Deleted = ' + str(len(LoadTest.DELETED_KEYS_SET)))
        values.sort()
        expected_values = list( set(expected_values) - LoadTest.DELETED_KEYS_SET )
        expected_values.sort()
        test_pass = values == expected_values
        self.logger.info('PASS = ' + str(test_pass))
        self.logger.info('Values:   ' + str(values))
        self.logger.info('Expected: ' + str(expected_values))
        return


class UnitTestObjectPersistence:

    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def __remove_files(self, obj_file_path, lock_file_path):
        try:
            os.remove(obj_file_path)
        except Exception:
            pass
        try:
            os.remove(lock_file_path)
        except Exception:
            pass

    def test(self):
        test_dir = '.'
        obj_file_path = test_dir + '/loadtest.objpers.obj'
        lock_file_path = test_dir + '/loadtest.objpers.obj.lock'

        #
        # If we don't do this the test will fail as the upcoming threads,
        # many of them will try to delete the file due to timeout, and will
        # accidentally delete one created by another thread
        #
        self.__remove_files(obj_file_path=obj_file_path, lock_file_path=lock_file_path)

        #
        # Test 1: Load Test
        #

        LoadTest(
            obj_file_path  = obj_file_path,
            lock_file_path = lock_file_path,
            count_to = 5,
            n_threads = 10,
            max_wait_time_secs = 10,
            logger = self.logger,
        ).test()

        #
        # Test 2:
        #

        obj_file_path = test_dir + '/objPersTest.b'
        lock_file_path = test_dir + '/lock.objPersTest.b'
        self.__remove_files(obj_file_path=obj_file_path, lock_file_path=lock_file_path)

        obj_pers_1 = ObjectPersistence(
            default_obj = [],
            obj_file_path = obj_file_path,
            lock_file_path = lock_file_path,
            logger = self.logger,
        )

        objects = [
            {
                'a': [1, 2, 3],
                'b': 'test object'
            },
            # Empty objects
            [],
            {},
            88,
            'eighty eight'
        ]

        for obj in objects:
            obj_pers_1.serialize_object_to_file(
                obj = obj,
            )

            b = obj_pers_1.deserialize_object_from_file()
            self.logger.info(str(b))
            ok = obj == b
            assert ok

        obj_file_path = test_dir + '/objPersTestAtomicUpdate.d'
        lock_file_path = test_dir + '/lock.objPersTestAtomicUpdate.d'
        self.__remove_files(obj_file_path=obj_file_path, lock_file_path=lock_file_path)
        obj_pers_2 = ObjectPersistence(
            default_obj    = {},
            obj_file_path  = obj_file_path,
            lock_file_path = lock_file_path
        )
        self.logger.info(
            obj_pers_2.atomic_update(new_items={1: 'hana', 2: 'dul'}, mode=ObjectPersistence.ATOMIC_UPDATE_MODE_ADD)
        )
        ok = obj_pers_2.read_persistent_object() == {1: 'hana', 2: 'dul'}
        assert ok
        self.logger.info(obj_pers_2.atomic_update(new_items={1: 'hana'}, mode=ObjectPersistence.ATOMIC_UPDATE_MODE_REMOVE))
        ok = obj_pers_2.read_persistent_object() == {2: 'dul'}
        assert ok
        self.logger.info(obj_pers_2.atomic_update(new_items={3: 'set'}, mode=ObjectPersistence.ATOMIC_UPDATE_MODE_ADD))
        ok = obj_pers_2.read_persistent_object() == {2: 'dul', 3: 'set'}
        assert ok

        # Purposely write wrong type
        obj_pers_2.update_persistent_object(new_obj=[1,2,3])
        x = ObjectPersistence(
            default_obj    = {},
            obj_file_path  = obj_file_path,
            lock_file_path = lock_file_path
        )
        ok = x.read_persistent_object() == {}
        assert ok
        print('OK TESTS PASSED')
        return


if __name__ == '__main__':
    lgr = Logging.get_default_logger(log_level=logging.INFO, propagate=False)

    obj_fp = 'test_tensor'
    lock_fp = 'lock_tensor'
    for f in [obj_fp, lock_fp]:
        # Need to remove, otherwise there will be lingering objects in the file
        os.remove(obj_fp) if os.path.exists(obj_fp) else 1

    obj_pers = ObjectPersistence(
        default_obj = [],
        obj_file_path = obj_fp,
        lock_file_path = lock_fp,
        logger = lgr,
    )
    UnitTestObjectPersistence(logger=lgr).test()

    import numpy as np
    objects = np.random.random(size=(7,3)).tolist()
    [print(o) for o in objects]

    for o in objects:
        obj_pers.atomic_update(
            new_items = o,
            mode = ObjectPersistence.ATOMIC_UPDATE_MODE_ADD,
            max_wait_time_secs = 2,
        )
        print('added item: ' + str(o))

    read_back = obj_pers.deserialize_object_from_file()
    print('Read back:')
    [print(o) for o in read_back]
    assert read_back == objects, 'Read back\n' + str(read_back) + '\nnot same with original\n' + str(objects)
    print('ALL OK')

    exit(0)
