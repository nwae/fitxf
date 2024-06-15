import logging
import threading
import time
import re
import numpy as np
from datetime import datetime
from fitxf.math.utils.Logging import Logging


#
# Proper wrapper to handle locking of mutexes
#   - will not lock forever
#   - when actual race condition occurs, allows easy debugging of who is locking who
#
class Lock:

    def __init__(
            self,
            mutex_names,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()

        self.__mutexes = {}
        for mtx_name in mutex_names:
            self.__mutexes[mtx_name] = {
                'mutex': threading.Lock(),
                'locked_by': None,
                'locked_since': datetime(year=2000, month=1, day=1),
            }
            self.logger.info('Created mutex lock for "' + str(mtx_name) + '"')
        return

    def is_locked(
            self,
            mutex,
    ):
        return self.__mutexes[mutex]['mutex'].locked()

    def acquire_mutexes(
            self,
            id,
            # list of mutex names, or plain string
            mutexes,
            try_count = 10,
            try_sleep_secs = 1.0,
            fake_race_condition_for_unit_test = False,
    ):
        mutexes = [mutexes] if type(mutexes) not in [list, tuple] else mutexes

        # if any exception occurs midway, we release all locked mutexes
        mutexes_locked_in_progress = []

        for mtx_name in mutexes:
            mtx = self.__mutexes[mtx_name]['mutex']
            fail_count = 0
            while True:
                fail_count += 1
                if fail_count >= try_count:
                    # release everything locked in this loop before
                    self.release_mutexes(mutexes=mutexes_locked_in_progress)
                    info = '"' + str(id) + '" wait for mutex "' + str(mtx_name) + '", tried ' + str(fail_count) + 'x.'
                    raise Exception('FAILED TO ACQUIRE mutex ' + str(info))

                if mtx.locked():
                    locked_by = self.__mutexes[mtx_name]['locked_by']
                    locked_since = self.__mutexes[mtx_name]['locked_since']
                    try:
                        tdiff = datetime.now() - locked_since
                    except Exception as ex_race_condition:
                        self.logger.info(
                            'Possible race condition, mutex must have been released by another thread while in this ' +
                            'loop setting "locked_since" to ' + str(locked_since) + ': ' + str(ex_race_condition)
                        )
                        time.sleep(try_sleep_secs)
                        continue
                    tdiff = tdiff.days*86400 + tdiff.seconds + tdiff.microseconds/1000000

                    info = '"' + str(id) + '" waiting for mutex "' + str(mtx_name) + '", tried ' + str(fail_count) \
                        + ' times, locked by "' + str(locked_by) + '", locked since "' + str(locked_since) \
                        + '" for ' + str(tdiff) + ' secs.'

                    self.logger.info(info + ' Trying again in ' + str(try_sleep_secs) + ' secs..')
                    time.sleep(try_sleep_secs)
                    continue
                else:
                    if fake_race_condition_for_unit_test:
                        mtx.acquire()
                    # timeout here only to detect race condition, since we are already sure mutex is free
                    res = mtx.acquire(timeout=1)
                    self.logger.debug('Mutex acquired or not = ' + str(res))
                    if not res:
                        # continue trying
                        time.sleep(try_sleep_secs)
                        continue
                    else:
                        mutexes_locked_in_progress.append(mtx_name)

                    self.__mutexes[mtx_name]['locked_by'] = id
                    self.__mutexes[mtx_name]['locked_since'] = datetime.now()

                    self.logger.info(
                        '"' + str(id) + '" acquired mutex "' + str(mtx_name) + '" successfully'
                    )
                    break
        return

    # When releasing a list of mutexes, make sure try/except so that none will be locked forever
    def release_mutexes(
            self,
            # list of mutex names, or plain string
            mutexes,
    ):
        mutexes = [mutexes] if type(mutexes) not in [list, tuple] else mutexes

        for mtx_name in mutexes:
            mtx = self.__mutexes[mtx_name]['mutex']
            try:
                if mtx.locked():
                    id_holder = self.__mutexes[mtx_name]['locked_by']
                    mtx.release()
                    self.__mutexes[mtx_name]['locked_by'] = None
                    self.__mutexes[mtx_name]['locked_since'] = None
                    self.logger.info(
                        'Mutex name "' + str(mtx_name) + '" released by "' + str(id_holder)
                        + '" successfully ' + str(mtx)
                    )
                else:
                    self.logger.debug('Mutex is not locked, not releasing ' + str(mtx))
            except Exception as ex:
                self.logger.error('Error releasing mutex ' + str(mtx) + ': ' + str(ex))


class LockUnitTest:
    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.lock1 = 'l1'
        self.lock2 = 'l2'
        self.lock_manager = Lock(logger=self.logger, mutex_names=[self.lock1, self.lock2])

        self.race_threads = []
        for i in range(10):
            thrd = threading.Thread(
                target = self.thread_for_test,
                args = [i, self.lock1],
            )
            self.race_threads.append(thrd)

        return

    def thread_for_test(
            self,
            thread_id,
            lock_name,
    ):
        sleep_secs = np.random.randint(low=5,high=10) / 10
        # Only thread id number 8 gets the lock
        sleep_secs = 0. if thread_id == 8 else sleep_secs
        time.sleep(sleep_secs)

        have_lock = False
        try:
            self.lock_manager.acquire_mutexes(
                id = thread_id,
                mutexes = lock_name,
                try_count = 2,
                try_sleep_secs = 0.5
            )
            have_lock = True
        except Exception as ex:
            self.logger.info('Thread for test exception ' + str(ex))

        test_res = have_lock==True if thread_id == 8 else have_lock==False
        assert test_res, 'Thread id "' + str(thread_id) + '", got lock "' + str(have_lock) + '"'
        return

    def race_to_get_lock(
            self,
    ):
        self.lock_manager.release_mutexes(mutexes=[self.lock1, self.lock2])
        for i in range(len(self.race_threads)):
            self.logger.info('Starting race thread #' + str(i))
            self.race_threads[i].start()
        return

    def test(self):
        lock1 = self.lock1
        lock2 = self.lock2
        obj = self.lock_manager

        obj.acquire_mutexes(id='1', mutexes=lock1, try_count=2, try_sleep_secs=0.5)
        assert obj.is_locked(mutex=lock1), 'Lock "' + str(lock1) + '" locked'

        # Test unable to obtain lock
        exc_msg = ''
        try:
            obj.acquire_mutexes(id='1', mutexes=lock1, try_count=2, try_sleep_secs=0.5)
        except Exception as ex:
            self.logger.info('Occured exception: ' + str(ex))
            exc_msg = str(ex)
        assert re.match(pattern='FAILED TO ACQUIRE mutex.*', string=exc_msg), \
            'Expect exception when cannot obtain lock1, but got exception message: ' + str(exc_msg)

        # Test unable to obtain lock by faking race condition
        obj.release_mutexes(mutexes=lock1)
        assert not obj.is_locked(mutex=lock1), 'Unlocked lock1'
        exc_msg = ''
        try:
            obj.acquire_mutexes(
                id='1', mutexes=lock1, try_count=2, try_sleep_secs=0.5,
                fake_race_condition_for_unit_test=True,
            )
        except Exception as ex:
            self.logger.info('Occured exception: ' + str(ex))
            exc_msg = str(ex)
        assert re.match(pattern='.*FAILED TO ACQUIRE mutex .*', string=exc_msg), \
            'Expect exception race condition lock1, but got exception message: ' + str(exc_msg)

        # Test able to obtain lock again
        obj.release_mutexes(mutexes=lock1)
        assert not obj.is_locked(mutex=lock1), 'Unlocked lock1'
        obj.acquire_mutexes(id='1', mutexes=[lock1, lock2], try_count=2, try_sleep_secs=0.5)
        assert obj.is_locked(mutex=lock1)
        assert obj.is_locked(mutex=lock2)

        print('Start real race to get lock')
        self.race_to_get_lock()

        for i in range(len(self.race_threads)):
            self.race_threads[i].join()

        print('ALL TESTS PASSED OK')
        return


if __name__ == '__main__':
    LockUnitTest(
        logger = Logging.get_default_logger(log_level=logging.INFO, propagate=False),
    ).test()
    exit(0)
