import logging
import datetime
import threading


class Profiling:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        self.time_records = {}
        self.__mutex = threading.Lock()
        return

    def start(self):
        return datetime.datetime.now()

    def stop(self):
        return datetime.datetime.now()

    def get_time_dif_secs(
            self,
            start,
            stop,
            decimals=10,
    ):
        diftime = (stop - start)
        diftime = round(diftime.days*86400 + diftime.seconds + diftime.microseconds / 1000000, decimals)
        return diftime

    def start_time_profiling(
            self,
            id,
            force_reset = False,
            max_records = 1000,
    ):
        try:
            self.__mutex.acquire()

            dt_now = self.start()
            new_record = {'start': dt_now, 'stop': dt_now, 'dif_secs': -1.}

            if force_reset or (id not in self.time_records):
                self.time_records[id] = [new_record]
            else:
                is_non_empty = len(self.time_records[id]) >= 1
                # Means stuck from previous measurement, never call record after start
                if is_non_empty and (self.time_records[id][-1]['dif_secs']) < 0.:
                    # self.logger.warning('Did not call record_time_profiling() after start')
                    self.time_records[id][-1] = new_record
                    self.logger.warning('Previous time record "' + str(id) + '" no end time, replace with new record')
                else:
                    self.time_records[id].append(new_record)
                    self.logger.debug('Append new time recording to "' + str(id) + '"')

            if len(self.time_records[id]) > max_records:
                self.time_records[id] = self.time_records[id][1:]
        finally:
            self.__mutex.release()

    def record_time_profiling(
            self,
            id,
            msg,
            decimals = 10,
            logmsg = False,
    ):
        try:
            self.__mutex.acquire()
            self.time_records[id][-1]['stop'] = self.stop()
            dur_s = self.get_time_dif_secs(
                start = self.time_records[id][-1]['start'],
                stop  = self.time_records[id][-1]['stop'],
                decimals = decimals,
            )
            self.time_records[id][-1]['dif_secs'] = dur_s

            # Clean up those with no stop times
            # self.time_records[id] = [rec for rec in self.time_records[id] if rec['dif_secs'] > 0.]

            dur_cum_s = sum([rec['dif_secs'] for rec in self.time_records[id]])
            mean_secs = round(dur_cum_s / len(self.time_records[id]), decimals)
            rps_mean = round(1/(mean_secs+0.0000000001), 3)
            if logmsg:
                self.logger.info(
                    '[' + str(id) + '] Profile speed: ' + str(msg) + ' took ' + str(dur_s)
                    + 's, cumulative mean ' + str(mean_secs) + 's (' + str(rps_mean) + ' rps) from total records '
                    + str(len(self.time_records[id]))
                )
            return dur_s, dur_cum_s
        finally:
            self.__mutex.release()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import time
    pf = Profiling()
    def f():
        time.sleep(0.01)

    for i in range(10):
        pf.start_time_profiling(id='test', max_records=5)
        pf.start_time_profiling(id='test', max_records=5)
        pf.start_time_profiling(id='test', max_records=5)

        f()
        pf.record_time_profiling(
            id = 'test',
            msg = str(i),
            logmsg = True,
        )

    exit(0)
