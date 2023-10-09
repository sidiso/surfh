import psutil
import concurrent.futures
import fnmatch
import inspect
from collections import OrderedDict
import time
import os
import sys

import pickle


global jobsHandler
jobsHandler = {}

class WorkerProcessError(Exception):
    pass

class AsyncProcessPool():

    def __init__(self, ncpu):

        print("#############################################")
        print("########## Init CFAPP #########################")
        print("#############################################")

        self.compute_executor   = concurrent.futures.ProcessPoolExecutor(max_workers=ncpu)
        self.io_executor        = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._compute_queue = []
        self._io_queue      = []
        self.tot = 0
        self._jobs  = []
        self._jobsBDA = []
        self._jobs_counter = []
        self._jobs_event   = []

        self.ncpu = ncpu


    def init(self, ncpu, affinity, parent_affinity, num_io_processes, verbose, pause_on_start):
        self.__init__(ncpu)

    def setCPU(self, ncpu):
        self.ncpu = ncpu
        self.compute_executor   = concurrent.futures.ProcessPoolExecutor(max_workers=ncpu)


    def terminate(self):
        pass

    def shutdown(self):
        self.compute_executor.shutdown()
        self.io_executor.shutdown()

    def start(self):
        self.compute_executor   = concurrent.futures.ProcessPoolExecutor(max_workers=self.ncpu)
        self.io_executor        = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._compute_queue = []
        self._io_queue      = []

        self.tot = 0

        self._jobs  = []
        self._jobsBDA = []
        self._jobs_counter = []
        self._jobs_event   = []


    def restart(self):
        self.compute_executor.shutdown()
        self.io_executor.shutdown()

        self.compute_executor   = concurrent.futures.ProcessPoolExecutor(max_workers=self.ncpu)
        self.io_executor        = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._compute_queue = []
        self._io_queue      = []

        self.tot = 0

        self._jobs  = []
        self._jobsBDA = []
        self._jobs_counter = []
        self._jobs_event   = []


    def runJob (self, job_id, handler=None, io=None, args=(), kwargs={},
                event=None, counter=None,
                singleton=False, collect_result=True,
                serial=False):


        # Check if handler is a function or a method
        if inspect.isfunction(handler):
            handler_id, method = id(handler), None
        elif inspect.ismethod(handler):
            instance = handler.__self__
            handler_id, method = id(instance), handler.__func__

        # TOTEST
        jobsHandler[id(handler)] = handler

        if serial:
            if method is None:
                handler(*args, **kwargs)
            else:
                method(instance, *args, **kwargs)
        else:
            future = None
            if io is not None:
                if method is None:
                    future = self.io_executor.submit(handler, *args, **kwargs)
                else:
                    future = self.io_executor.submit(method, instance, *args, **kwargs)
                self._io_queue.append(future)
                self._jobs.append((job_id, future))
            else:
                if method is None:
                    future = self.compute_executor.submit(handler, *args, **kwargs)
                else:
                    future = self.compute_executor.submit(method, instance, *args, **kwargs)

                self._compute_queue.append(future)

                self._jobs.append((job_id, future))

    def awaitJobResults(self, jobspecs, progress=None, timing=None):
        if type(jobspecs) is str:
            jobspecs = [jobspecs]

        fs = []
        result_values = []
        job_results = OrderedDict()
        #print("Iter on jobspecs")
        for jobspec in jobspecs:
            for job_id, f in self._jobs:
                if fnmatch.fnmatch(job_id, jobspec):
                    fs.append(f)
                    job_results[jobspec] = (1, [])


        total_jobs = complete_jobs = 0
        total_jobs = len(fs)
        if progress:
            #pBAR = ProgressBar(Title="  "+progress)
            #pBAR.render(complete_jobs,(total_jobs or 1))
            while ((complete_jobs != total_jobs) and concurrent.futures.wait(fs, timeout=2)):
                complete_jobs = 0
                for i in fs:
                    if i.done():
                        complete_jobs +=1
                #pBAR.render(complete_jobs,(total_jobs or 1))
        else:
            concurrent.futures.wait(fs)

        result_values = []
        for item in fs:
            if item.exception() is not None:
                print("CF DEBUG : ITEM EXCEPTION IS NOT NONE for jobspecs %s, It is :"%jobspecs)
                try:
                    data = item.result()
                except Exception as exc:
                    print("Item exception from result is ", exc)
                    print(item.result())
                else:
                    print("Else ", item.exception())
                
            if item.result() is not None:
                result_values.append(item.result())

        self._compute_queue = [x for x in self._compute_queue if not x.done()]
        self._io_queue      = [x for x in self._io_queue if not x.done()]
        for jobspec in jobspecs:
            self._jobs  = [(job_id, f) for (job_id, f) in self._jobs if not fnmatch.fnmatch(job_id, jobspec) ]

        #print("Just leaving awaitJobReasults for", jobspecs)
        return result_values[0] if len(result_values) == 1 else result_values


 

CFAPP = None
def _init_default():
    global CFAPP
    if CFAPP is None:
        print("CFAPP Initialized with %d cpu"%(psutil.cpu_count()))
        #CFAPP = AsyncProcessPool(psutil.cpu_count())
        CFAPP = AsyncProcessPool(16)

_init_default()

def init(ncpu=None, affinity=None, parent_affinity=0, num_io_processes=1, verbose=0, pause_on_start=False):
    global CFAPP
    CFAPP.init(ncpu, affinity, parent_affinity, num_io_processes, verbose, pause_on_start=pause_on_start)
