from multiprocessing import Process, Queue
import numpy as np
import fnmatch
import inspect


class AsyncProcessPoolLight:

    def __init__(self):
        pass

    def init(self):


        self._compute_workers = []
        #self._compute_queue   = Queue()
        #self._result_queue    = Queue()
        

    def runJob(self,job_id, handler=None, io=None, args=(), kwargs={}, collect_result=True, serial=False):

        # Check if handler is a function or a method
        if inspect.isfunction(handler):
            handler_id, method = id(handler), None
        elif inspect.ismethod(handler):
            instance = handler.__self__
            handler_id, method = id(instance), handler.__func__


        jobitem = dict(job_id=job_id, handler=(instance, handler_id, method),
                       collect_result=collect_result,
                       args=args, kwargs=kwargs)


        if serial:
            self._dispatch_job(jobitem)
        else:
            self._compute_workers.append(Process(target=self._dispatch_job, args=(jobitem,)))
            self._compute_workers[-1].start()

    def _dispatch_job(self, jobitem):
        
        try:
            job_id, args, kwargs = [jobitem.get(attr) for attr in
                                                         ["job_id", "args", "kwargs"]]
            instance, handler_id, method = jobitem["handler"]
            
            args = [ arg for arg in args ]

            if method is None:
                # Call the object directly
                result = handler(*args, **kwargs)
            else:
                result = method(instance, *args, **kwargs)


        except KeyboardInterrupt:
            raise
        
        
    def awaitJobResult(self):
        for job in self._compute_workers:
            job.join()




APPL = None

def _init_default():
    global APPL
    if APPL is None:
        APPL = AsyncProcessPoolLight()
        APPL.init()

_init_default()