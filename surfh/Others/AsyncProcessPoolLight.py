from multiprocessing import Process, Queue
from collections import OrderedDict
import numpy as np
import fnmatch
import inspect

import progressbar

class AsyncProcessPoolLight:

    def __init__(self):
        pass

    def init(self):
        self._compute_workers = []
        self._jobs  = []
        

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
            process = Process(target=self._dispatch_job, args=(jobitem,))
            process.start()
            self._compute_workers.append(process)
            self._jobs.append((job_id, process))

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
        
        
    def awaitJobResult(self, jobspecs, progress=None, timing=None):
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

            widgets = [
                    f'\x1b[34;49;1m{jobspecs[0]} : \x1b[0m',
                    progressbar.Percentage(),
                    progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
                    progressbar.widgets.Timer(),
                ]

            with progressbar.ProgressBar(widgets=widgets, max_value=total_jobs) as bar:
                while ((complete_jobs != total_jobs)):
                    #print("Total fs jobs are ", fs)
                    complete_jobs = 0
                    for i in fs:
                        if not i.is_alive():
                            complete_jobs +=1
                    bar.update(complete_jobs)
                    #pBAR.render(complete_jobs,(total_jobs or 1))
        else:
            for job in fs:
                job.join()




        self._compute_workers = [x for x in self._compute_workers if x.is_alive()]
        for jobspec in jobspecs:
            self._jobs  = [(job_id, f) for (job_id, f) in self._jobs if not fnmatch.fnmatch(job_id, jobspec) ]

        


APPL = None

def _init_default():
    global APPL
    if APPL is None:
        APPL = AsyncProcessPoolLight()
        APPL.init()

_init_default()
